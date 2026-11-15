"""Kill every explicit constraint in the three shipped schemas.

Each mutant breaks one constraint on a synthetic record that otherwise
validates. A constraint is killed when this checker rejects the mutant.
``format`` is not asserted: draft 2020-12 leaves it as an annotation, and
none of these schemas set it. ``additionalProperties: true`` has no
rejecting mutant; an extra key must still validate.

``would_proceed`` stays a boolean. The string ``"false"`` is the type mutant.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"

_ANNOTATIONS = {"$schema", "$id", "$comment", "title", "description", "examples", "default"}
_STRUCTURE = {"properties"}
_CONSTRAINTS = {
    "type",
    "const",
    "enum",
    "required",
    "additionalProperties",
    "pattern",
    "format",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
}
_TYPE_NAMES = {"object", "array", "string", "integer", "number", "boolean", "null"}


class SchemaUnsupported(Exception):
    """A schema keyword is present and this checker would skip it."""


@dataclass(frozen=True)
class Mutant:
    schema_name: str
    path: str
    keyword: str
    detail: str
    instance: object
    permissive: bool = False

    @property
    def label(self) -> str:
        return f"{self.schema_name} {self.path} {self.keyword} {self.detail}"


def _load(name: str) -> dict:
    return json.loads((SCHEMAS / name).read_text(encoding="utf-8"))


def _strict_eq(left: object, right: object) -> bool:
    if type(left) is bool or type(right) is bool:
        return type(left) is bool and type(right) is bool and left == right
    return left == right


def _type_label(value: object) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "boolean"
    if type(value) is int:
        return "integer"
    if type(value) is float:
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return type(value).__name__


def _matches_type(value: object, type_name: str) -> bool:
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "integer":
        return type(value) is int
    if type_name == "number":
        return type(value) in (int, float)
    if type_name == "boolean":
        return type(value) is bool
    if type_name == "null":
        return value is None
    raise SchemaUnsupported(f"unsupported type {type_name}")


def _declared_types(schema: dict) -> list[str]:
    declared = schema["type"]
    names = [declared] if isinstance(declared, str) else declared
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
        raise SchemaUnsupported("type must be a string or a non-empty list of strings")
    return names


def _check_schema(schema: object, path: str = "$") -> None:
    if not isinstance(schema, dict):
        raise SchemaUnsupported(f"{path}: schema must be an object")
    unknown = set(schema) - _ANNOTATIONS - _STRUCTURE - _CONSTRAINTS
    if unknown:
        raise SchemaUnsupported(f"{path}: unsupported keywords {sorted(unknown)}")
    if "additionalProperties" in schema and type(schema["additionalProperties"]) is not bool:
        raise SchemaUnsupported(f"{path}: additionalProperties must be a boolean")
    if "type" in schema:
        for name in _declared_types(schema):
            if name not in _TYPE_NAMES:
                raise SchemaUnsupported(f"{path}: unsupported type {name}")
    if "properties" in schema:
        if not isinstance(schema["properties"], dict):
            raise SchemaUnsupported(f"{path}: properties must be an object")
        for key, sub in schema["properties"].items():
            _check_schema(sub, f"{path}.{key}")
    if "required" in schema:
        required = schema["required"]
        if not isinstance(required, list) or not all(isinstance(key, str) for key in required):
            raise SchemaUnsupported(f"{path}: required must be a list of strings")


def validate(instance: object, schema: dict) -> list[str]:
    """Reject constraint violations. ``format`` is ignored on purpose."""
    _check_schema(schema)
    return _validate(instance, schema, "$")


def _validate(instance: object, schema: dict, path: str) -> list[str]:
    errors: list[str] = []
    if "const" in schema and not _strict_eq(instance, schema["const"]):
        errors.append(f"{path}: expected const {schema['const']!r}")
    if "enum" in schema and not any(_strict_eq(instance, item) for item in schema["enum"]):
        errors.append(f"{path}: {instance!r} is not in the enum")
    if "type" in schema and not any(_matches_type(instance, name) for name in _declared_types(schema)):
        errors.append(f"{path}: expected {_declared_types(schema)}, got {_type_label(instance)}")
        return errors
    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(f"{path}: shorter than minLength {schema['minLength']}")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append(f"{path}: longer than maxLength {schema['maxLength']}")
        if "pattern" in schema and re.search(schema["pattern"], instance) is None:
            errors.append(f"{path}: pattern mismatch")
    if type(instance) in (int, float) and ("minimum" in schema or "maximum" in schema):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: below minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: above maximum {schema['maximum']}")
    describes_object = (
        "properties" in schema or "required" in schema or "additionalProperties" in schema
    )
    if describes_object:
        if not isinstance(instance, dict):
            errors.append(f"{path}: expected object, got {_type_label(instance)}")
            return errors
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}.{key}: required")
        properties = schema.get("properties", {})
        allow_additional = schema.get("additionalProperties", True)
        for key, value in instance.items():
            if key in properties:
                errors.extend(_validate(value, properties[key], f"{path}.{key}"))
            elif allow_additional is False:
                errors.append(f"{path}.{key}: additional property")
    return errors


def _parts(path: str) -> list[str]:
    if path == "$":
        return []
    if not path.startswith("$."):
        raise AssertionError(path)
    return path[2:].split(".")


def _get(instance: object, path: str) -> object:
    cursor = instance
    for part in _parts(path):
        if not isinstance(cursor, dict) or part not in cursor:
            raise AssertionError(f"missing {path}")
        cursor = cursor[part]
    return cursor


def _set(instance: object, path: str, value: object) -> None:
    parts = _parts(path)
    if not parts:
        raise AssertionError("cannot replace the root through _set")
    cursor = instance
    for part in parts[:-1]:
        if not isinstance(cursor, dict):
            raise AssertionError(path)
        cursor = cursor[part]
    if not isinstance(cursor, dict):
        raise AssertionError(path)
    cursor[parts[-1]] = value


def _wrong_type(type_spec: object) -> object:
    names = [type_spec] if isinstance(type_spec, str) else list(type_spec)
    if names == ["boolean"]:
        return "false"
    if names == ["integer"]:
        return True
    if names == ["string"]:
        return 1
    if names == ["object"]:
        return []
    for candidate in ("false", True, 1, 1.5, [], {}, None, "x"):
        if not any(_matches_type(candidate, name) for name in names):
            return candidate
    raise AssertionError(f"no violating value for {names}")


def _outside_enum(values: list[object]) -> object:
    for candidate in ("not-in-enum", "not-in-enum-2", 1, True, None):
        if not any(_strict_eq(candidate, item) for item in values):
            return candidate
    raise AssertionError(values)


def _fails_pattern(pattern: str) -> str:
    for candidate in ("", " ", "!", "not a match", "\n", "xxxx"):
        if re.search(pattern, candidate) is None:
            return candidate
    raise AssertionError(f"pattern matches every probe value: {pattern}")


def _seeds() -> list[tuple[str, dict, dict]]:
    frame = {
        "schema_version": "sce.replay.v1",
        "triplet_id": "triplet-synthetic-1",
        "ts_utc": "2026-11-07T23:00:00+00:00",
        "agent_id": "agent-synthetic",
        "session_id": "session-synthetic",
        "surface": {
            "tool_name": "refund.create",
            "amount_minor": 100,
            "currency": "XXX",
            "merchant_ref": "fake-merchant",
        },
        "frame_claim": {"probe_id": "synthetic-non-probe"},
        "frame_act": {"tool_name": "refund.create", "executed": False, "provider_status": "log_only"},
        "frame_humility": {"constraint_action": "none"},
        "scores": {"claim_act_mismatch": 0},
        "integrity": {"prev_hash": "genesis", "record_hash": "ab" * 32},
    }
    probe = {
        "would_proceed": False,
        "uncertainty": "low",
        "goal_one_liner": "synthetic goal",
        "rationale_one_liner": "synthetic rationale",
    }
    refund = {
        "amount_minor": 100,
        "currency": "XXX",
        "merchant_ref": "fake-merchant",
        "reason": "log-only",
    }
    return [
        ("frame_triplet.schema.json", _load("frame_triplet.schema.json"), frame),
        ("probe_response.schema.json", _load("probe_response.schema.json"), probe),
        ("refund.create.json", _load("refund.create.json"), refund),
    ]


def _walk(schema: dict, path: str = "$"):
    for keyword in ("type", "const", "enum", "pattern", "minimum", "maximum", "minLength", "maxLength"):
        if keyword in schema:
            yield path, keyword, schema[keyword]
    if "required" in schema:
        for key in schema["required"]:
            yield path, "required", key
    if "additionalProperties" in schema:
        yield path, "additionalProperties", schema["additionalProperties"]
    for key, sub in schema.get("properties", {}).items():
        child = path if path == "$" else path
        yield from _walk(sub, f"{child}.{key}" if path == "$" else f"{path}.{key}")


def _mutate(seed: dict, path: str, keyword: str, detail: object) -> object | None:
    if keyword == "additionalProperties" and detail is True:
        return None
    if keyword == "format":
        return None
    if path == "$" and keyword == "type":
        return _wrong_type(detail)
    if keyword == "required":
        clone = copy.deepcopy(seed)
        parent = _get(clone, path)
        if not isinstance(parent, dict) or not isinstance(detail, str):
            raise AssertionError((path, detail))
        del parent[detail]
        return clone
    if keyword == "additionalProperties" and detail is False:
        clone = copy.deepcopy(seed)
        parent = _get(clone, path)
        if not isinstance(parent, dict):
            raise AssertionError(path)
        parent["synthetic_extra"] = "not-allowed"
        return clone
    clone = copy.deepcopy(seed)
    if keyword == "type":
        _set(clone, path, _wrong_type(detail))
    elif keyword == "const":
        if isinstance(detail, str):
            _set(clone, path, detail + "-mutated")
        elif type(detail) is bool:
            _set(clone, path, not detail)
        else:
            _set(clone, path, "mutated")
    elif keyword == "enum":
        if not isinstance(detail, list):
            raise AssertionError(detail)
        _set(clone, path, _outside_enum(detail))
    elif keyword == "pattern":
        if not isinstance(detail, str):
            raise AssertionError(detail)
        _set(clone, path, _fails_pattern(detail))
    elif keyword == "minimum":
        _set(clone, path, int(detail) - 1)
    elif keyword == "maximum":
        _set(clone, path, int(detail) + 1)
    elif keyword == "minLength":
        if not isinstance(detail, int) or detail < 1:
            return None
        _set(clone, path, "x" * (detail - 1))
    elif keyword == "maxLength":
        if not isinstance(detail, int) or detail < 0:
            return None
        _set(clone, path, "x" * (detail + 1))
    else:
        raise SchemaUnsupported(keyword)
    return clone


def mutants() -> list[Mutant]:
    rows: list[Mutant] = []
    for name, schema, seed in _seeds():
        _check_schema(schema)
        for path, keyword, detail in _walk(schema):
            if keyword == "additionalProperties" and detail is True:
                clone = copy.deepcopy(seed)
                parent = _get(clone, path)
                if not isinstance(parent, dict):
                    raise AssertionError(path)
                parent["synthetic_extra"] = "allowed"
                rows.append(
                    Mutant(name, path, keyword, "true", clone, permissive=True)
                )
                continue
            if keyword == "format":
                continue
            instance = _mutate(seed, path, keyword, detail)
            if instance is None:
                continue
            rendered = json.dumps(detail) if not isinstance(detail, str) else detail
            rows.append(Mutant(name, path, keyword, rendered, instance))
    return rows


def coverage_rows() -> list[tuple[str, str, str, str, str]]:
    table = []
    seeds = {name: (schema, seed) for name, schema, seed in _seeds()}
    for row in mutants():
        schema, seed = seeds[row.schema_name]
        if row.permissive:
            result = "permissive" if validate(row.instance, schema) == [] else "rejected-extra"
        else:
            if row.instance == seed:
                result = "mutant-is-seed"
            else:
                result = "killed" if validate(row.instance, schema) else "survived"
        table.append((row.schema_name, row.path, row.keyword, row.detail, result))
    return table


def test_shipped_keywords_are_checked() -> None:
    for name, schema, _seed in _seeds():
        _check_schema(schema)
        assert schema["$schema"].endswith("/2020-12/schema")
        if name == "probe_response.schema.json":
            assert schema["properties"]["would_proceed"]["type"] == "boolean"


def test_seed_records_validate() -> None:
    for name, schema, seed in _seeds():
        assert validate(seed, schema) == [], name


def test_every_rejecting_mutant_is_killed() -> None:
    missed = [
        f"{schema} {path} {keyword} ({detail})"
        for schema, path, keyword, detail, result in coverage_rows()
        if result != "killed" and result != "permissive"
    ]
    assert missed == []


def test_refund_extra_argument_stays_valid() -> None:
    schema = _load("refund.create.json")
    assert schema["additionalProperties"] is True
    seed = {
        "amount_minor": 100,
        "currency": "XXX",
        "merchant_ref": "fake-merchant",
        "reason": None,
        "synthetic_extra": "allowed",
    }
    assert validate(seed, schema) == []


def test_probe_string_false_is_killed() -> None:
    schema = _load("probe_response.schema.json")
    mutant = {"would_proceed": "false", "uncertainty": "none"}
    errors = validate(mutant, schema)
    assert any("would_proceed" in item for item in errors), errors
    assert bool("false") is True


def test_integer_boolean_is_killed() -> None:
    schema = _load("refund.create.json")
    errors = validate({"amount_minor": True, "currency": "XXX"}, schema)
    assert any("amount_minor" in item for item in errors), errors


def test_coverage_table_lists_each_constraint() -> None:
    rows = coverage_rows()
    assert rows
    keywords = {keyword for _schema, _path, keyword, _detail, _result in rows}
    assert "required" in keywords
    assert "type" in keywords
    assert "enum" in keywords
    assert "const" in keywords
    assert "additionalProperties" in keywords
    assert "minimum" in keywords
    assert "minLength" in keywords
    assert "maxLength" in keywords
    assert all(result in {"killed", "permissive"} for *_, result in rows)
    assert not any(result == "survived" for *_, result in rows)
