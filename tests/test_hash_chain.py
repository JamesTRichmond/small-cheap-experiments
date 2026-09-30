from probe_runtime.hash_chain import attach_integrity, verify_chain


def test_chain_roundtrip():
    a = attach_integrity({"triplet_id": "1", "n": 1})
    b = attach_integrity({"triplet_id": "2", "n": 2}, a["integrity"]["record_hash"])
    assert verify_chain([a, b])
