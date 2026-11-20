"""Call the head-to-head comparison once per case.

``compare_batch`` owns STOP, INCONCLUSIVE, and EXPERIMENTAL_INCREMENTAL_SIGNAL.
This module does not restate that table. A one-case batch is whatever
``compare_batch`` returns for that one case, including INCONCLUSIVE when the
case cannot show both a bad refund and a good control.

Nothing here blocks a refund. Synthetic cases are not market evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from probe_runtime.head_to_head import Comparison, compare_batch


def per_case_comparisons(cases: Sequence[Mapping]) -> tuple[Comparison, ...]:
    """One ``compare_batch`` call per case. The batch is not scored as a whole."""
    return tuple(compare_batch((case,)) for case in cases)
