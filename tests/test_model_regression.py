"""scripts/check_regression.py::check_regression -- la lógica pura de comparación
entre las últimas dos versiones de un registry.json, sin necesidad de un registry
real (que solo existe tras entrenar sobre datos de Kaggle, ver README)."""
from scripts.check_regression import check_regression


def _entry(*wrmsse_values: float) -> dict:
    """Construye un entry de registry con una versión por valor de wrmsse_test,
    timestamps crecientes para que el orden de "últimas dos" sea determinista."""
    return {
        "versions": {
            f"2026010{i + 1}T000000Z": {"wrmsse_test": v, "wape_test": v}
            for i, v in enumerate(wrmsse_values)
        },
    }


def test_no_violations_when_metric_improves():
    assert check_regression(_entry(0.50, 0.45), tolerance=0.05) == []


def test_flags_regression_beyond_tolerance():
    violations = check_regression(_entry(0.50, 0.60), tolerance=0.05)  # +20%
    assert len(violations) == 2  # wrmsse_test y wape_test comparten el mismo valor acá
    assert "wrmsse_test" in violations[0]


def test_small_regression_within_tolerance_is_not_flagged():
    assert check_regression(_entry(0.50, 0.52), tolerance=0.10) == []  # +4% < 10%


def test_single_version_has_nothing_to_compare():
    assert check_regression(_entry(0.50), tolerance=0.05) == []


def test_missing_metric_in_either_version_is_skipped_not_crashed():
    entry = {
        "versions": {
            "20260101T000000Z": {"wrmsse_test": 0.5},  # sin wape_test
            "20260102T000000Z": {"wrmsse_test": 0.9, "wape_test": 0.9},
        },
    }
    violations = check_regression(entry, tolerance=0.05)
    assert len(violations) == 1
    assert "wrmsse_test" in violations[0]
