#!/usr/bin/env python3
"""
Simple smoke test runner that does not require pytest.
It imports the tests in tests_smoke/ and calls their test functions directly.
"""
import sys


def main() -> int:
    failed = 0
    passed = 0

    try:
        from tests_smoke.test_dfc_speed_smoke import test_dfc_speed_synthetic_small

        test_dfc_speed_synthetic_small()
        print("[OK] test_dfc_speed_synthetic_small")
        passed += 1
    except Exception as e:
        print(f"[FAIL] test_dfc_speed_synthetic_small: {e}")
        failed += 1

    try:
        from tests_smoke.test_meta_allegiance_smoke import (
            test_meta_connectivity_and_allegiance_tiny,
        )

        test_meta_connectivity_and_allegiance_tiny()
        print("[OK] test_meta_connectivity_and_allegiance_tiny")
        passed += 1
    except Exception as e:
        print(f"[FAIL] test_meta_connectivity_and_allegiance_tiny: {e}")
        failed += 1

    try:
        from tests_smoke.test_matplotlib_smoke import test_minimal_plot_backend_agg

        test_minimal_plot_backend_agg()
        print("[OK] test_minimal_plot_backend_agg")
        passed += 1
    except Exception as e:
        print(f"[FAIL] test_minimal_plot_backend_agg: {e}")
        failed += 1

    print(f"[RESULT] passed={passed}, failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

