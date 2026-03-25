"""Shared test helpers for the custom test harness.

Usage in test files:
    from tests.test_utils import TestCounter
    tc = TestCounter()
    tc.check("my test", some_condition, "optional detail on failure")
    ...
    tc.summary("Connect4 game logic")
"""


class TestCounter:
    """Tracks pass/fail counts and provides check() and summary() helpers."""

    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(self, name, condition, detail=""):
        """Record a single test result."""
        if condition:
            self.passed += 1
            print(f"  PASS: {name}")
        else:
            self.failed += 1
            print(f"  FAIL: {name}  {detail}")

    def summary(self, label=""):
        """Print final pass/fail summary and return exit-code-friendly bool."""
        print(f"\n{'='*50}")
        print(f"RESULTS: {self.passed} passed, {self.failed} failed")
        if self.failed == 0:
            msg = f"All {label} tests PASSED!" if label else "All tests passed!"
            print(msg)
        else:
            print(f"WARNING: {self.failed} test(s) FAILED!")
        return self.failed == 0
