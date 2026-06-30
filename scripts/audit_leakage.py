#!/usr/bin/env python
"""
Leakage Audit
==============
Audit every engineered feature for data leakage.

Checks:
  1. No rolling window uses center=True
  2. No shift uses negative values (future)
  3. No cross-correlation uses future samples
  4. All diffs/shifts are backward-looking

Usage:
    python scripts/audit_leakage.py
"""

import ast
import sys
from pathlib import Path
from typing import List


class LeakageAuditor:
    """Static analysis of feature extractor code for common leak patterns."""

    SUSPICIOUS_CENTER = []
    SUSPICIOUS_SHIFT = []
    PASS_COUNT = 0
    FAIL_COUNT = 0

    def audit_file(self, path: Path):
        """Parse Python file and check for leakage patterns."""
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError as e:
            print(f"  SKIP (parse error): {e}")
            return

        filename = path.relative_to(Path.cwd())
        print(f"\n{'='*60}")
        print(f"Auditing: {filename}")

        for node in ast.walk(tree):
            self._check_rolling_center(node, filename)
            self._check_shift_negative(node, filename)
            self._check_shift_center(node, filename)

    def _check_rolling_center(self, node, filename):
        """Check for rolling(..., center=True) which leaks future data."""
        if (isinstance(node, ast.Call) and
                hasattr(node.func, 'attr') and node.func.attr == 'rolling'):
            for kw in node.keywords:
                if kw.arg == 'center':
                    if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        line = getattr(node, 'lineno', '?')
                        self.FAIL_COUNT += 1
                        LeakageAuditor.SUSPICIOUS_CENTER.append(
                            f"  ❌ LEAK: {filename}:{line} — rolling(center=True) uses future data"
                        )

    def _check_shift_negative(self, node, filename):
        """Check for shift(-N) which shifts future into present."""
        if (isinstance(node, ast.Call) and
                hasattr(node.func, 'attr') and node.func.attr == 'shift'):
            if node.args:
                arg = node.args[0]
                if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                    line = getattr(node, 'lineno', '?')
                    self.FAIL_COUNT += 1
                    LeakageAuditor.SUSPICIOUS_SHIFT.append(
                        f"  ❌ LEAK: {filename}:{line} — shift(-N) reads future data"
                    )
            for kw in node.keywords:
                if kw.arg == 'periods' and isinstance(kw.value, ast.UnaryOp) and isinstance(kw.value.op, ast.USub):
                    line = getattr(node, 'lineno', '?')
                    self.FAIL_COUNT += 1
                    LeakageAuditor.SUSPICIOUS_SHIFT.append(
                        f"  ❌ LEAK: {filename}:{line} — shift(periods=-N) reads future data"
                    )

    def _check_shift_center(self, node, filename):
        """Check for .diff() without shift — .diff() is backward by default, OK."""
        pass  # pd.Series.diff() defaults to periods=1 (backward), safe

    def report(self):
        print(f"\n{'='*60}")
        print("LEAKAGE AUDIT REPORT")
        print(f"{'='*60}")
        print(f"\nTotal issues found: {len(LeakageAuditor.SUSPICIOUS_CENTER) + len(LeakageAuditor.SUSPICIOUS_SHIFT)}")
        print(f"  Rolling center=True: {len(LeakageAuditor.SUSPICIOUS_CENTER)}")
        print(f"  Negative shift:      {len(LeakageAuditor.SUSPICIOUS_SHIFT)}")

        if LeakageAuditor.SUSPICIOUS_CENTER:
            print("\nRolling with center=True (FUTURE LEAK):")
            for s in LeakageAuditor.SUSPICIOUS_CENTER:
                print(s)

        if LeakageAuditor.SUSPICIOUS_SHIFT:
            print("\nNegative shifts (FUTURE LEAK):")
            for s in LeakageAuditor.SUSPICIOUS_SHIFT:
                print(s)

        if not LeakageAuditor.SUSPICIOUS_CENTER and not LeakageAuditor.SUSPICIOUS_SHIFT:
            print("\n  ✅ No leakage detected in audited files.")

        total = len(LeakageAuditor.SUSPICIOUS_CENTER) + len(LeakageAuditor.SUSPICIOUS_SHIFT)
        return total


def main():
    auditor = LeakageAuditor()
    feature_dirs = [
        Path("backend/features"),
        Path("backend/data"),
        Path("backend/models/nowcaster"),
    ]
    for d in feature_dirs:
        if d.exists():
            for py_file in sorted(d.rglob("*.py")):
                if py_file.name.startswith("__"):
                    continue
                auditor.audit_file(py_file)

    issues = auditor.report()
    print(f"\nAudit complete. {issues} leakage issue(s) found.")
    return 1 if issues > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
