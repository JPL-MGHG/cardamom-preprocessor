#!/usr/bin/env python3
"""
Simple test runner for CARDAMOM preprocessing tests.

This script runs the basic test suite and provides a summary of results.
It can be run with the project's Python environment.
"""

import sys
import subprocess
from pathlib import Path


def run_tests():
    """Run all tests and return results"""
    test_dir = Path(__file__).parent / "tests"

    if not test_dir.exists():
        print("Error: tests directory not found")
        return False

    # Find all test files
    test_files = list(test_dir.glob("test_*.py"))

    if not test_files:
        print("Error: no test files found")
        return False

    print("CARDAMOM Preprocessing Test Suite")
    print("=" * 40)
    print(f"Found {len(test_files)} test files")
    print()

    # Try to run pytest if available
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            str(test_dir),
            "-v",
            "--tb=short"
        ], capture_output=True, text=True)

        print("Test Results:")
        print(result.stdout)

        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)

        return result.returncode == 0

    except FileNotFoundError:
        print("pytest not available, running tests individually...")

        # Fallback: run tests individually
        all_passed = True
        for test_file in test_files:
            print(f"Running {test_file.name}...")
            try:
                result = subprocess.run([
                    sys.executable, str(test_file)
                ], capture_output=True, text=True)

                if result.returncode == 0:
                    print(f"  ✓ {test_file.name} PASSED")
                else:
                    print(f"  ✗ {test_file.name} FAILED")
                    print(f"    Error: {result.stderr}")
                    all_passed = False

            except Exception as e:
                print(f"  ✗ {test_file.name} ERROR: {e}")
                all_passed = False

        return all_passed


def main():
    """Main function"""
    print("Starting CARDAMOM preprocessing tests...")
    print(f"Python executable: {sys.executable}")
    print()

    success = run_tests()

    print()
    if success:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()