#!/usr/bin/env python3
"""
Test Runner - Εκτέλεση όλων των unit tests
Test Runner - Execute all unit tests
"""

import os
import sys
import unittest

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test modules
from test_backend import *
from test_components import *


def run_all_tests():
    """Εκτέλεση όλων των tests"""

    print("=" * 80)
    print("EEG ARTIFACT CLEANER - UNIT TESTS")
    print("=" * 80)

    # Create test suite
    suite = unittest.TestSuite()

    # Backend tests
    print("\n📦 Backend Tests:")
    print("-" * 50)

    backend_tests = [
        TestEEGDataManager,
        TestEEGPreprocessor,
        TestICAProcessor,
        TestArtifactDetector,
        TestEEGArtifactCleaningService,
    ]

    for test_class in backend_tests:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
        print(f"✓ {test_class.__name__}")

    # GUI Component tests
    print("\n🖥️  GUI Component Tests:")
    print("-" * 50)

    gui_tests = [
        TestICAComponentSelector,
        TestResultsDisplayWidget,
        TestStatisticsTableWidget,
        TestComponentIntegration,
    ]

    for test_class in gui_tests:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
        print(f"✓ {test_class.__name__}")

    print("\n" + "=" * 80)
    print("ΕΚΤΕΛΕΣΗ TESTS...")
    print("=" * 80)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=True)

    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 80)
    print("ΠΕΡΙΛΗΨΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ")
    print("=" * 80)

    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, "skipped") else 0
    successful = total_tests - failures - errors - skipped

    print(f"📊 Total tests: {total_tests}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failures: {failures}")
    print(f"🚫 Errors: {errors}")
    print(f"⏭️  Skipped: {skipped}")

    success_rate = (successful / total_tests * 100) if total_tests > 0 else 0
    print(f"📈 Success rate: {success_rate:.1f}%")

    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")

    if result.errors:
        print("\n🚫 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")

    print("=" * 80)

    # Return success status
    return failures == 0 and errors == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
