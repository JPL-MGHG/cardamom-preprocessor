#!/usr/bin/env python3
"""
Test script for ecmwf-datastores-client batch download functionality
Tests with a minimal dataset to verify the implementation
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ecmwf_downloader import ECMWFDownloader


def test_small_hourly_batch():
    """Test hourly batch download with minimal dataset"""
    print("\n" + "="*70)
    print("Testing Hourly Batch Download (Small Dataset)")
    print("="*70)

    # Create downloader with small test area (Northeast US)
    test_area = [45, -75, 40, -70]  # Small region
    downloader = ECMWFDownloader(
        area=test_area,
        output_dir="./test_output_hourly"
    )

    # Test with just 1 variable, 1 month, 1 year
    test_variables = ["2m_temperature"]
    test_years = [2024]
    test_months = [1]

    print(f"\nTest parameters:")
    print(f"  Variables: {test_variables}")
    print(f"  Year: {test_years}")
    print(f"  Month: {test_months}")
    print(f"  Area: {test_area}")

    try:
        downloader.download_hourly_data(
            variables=test_variables,
            years=test_years,
            months=test_months,
            file_prefix="TEST_HOURLY"
        )
        print("\n✓ Hourly batch download test completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Hourly batch download test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_small_monthly_batch():
    """Test monthly batch download with minimal dataset"""
    print("\n" + "="*70)
    print("Testing Monthly Batch Download (Small Dataset)")
    print("="*70)

    # Create downloader with small test area
    test_area = [45, -75, 40, -70]  # Small region
    downloader = ECMWFDownloader(
        area=test_area,
        output_dir="./test_output_monthly"
    )

    # Test with just 2 variables, 1 month, 1 year
    test_variables = ["2m_temperature", "total_precipitation"]
    test_years = [2024]
    test_months = [1]

    print(f"\nTest parameters:")
    print(f"  Variables: {test_variables}")
    print(f"  Year: {test_years}")
    print(f"  Month: {test_months}")
    print(f"  Area: {test_area}")

    try:
        downloader.download_monthly_data(
            variables=test_variables,
            years=test_years,
            months=test_months,
            product_type="monthly_averaged_reanalysis",
            file_prefix="TEST_MONTHLY"
        )
        print("\n✓ Monthly batch download test completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Monthly batch download test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("ECMWF Batch Download Tests")
    print("Testing ecmwf-datastores-client implementation")
    print("="*70)

    results = []

    # Test 1: Hourly batch download
    results.append(("Hourly Batch", test_small_hourly_batch()))

    # Test 2: Monthly batch download
    results.append(("Monthly Batch", test_small_monthly_batch()))

    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False

    print("="*70)

    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed. Check output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
