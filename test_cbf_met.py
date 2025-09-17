#!/usr/bin/env python3
"""
Test script for CBF Meteorological Processing

Tests the separated download and processing workflow for generating
CBF meteorological driver files from ERA5 data.
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ecmwf_downloader import ECMWFDownloader
from src.cbf_met_processor import CBFMetProcessor
import logging


def test_cbf_variables_listing():
    """Test listing of supported CBF variables."""
    print("=" * 60)
    print("TEST: CBF Variables and Requirements")
    print("=" * 60)

    processor = CBFMetProcessor()

    print("\nCBF Meteorological Variables (Target Output):")
    print("-" * 45)
    cbf_vars = processor.get_supported_variables()
    for i, var in enumerate(cbf_vars, 1):
        print(f"{i:2d}. {var}")

    print(f"\nTotal CBF variables: {len(cbf_vars)}")

    print("\nRequired ERA5 Input Variables:")
    print("-" * 30)
    era5_vars = processor.get_era5_requirements()
    for i, var in enumerate(era5_vars, 1):
        print(f"{i:2d}. {var}")

    print(f"Total ERA5 variables needed: {len(era5_vars)}")

    print("\nVariable Coverage Analysis:")
    print("-" * 25)
    mapping = processor.era5_to_cbf_mapping
    era5_covered = 0
    for era5_var, cbf_var in mapping.items():
        if isinstance(cbf_var, list):
            cbf_str = ", ".join(cbf_var)
            era5_covered += len(cbf_var)
        else:
            cbf_str = cbf_var
            era5_covered += 1
        print(f"  {era5_var:<35} -> {cbf_str}")

    print(f"\nERA5 can provide: {era5_covered}/10 CBF variables ({era5_covered/10*100:.0f}%)")
    print("Missing from ERA5: CO2 (external), BURNED_AREA (external)")


def test_separate_workflow_simulation():
    """Test the separated workflow with simulation (no actual downloads)."""
    print("\n" + "=" * 60)
    print("TEST: Separated Download/Processing Workflow")
    print("=" * 60)

    logger = logging.getLogger(__name__)

    # Test ERA5 variables that would be needed for CBF processing
    test_variables = [
        '2m_temperature',           # -> T2M_MIN, T2M_MAX
        '2m_dewpoint_temperature',  # -> VPD (with temperature)
        'total_precipitation',      # -> TOTAL_PREC
        'surface_solar_radiation_downwards',   # -> SSRD
        'surface_thermal_radiation_downwards', # -> STRD
        'snowfall',                # -> SNOWFALL
        'skin_temperature'         # -> SKT
    ]

    print(f"\nTesting with ERA5 variables: {len(test_variables)}")
    for var in test_variables:
        print(f"  - {var}")

    # Test 1: Download specification
    print("\n1. Download Phase Specification:")
    print("-" * 35)

    try:
        downloader = ECMWFDownloader(output_dir="./test_downloads")

        # Validate variables before "download"
        validation_results = downloader.validate_variables(test_variables)
        valid_vars = [v for v, valid in validation_results.items() if valid]
        invalid_vars = [v for v, valid in validation_results.items() if not valid]

        print(f"   Valid variables: {len(valid_vars)}/{len(test_variables)}")
        if invalid_vars:
            print(f"   Invalid variables: {invalid_vars}")

        # Get variable metadata
        print("\n   Variable Metadata Check:")
        for var in valid_vars[:3]:  # Check first 3 variables
            metadata = downloader.get_variable_metadata(var)
            if metadata:
                cbf_names = metadata.get('cbf_names', [var])
                print(f"     {var} -> {cbf_names}")

        print("   ✓ Download phase validation completed")

    except Exception as e:
        logger.error(f"Download validation failed: {e}")
        return False

    # Test 2: Processing specification
    print("\n2. Processing Phase Specification:")
    print("-" * 35)

    try:
        processor = CBFMetProcessor(output_dir="./test_output")

        # Test variable mapping
        era5_to_cbf = processor.era5_to_cbf_mapping
        print(f"   ERA5->CBF mappings defined: {len(era5_to_cbf)}")

        # Test unit conversions
        conversions = processor.unit_conversions
        print(f"   Unit conversions defined: {len(conversions)}")
        for var, conversion in list(conversions.items())[:2]:
            print(f"     {var}: {conversion['from']} -> {conversion['to']}")

        print("   ✓ Processing phase specification completed")

    except Exception as e:
        logger.error(f"Processing specification failed: {e}")
        return False

    print("\n✓ Separated workflow validation PASSED")
    return True


def test_cbf_file_structure():
    """Test CBF file structure requirements."""
    print("\n" + "=" * 60)
    print("TEST: CBF File Structure Requirements")
    print("=" * 60)

    processor = CBFMetProcessor()

    # Required variables from erens_cbf_code.py
    required_cbf_vars = processor.cbf_met_variables

    print("Required CBF MET variables:")
    print("-" * 30)
    for i, var in enumerate(required_cbf_vars, 1):
        print(f"{i:2d}. {var}")

    # Check which can be derived from ERA5
    era5_mapping = processor.era5_to_cbf_mapping
    era5_provided = []
    external_needed = []

    for cbf_var in required_cbf_vars:
        found = False
        for era5_var, cbf_targets in era5_mapping.items():
            if isinstance(cbf_targets, list):
                if cbf_var in cbf_targets:
                    era5_provided.append(cbf_var)
                    found = True
                    break
            else:
                if cbf_var == cbf_targets:
                    era5_provided.append(cbf_var)
                    found = True
                    break

        if not found:
            if cbf_var == 'VPD':
                era5_provided.append(cbf_var)  # Derived from temperature + dewpoint
            else:
                external_needed.append(cbf_var)

    print(f"\nCoverage Analysis:")
    print("-" * 20)
    print(f"ERA5 can provide: {len(era5_provided)}/10 variables")
    print(f"  Variables: {era5_provided}")
    print(f"External sources needed: {len(external_needed)}/10 variables")
    print(f"  Variables: {external_needed}")

    coverage_percent = len(era5_provided) / len(required_cbf_vars) * 100
    print(f"\nCoverage: {coverage_percent:.0f}%")

    if coverage_percent >= 80:
        print("✓ Sufficient coverage for CBF processing")
        return True
    else:
        print("✗ Insufficient coverage for CBF processing")
        return False


def main():
    """Run all CBF MET processing tests."""
    print("CBF Meteorological Processing Test Suite")
    print("========================================")

    # Run tests
    test_results = []

    try:
        # Test 1: Variable listing and mapping
        test_cbf_variables_listing()
        test_results.append(("Variable Listing", True))

        # Test 2: Separated workflow
        result2 = test_separate_workflow_simulation()
        test_results.append(("Separated Workflow", result2))

        # Test 3: CBF file structure
        result3 = test_cbf_file_structure()
        test_results.append(("CBF Structure", result3))

    except Exception as e:
        print(f"\nTest suite failed with error: {e}")
        return 1

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1

    print(f"\nTests passed: {passed}/{len(test_results)}")

    if passed == len(test_results):
        print("✓ All tests PASSED - CBF MET processing ready")
        return 0
    else:
        print("✗ Some tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())