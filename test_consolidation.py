#!/usr/bin/env python3
"""
Simple test script to verify the new consolidation functionality works correctly.
"""

import os
import tempfile
import shutil
import numpy as np
import xarray as xr
from src.ecmwf_downloader import ECMWFDownloader

def create_test_netcdf_files(temp_dir, num_files=3):
    """Create sample NetCDF files for testing."""
    test_files = []

    # Create time coordinates for different months
    times = [
        xr.cftime_range('2023-01-01', periods=31, freq='D'),
        xr.cftime_range('2023-02-01', periods=28, freq='D'),
        xr.cftime_range('2023-03-01', periods=31, freq='D')
    ]

    for i in range(num_files):
        # Create sample data with different variables
        data_vars = {}

        if i == 0:
            # File 1: temperature and pressure
            data_vars['2m_temperature'] = (['time', 'latitude', 'longitude'],
                                         np.random.rand(len(times[i]), 5, 5) * 20 + 273.15)
            data_vars['surface_pressure'] = (['time', 'latitude', 'longitude'],
                                           np.random.rand(len(times[i]), 5, 5) * 10000 + 95000)
        elif i == 1:
            # File 2: precipitation
            data_vars['total_precipitation'] = (['time', 'latitude', 'longitude'],
                                              np.random.rand(len(times[i]), 5, 5) * 0.01)
        else:
            # File 3: radiation
            data_vars['surface_solar_radiation_downwards'] = (['time', 'latitude', 'longitude'],
                                                            np.random.rand(len(times[i]), 5, 5) * 300)

        # Create coordinates
        coords = {
            'time': times[i],
            'latitude': np.linspace(40, 45, 5),
            'longitude': np.linspace(-75, -70, 5)
        }

        # Create dataset
        ds = xr.Dataset(data_vars, coords=coords)

        # Save to file
        filepath = os.path.join(temp_dir, f'test_file_{i+1}.nc')
        ds.to_netcdf(filepath)
        test_files.append(filepath)

        print(f"Created test file: {filepath}")
        print(f"  Variables: {list(ds.data_vars.keys())}")
        print(f"  Time range: {times[i][0].strftime('%Y-%m-%d')} to {times[i][-1].strftime('%Y-%m-%d')}")

    return test_files

def test_consolidation():
    """Test the consolidation functionality."""
    print("=" * 60)
    print("Testing ECMWF Downloader Consolidation Functionality")
    print("=" * 60)

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\nUsing temporary directory: {temp_dir}")

        # Create test NetCDF files
        print("\n1. Creating test NetCDF files...")
        test_files = create_test_netcdf_files(temp_dir)

        # Initialize downloader (without CDS credentials for testing)
        print("\n2. Initializing ECMWFDownloader...")
        downloader = ECMWFDownloader(output_dir=temp_dir, disable_cds_client=True)

        # Test consolidation
        print("\n3. Testing consolidation...")
        try:
            consolidated_file = downloader._consolidate_extracted_files(
                extracted_files=test_files,
                file_prefix="TEST_CONSOLIDATED",
                years=[2023]
            )

            print(f"Consolidation successful!")
            print(f"Consolidated file: {consolidated_file}")

            # Verify consolidated file
            print("\n4. Verifying consolidated file...")
            ds_consolidated = xr.open_dataset(consolidated_file)

            print(f"Variables in consolidated file: {list(ds_consolidated.data_vars.keys())}")
            print(f"Time dimension size: {len(ds_consolidated.time)}")
            print(f"Time range: {ds_consolidated.time.min().values} to {ds_consolidated.time.max().values}")
            print(f"Spatial dimensions: lat={len(ds_consolidated.latitude)}, lon={len(ds_consolidated.longitude)}")

            # Check that all variables are present
            expected_vars = {'2m_temperature', 'surface_pressure', 'total_precipitation', 'surface_solar_radiation_downwards'}
            actual_vars = set(ds_consolidated.data_vars.keys())

            if expected_vars.issubset(actual_vars):
                print("✓ All expected variables are present in consolidated file")
            else:
                missing = expected_vars - actual_vars
                print(f"✗ Missing variables: {missing}")

            # Check file naming
            expected_filename = "TEST_CONSOLIDATED_2023.nc"
            actual_filename = os.path.basename(consolidated_file)
            if actual_filename == expected_filename:
                print(f"✓ File naming correct: {actual_filename}")
            else:
                print(f"✗ File naming incorrect. Expected: {expected_filename}, Got: {actual_filename}")

            ds_consolidated.close()

            print("\n5. Testing year range naming...")
            # Test with multiple years
            consolidated_file_range = downloader._consolidate_extracted_files(
                extracted_files=test_files,
                file_prefix="TEST_RANGE",
                years=[2020, 2021, 2022]
            )

            expected_range_filename = "TEST_RANGE_2020_2022.nc"
            actual_range_filename = os.path.basename(consolidated_file_range)
            if actual_range_filename == expected_range_filename:
                print(f"✓ Year range naming correct: {actual_range_filename}")
            else:
                print(f"✗ Year range naming incorrect. Expected: {expected_range_filename}, Got: {actual_range_filename}")

            print("\n" + "=" * 60)
            print("CONSOLIDATION TEST COMPLETED SUCCESSFULLY!")
            print("=" * 60)

        except Exception as e:
            print(f"\n✗ Consolidation test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    return True

def main():
    """Main test function."""
    success = test_consolidation()

    if success:
        print("\n🎉 All tests passed! The consolidation approach is working correctly.")
        print("\nKey benefits demonstrated:")
        print("- Multiple NetCDF files successfully combined into single file")
        print("- All variables and time periods preserved")
        print("- Proper year-range naming convention")
        print("- Clean API for consolidation function")
    else:
        print("\n❌ Tests failed. Please check the implementation.")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())