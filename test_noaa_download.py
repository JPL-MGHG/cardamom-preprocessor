#!/usr/bin/env python
"""
Test script for NOAA CO2 downloader.

This script tests the NOAADownloader functionality by downloading CO2 data
in two modes:
1. Full dataset download (default, recommended)
2. Single month download (backwards compatibility)

Usage:
    .venv/bin/python test_noaa_download.py

Scientific Context:
NOAA Global Monitoring Laboratory provides monthly atmospheric CO2 concentrations
measured at multiple observation sites globally. This data is essential for
CARDAMOM carbon cycle modeling, providing the atmospheric CO2 context for
photosynthesis calculations.

Since the source data is a small CSV file (~100KB) containing the entire
historical record, it's more efficient to download all data at once rather
than requesting individual months.

Expected output:
    ✓ NOAA downloader initialized successfully
    ✓ Downloaded complete CO2 dataset (1958-present)
    ✓ Generated NetCDF file with spatially-replicated CO2 grid
    ✓ Created STAC metadata for data discovery
    ✓ Single-month download mode also works (backwards compatibility)
"""

import sys
from pathlib import Path
import tempfile
import logging
import xarray as xr

# Configure logging for test output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the NOAA downloader from the installed package
from downloaders.noaa_downloader import NOAADownloader


def validate_netcdf_structure(netcdf_file_path):
    """
    Validate NetCDF file structure and contents.

    Args:
        netcdf_file_path (Path): Path to NetCDF file

    Raises:
        AssertionError: If validation fails
    """

    # Load dataset
    ds = xr.open_dataset(netcdf_file_path)

    # Check required dimensions
    required_dims = ['time', 'latitude', 'longitude']
    for dim in required_dims:
        assert dim in ds.dims, f"Missing dimension: {dim}"
    print(f"  ✓ Dimensions: {list(ds.dims.keys())}")

    # Check required variables
    assert 'CO2' in ds.data_vars, "Missing CO2 variable"
    print(f"  ✓ Variables: {list(ds.data_vars.keys())}")

    # Check spatial extent (0.5° CARDAMOM grid)
    lat_size = len(ds.latitude)
    lon_size = len(ds.longitude)
    print(f"  ✓ Spatial grid: {lat_size} × {lon_size} (0.5° resolution)")

    # Check temporal dimension
    time_size = len(ds.time)
    print(f"  ✓ Temporal steps: {time_size}")

    # Check CO2 values are physically reasonable
    co2_values = ds['CO2'].values
    co2_min = float(co2_values.min())
    co2_max = float(co2_values.max())
    co2_mean = float(co2_values.mean())

    # CO2 typical range: 280-430 ppm
    assert 250 < co2_min < 500, f"CO2 min value out of range: {co2_min}"
    assert 250 < co2_max < 500, f"CO2 max value out of range: {co2_max}"
    print(f"  ✓ CO2 values: min={co2_min:.2f}, mean={co2_mean:.2f}, max={co2_max:.2f} ppm")

    # Verify spatially replicated (all values should be identical for CO2)
    # Calculate standard deviation across space
    co2_spatial_std = float(co2_values.std())
    assert co2_spatial_std < 0.01, f"CO2 not spatially uniform: std={co2_spatial_std}"
    print(f"  ✓ Verified spatial replication (std={co2_spatial_std:.4f})")

    ds.close()


def test_full_dataset_download():
    """
    Test NOAA downloader with full dataset download (default mode).

    This test:
    1. Creates a temporary output directory
    2. Initializes NOAADownloader
    3. Downloads all available CO2 data
    4. Validates NetCDF structure and contents
    5. Checks STAC metadata generation
    """

    print("\n" + "="*70)
    print("NOAA CO2 Downloader Test - Full Dataset Mode")
    print("="*70)

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        print(f"\nTest output directory: {output_dir}")

        try:
            # Initialize downloader
            print("\n[1] Initializing NOAADownloader...")
            downloader = NOAADownloader(
                output_directory=str(output_dir),
                keep_raw_files=False,
                verbose=True
            )
            print("    ✓ NOAADownloader initialized successfully")

            # Download all available CO2 data
            print("\n[2] Downloading all available NOAA CO2 data...")
            results = downloader.download_and_process()

            # Validate results dictionary structure
            print(f"\n[3] Validating results...")
            assert results['success'], "Download marked as failed"
            print(f"    ✓ Download successful")

            # Check time range information
            assert 'time_range' in results, "No time_range in results"
            start_year, end_year = results['time_range']
            num_steps = results['num_time_steps']
            print(f"    ✓ Time range: {start_year}-{end_year} ({num_steps} months)")

            # Check output files
            output_files = results['output_files']
            assert len(output_files) > 0, "No output files generated"
            print(f"    ✓ Generated {len(output_files)} NetCDF file(s)")

            # Validate each NetCDF file
            for output_file in output_files:
                output_path = Path(output_file)
                assert output_path.exists(), f"Output file not found: {output_file}"
                file_size = output_path.stat().st_size
                print(f"\n    Validating NetCDF: {output_path.name} ({file_size:,} bytes)")

                # Load dataset
                ds = xr.open_dataset(output_path)

                # Check required dimensions
                required_dims = ['time', 'latitude', 'longitude']
                for dim in required_dims:
                    assert dim in ds.dims, f"Missing dimension: {dim}"
                print(f"      ✓ Dimensions: {list(ds.dims.keys())}")

                # Check required variables
                assert 'CO2' in ds.data_vars, "Missing CO2 variable"
                print(f"      ✓ Variables: {list(ds.data_vars.keys())}")

                # Check time dimension matches expected
                time_size = len(ds.time)
                assert time_size == num_steps, f"Time dimension mismatch: {time_size} != {num_steps}"
                print(f"      ✓ Time steps: {time_size}")

                # Check CO2 values are physically reasonable
                co2_values = ds['CO2'].values
                co2_min = float(co2_values.min())
                co2_max = float(co2_values.max())
                co2_mean = float(co2_values.mean())

                # CO2 typical range: 280-500 ppm
                assert 250 < co2_min < 500, f"CO2 min value out of range: {co2_min}"
                assert 250 < co2_max < 500, f"CO2 max value out of range: {co2_max}"
                print(f"      ✓ CO2 values: min={co2_min:.2f}, mean={co2_mean:.2f}, max={co2_max:.2f} ppm")

                ds.close()

            # Check STAC items
            stac_items = results.get('stac_items', [])
            assert len(stac_items) > 0, "No STAC items generated"
            print(f"\n    ✓ Generated {len(stac_items)} STAC item(s)")

            # Check collection ID
            collection_id = results.get('collection_id')
            assert collection_id, "No collection ID returned"
            print(f"    ✓ Collection: {collection_id}")

            print("\n" + "="*70)
            print("✓ Full dataset download test passed!")
            print("="*70 + "\n")

            return True

        except Exception as e:
            print(f"\n✗ Test failed with error:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_single_month_download():
    """
    Test NOAA downloader with single month download (backwards compatibility).

    This test:
    1. Creates a temporary output directory
    2. Initializes NOAADownloader
    3. Downloads CO2 data for January 2020
    4. Validates NetCDF structure and contents
    5. Checks STAC metadata generation
    """

    print("\n" + "="*70)
    print("NOAA CO2 Downloader Test - Single Month Mode (Backwards Compatibility)")
    print("="*70)

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        print(f"\nTest output directory: {output_dir}")

        try:
            # Initialize downloader
            print("\n[1] Initializing NOAADownloader...")
            downloader = NOAADownloader(
                output_directory=str(output_dir),
                keep_raw_files=False,
                verbose=True
            )
            print("    ✓ NOAADownloader initialized successfully")

            # Download CO2 data for January 2020
            print("\n[2] Downloading NOAA CO2 data for January 2020...")
            results = downloader.download_and_process(
                year=2020,
                month=1
            )

            # Validate results dictionary structure
            print(f"\n[3] Validating results...")
            assert results['success'], "Download marked as failed"
            print(f"    ✓ Download successful")

            # Check output files
            output_files = results['output_files']
            assert len(output_files) > 0, "No output files generated"
            print(f"    ✓ Generated {len(output_files)} NetCDF file(s)")

            # Validate each NetCDF file
            for output_file in output_files:
                output_path = Path(output_file)
                assert output_path.exists(), f"Output file not found: {output_file}"
                file_size = output_path.stat().st_size
                print(f"\n    Validating NetCDF: {output_path.name} ({file_size:,} bytes)")

                # Validate structure and contents
                validate_netcdf_structure(output_path)

            # Check STAC items
            stac_items = results.get('stac_items', [])
            assert len(stac_items) > 0, "No STAC items generated"
            print(f"\n    ✓ Generated {len(stac_items)} STAC item(s)")

            # Check collection ID
            collection_id = results.get('collection_id')
            assert collection_id, "No collection ID returned"
            print(f"    ✓ Collection: {collection_id}")

            print("\n" + "="*70)
            print("✓ Single month download test passed!")
            print("="*70 + "\n")

            return True

        except Exception as e:
            print(f"\n✗ Test failed with error:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == '__main__':
    # Run both tests
    print("="*70)
    print("NOAA CO2 Downloader Test Suite")
    print("="*70)

    # Test 1: Full dataset download (recommended mode)
    test1_success = test_full_dataset_download()

    # Test 2: Single month download (backwards compatibility)
    test2_success = test_single_month_download()

    # Summary
    print("\n" + "="*70)
    print("Test Suite Summary")
    print("="*70)
    print(f"  Full dataset download: {'✓ PASSED' if test1_success else '✗ FAILED'}")
    print(f"  Single month download: {'✓ PASSED' if test2_success else '✗ FAILED'}")
    print("="*70 + "\n")

    # Exit with appropriate code
    all_passed = test1_success and test2_success
    sys.exit(0 if all_passed else 1)
