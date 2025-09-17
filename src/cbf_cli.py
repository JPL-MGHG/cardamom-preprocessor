#!/usr/bin/env python3
"""
CBF Meteorological Processor CLI

Command-line interface for processing downloaded ERA5 meteorological data
into CARDAMOM Binary Format (CBF) compatible files.

Usage examples:
    # Process downloaded ERA5 files into CBF format
    python cbf_cli.py process-met ./downloaded_era5/ --output AllMet05x05_LFmasked.nc

    # Process with land masking
    python cbf_cli.py process-met ./downloaded_era5/ --land-fraction land_frac.nc --output AllMet05x05_LFmasked.nc

    # Process with external data integration
    python cbf_cli.py process-met ./downloaded_era5/ --co2-data ./co2_data/ --fire-data ./fire_data/
"""

import argparse
import sys
import os
from pathlib import Path

from cbf_met_processor import CBFMetProcessor
import logging


def create_parser():
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="CBF Meteorological Data Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic processing
  %(prog)s process-met ./era5_downloads/ --output AllMet05x05_LFmasked.nc

  # With land masking
  %(prog)s process-met ./era5_downloads/ \\
    --land-fraction input/CARDAMOM-MAPS_05deg_LAND_SEA_FRAC.nc \\
    --land-threshold 0.5 \\
    --output AllMet05x05_LFmasked.nc

  # With external data
  %(prog)s process-met ./era5_downloads/ \\
    --co2-data ./noaa_co2/ \\
    --fire-data ./gfed_fire/ \\
    --output AllMet05x05_LFmasked.nc

  # List supported variables
  %(prog)s list-variables
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Process meteorological data command
    process_parser = subparsers.add_parser(
        'process-met',
        help='Process downloaded ERA5 files into CBF meteorological drivers'
    )
    process_parser.add_argument(
        'input_dir',
        help='Directory containing downloaded ERA5 NetCDF files'
    )
    process_parser.add_argument(
        '--output', '-o',
        default='AllMet05x05_LFmasked.nc',
        help='Output CBF filename (default: AllMet05x05_LFmasked.nc)'
    )
    process_parser.add_argument(
        '--output-dir',
        default='.',
        help='Output directory (default: current directory)'
    )
    process_parser.add_argument(
        '--land-fraction',
        help='Path to land fraction NetCDF file for masking'
    )
    process_parser.add_argument(
        '--land-threshold',
        type=float,
        default=0.5,
        help='Land fraction threshold for masking (default: 0.5)'
    )
    process_parser.add_argument(
        '--co2-data',
        help='Directory containing NOAA CO2 data files'
    )
    process_parser.add_argument(
        '--fire-data',
        help='Directory containing GFED fire data files'
    )

    # List variables command
    list_parser = subparsers.add_parser(
        'list-variables',
        help='List supported CBF variables and ERA5 requirements'
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        'validate',
        help='Validate CBF file for compatibility'
    )
    validate_parser.add_argument(
        'cbf_file',
        help='Path to CBF NetCDF file to validate'
    )

    return parser


def process_meteorological_data(args):
    """Process meteorological data into CBF format."""
    logger = logging.getLogger(__name__)

    # Validate input directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        return 1

    if not input_path.is_dir():
        logger.error(f"Input path is not a directory: {input_path}")
        return 1

    # Check for NetCDF files in input directory
    nc_files = list(input_path.glob("*.nc"))
    if not nc_files:
        logger.error(f"No NetCDF files found in input directory: {input_path}")
        return 1

    logger.info(f"Found {len(nc_files)} NetCDF files in input directory")

    # Validate optional inputs
    if args.land_fraction and not os.path.exists(args.land_fraction):
        logger.error(f"Land fraction file does not exist: {args.land_fraction}")
        return 1

    if args.co2_data and not os.path.exists(args.co2_data):
        logger.error(f"CO2 data directory does not exist: {args.co2_data}")
        return 1

    if args.fire_data and not os.path.exists(args.fire_data):
        logger.error(f"Fire data directory does not exist: {args.fire_data}")
        return 1

    try:
        # Initialize processor
        processor = CBFMetProcessor(output_dir=args.output_dir)

        # Process files
        output_file = processor.process_downloaded_files_to_cbf_met(
            input_dir=str(input_path),
            output_filename=args.output,
            land_fraction_file=args.land_fraction,
            land_threshold=args.land_threshold,
            co2_data_dir=args.co2_data,
            fire_data_dir=args.fire_data
        )

        logger.info(f"CBF processing completed successfully!")
        logger.info(f"Output file: {output_file}")
        return 0

    except Exception as e:
        logger.error(f"CBF processing failed: {e}")
        return 1


def list_supported_variables():
    """List supported CBF variables and ERA5 requirements."""
    processor = CBFMetProcessor()

    print("CBF Meteorological Variables (Target Output):")
    print("=" * 50)
    cbf_vars = processor.get_supported_variables()
    for i, var in enumerate(cbf_vars, 1):
        print(f"{i:2d}. {var}")

    print("\nRequired ERA5 Input Variables:")
    print("=" * 35)
    era5_vars = processor.get_era5_requirements()
    for i, var in enumerate(era5_vars, 1):
        print(f"{i:2d}. {var}")

    print("\nVariable Mapping:")
    print("=" * 20)
    mapping = processor.era5_to_cbf_mapping
    for era5_var, cbf_var in mapping.items():
        if isinstance(cbf_var, list):
            cbf_str = ", ".join(cbf_var)
        else:
            cbf_str = cbf_var
        print(f"  {era5_var:<35} -> {cbf_str}")

    print("\nDerived Variables:")
    print("=" * 20)
    print("  VPD: Calculated from 2m_temperature + 2m_dewpoint_temperature")
    print("  TMIN/TMAX: Derived from 2m_temperature monthly extremes")

    print("\nExternal Data (Optional):")
    print("=" * 30)
    print("  CO2_2: NOAA CO2 concentration data (fallback: constant 415 ppm)")
    print("  BURN_2: GFED fire data (fallback: zeros)")
    print("  DISTURBANCE_FLUX: Framework variable (zeros)")
    print("  YIELD: Framework variable (zeros)")

    return 0


def validate_cbf_file(args):
    """Validate CBF file for compatibility."""
    logger = logging.getLogger(__name__)

    cbf_path = Path(args.cbf_file)
    if not cbf_path.exists():
        logger.error(f"CBF file does not exist: {cbf_path}")
        return 1

    try:
        processor = CBFMetProcessor()
        is_valid = processor._validate_cbf_file(str(cbf_path))

        if is_valid:
            print(f"✓ CBF file validation PASSED: {cbf_path.name}")
            return 0
        else:
            print(f"✗ CBF file validation FAILED: {cbf_path.name}")
            return 1

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == 'process-met':
        return process_meteorological_data(args)
    elif args.command == 'list-variables':
        return list_supported_variables()
    elif args.command == 'validate':
        return validate_cbf_file(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())