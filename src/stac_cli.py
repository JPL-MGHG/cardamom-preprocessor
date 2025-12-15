"""
CARDAMOM STAC-Based Preprocessor Command Line Interface

This module provides CLI entry points for the decoupled STAC-based architecture,
allowing independent execution of downloaders and the CBF generator.

Usage:
    # Download ERA5 meteorological data
    python -m src.stac_cli ecmwf \
        --variables t2m_min,t2m_max,vpd \
        --year 2020 --month 1 \
        --output ./era5_output

    # Download NOAA CO2 data
    python -m src.stac_cli noaa \
        --year 2020 --month 1 \
        --output ./co2_output

    # Download GFED burned area
    python -m src.stac_cli gfed \
        --year 2020 --month 1 \
        --output ./gfed_output

    # Generate CBF files from STAC catalog
    python -m src.stac_cli cbf-generate \
        --stac-api https://stac.maap-project.org \
        --start 2020-01 --end 2020-12 \
        --region conus \
        --output ./cbf_output
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from downloaders.ecmwf_downloader import ECMWFDownloader
from downloaders.noaa_downloader import NOAADownloader
from downloaders.gfed_downloader import GFEDDownloader
from cbf_generator import CBFGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_variable_list(variables_str: str) -> List[str]:
    """Parse comma-separated variable list."""
    return [v.strip() for v in variables_str.split(',')]


def create_ecmwf_parser(subparsers) -> None:
    """Create ECMWF downloader subcommand."""

    parser = subparsers.add_parser(
        'ecmwf',
        help='Download ERA5 meteorological data from ECMWF'
    )

    parser.add_argument(
        '--variables',
        required=True,
        help='Comma-separated list of variables to download '
             '(t2m_min, t2m_max, vpd, total_prec, ssrd, strd, skt, snowfall)'
    )

    parser.add_argument(
        '--year',
        type=int,
        required=True,
        help='Year to download (e.g., 2020)'
    )

    parser.add_argument(
        '--month',
        type=int,
        required=True,
        help='Month to download (1-12)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory path'
    )

    parser.add_argument(
        '--keep-raw',
        action='store_true',
        help='Keep raw ERA5 files after processing (default: delete)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print debug information'
    )

    parser.set_defaults(func=handle_ecmwf_download)


def create_noaa_parser(subparsers) -> None:
    """Create NOAA downloader subcommand."""

    parser = subparsers.add_parser(
        'noaa',
        help='Download NOAA CO2 data'
    )

    parser.add_argument(
        '--year',
        type=int,
        required=False,
        default=None,
        help='Year to process (optional - if omitted, downloads all available data)'
    )

    parser.add_argument(
        '--month',
        type=int,
        required=False,
        default=None,
        help='Month to process (1-12) (optional - if omitted, downloads all available data)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory path'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print debug information'
    )

    parser.set_defaults(func=handle_noaa_download)


def create_gfed_parser(subparsers) -> None:
    """Create GFED downloader subcommand."""

    parser = subparsers.add_parser(
        'gfed',
        help='Download GFED burned area data'
    )

    parser.add_argument(
        '--year',
        type=int,
        required=True,
        help='Year to process'
    )

    parser.add_argument(
        '--month',
        type=int,
        required=True,
        help='Month to process (1-12)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory path'
    )

    parser.add_argument(
        '--keep-raw',
        action='store_true',
        help='Keep raw HDF5 files (default: delete)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print debug information'
    )

    parser.set_defaults(func=handle_gfed_download)


def create_cbf_parser(subparsers) -> None:
    """Create CBF generator subcommand."""

    parser = subparsers.add_parser(
        'cbf-generate',
        help='Generate CARDAMOM Binary Format (CBF) files from STAC data'
    )

    parser.add_argument(
        '--stac-api',
        required=True,
        help='URL to STAC API endpoint (e.g., https://stac.maap-project.org)'
    )

    parser.add_argument(
        '--start',
        required=True,
        help='Start date in YYYY-MM format'
    )

    parser.add_argument(
        '--end',
        required=True,
        help='End date in YYYY-MM format'
    )

    parser.add_argument(
        '--region',
        default='conus',
        choices=['global', 'conus'],
        help='Geographic region for CBF generation (default: conus)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory path'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print debug information'
    )

    parser.set_defaults(func=handle_cbf_generate)


def handle_ecmwf_download(args) -> int:
    """Handle ECMWF downloader invocation."""

    try:
        logger.info("Starting ECMWF downloader")

        # Parse variables
        variables = parse_variable_list(args.variables)
        logger.info(f"Requesting variables: {variables}")

        # Initialize downloader
        downloader = ECMWFDownloader(
            output_directory=args.output,
            keep_raw_files=args.keep_raw,
            verbose=args.verbose,
        )

        # Download and process
        results = downloader.download_and_process(
            variables=variables,
            year=args.year,
            month=args.month,
        )

        logger.info(f"✓ ECMWF download successful")
        logger.info(f"  Generated {len(results['output_files'])} files")
        logger.info(f"  Output directory: {args.output}")

        return 0

    except Exception as e:
        logger.error(f"✗ ECMWF download failed: {e}", exc_info=args.verbose)
        return 1


def handle_noaa_download(args) -> int:
    """Handle NOAA downloader invocation."""

    try:
        # Log what we're downloading
        if args.year is not None and args.month is not None:
            logger.info(f"Starting NOAA downloader for {args.year}-{args.month:02d}")
        else:
            logger.info("Starting NOAA downloader for entire available dataset")

        # Initialize downloader
        downloader = NOAADownloader(
            output_directory=args.output,
            verbose=args.verbose,
        )

        # Download and process
        results = downloader.download_and_process(
            year=args.year,
            month=args.month,
        )

        logger.info(f"✓ NOAA download successful")
        logger.info(f"  Generated {len(results['output_files'])} files")

        # Show additional info for full dataset download
        if 'time_range' in results:
            start_year, end_year = results['time_range']
            num_steps = results['num_time_steps']
            logger.info(f"  Time range: {start_year}-{end_year} ({num_steps} months)")

        logger.info(f"  Output directory: {args.output}")

        return 0

    except Exception as e:
        logger.error(f"✗ NOAA download failed: {e}", exc_info=args.verbose)
        return 1


def handle_gfed_download(args) -> int:
    """Handle GFED downloader invocation."""

    try:
        logger.info("Starting GFED downloader")

        # Initialize downloader
        downloader = GFEDDownloader(
            output_directory=args.output,
            keep_raw_files=args.keep_raw,
            verbose=args.verbose,
        )

        # Download and process
        results = downloader.download_and_process(
            year=args.year,
            month=args.month,
        )

        logger.info(f"✓ GFED download successful")
        logger.info(f"  Generated {len(results['output_files'])} files")
        logger.info(f"  Output directory: {args.output}")

        return 0

    except Exception as e:
        logger.error(f"✗ GFED download failed: {e}", exc_info=args.verbose)
        return 1


def handle_cbf_generate(args) -> int:
    """Handle CBF generator invocation."""

    try:
        logger.info("Starting CBF generator")

        # Initialize CBF generator
        generator = CBFGenerator(
            stac_api_url=args.stac_api,
            output_directory=args.output,
            verbose=args.verbose,
        )

        # Generate CBF files
        results = generator.generate(
            start_date=args.start,
            end_date=args.end,
            region=args.region,
        )

        logger.info(f"✓ CBF generation successful")
        logger.info(f"  Generated {results['metadata']['num_files']} CBF files")
        logger.info(f"  Output directory: {args.output}")
        logger.info(f"  Region: {results['metadata']['region']}")

        return 0

    except Exception as e:
        logger.error(f"✗ CBF generation failed: {e}", exc_info=args.verbose)
        return 1


def main(argv: List[str] = None) -> int:
    """
    Main CLI entry point.

    Args:
        argv: Command line arguments. If None, uses sys.argv[1:]

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """

    parser = argparse.ArgumentParser(
        description='CARDAMOM STAC-Based Preprocessor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Download ERA5 variables for 2020-01
  %(prog)s ecmwf --variables t2m_min,vpd --year 2020 --month 1 --output ./output

  # Download NOAA CO2 for entire available dataset (recommended - small file)
  %(prog)s noaa --output ./output

  # Download NOAA CO2 for specific month (backwards compatibility)
  %(prog)s noaa --year 2020 --month 1 --output ./output

  # Generate CBF files from STAC data
  %(prog)s cbf-generate --stac-api https://stac.maap.org \\
    --start 2020-01 --end 2020-12 --region conus --output ./cbf

For more information, see the STAC-Based Architecture Plan in plans/
        '''
    )

    subparsers = parser.add_subparsers(
        title='commands',
        description='Available commands',
        dest='command',
        help='Command to execute'
    )

    # Create subcommand parsers
    create_ecmwf_parser(subparsers)
    create_noaa_parser(subparsers)
    create_gfed_parser(subparsers)
    create_cbf_parser(subparsers)

    # Parse arguments
    args = parser.parse_args(argv)

    # Execute command
    if hasattr(args, 'func'):
        return args.func(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
