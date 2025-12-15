#!/usr/bin/env python3
"""
Generic ECMWF Data Downloader for CARDAMOM
Combines functionality from existing scripts with configurable parameters
Uses ecmwf-datastores-client for efficient batch processing
"""

import os
import argparse
import json
import time
from typing import List, Dict, Optional, Union, Tuple
from ecmwf.datastores import Client

from cardamom_variables import get_cbf_name, get_variables_by_product_type


class ECMWFDownloader:
    """Generic ECMWF data downloader with configurable parameters"""
    
    def __init__(self, 
                 area: List[float] = None,
                 grid: List[str] = None,
                 data_format: str = "netcdf",
                 download_format: str = "unarchived",
                 output_dir: str = "."):
        """
        Initialize ECMWF downloader
        
        Args:
            area: [North, West, South, East] bounding box
            grid: Grid resolution (default: 0.5/0.5)
            data_format: Output format (default: netcdf)
            download_format: Download format (default: unarchived)
            output_dir: Output directory for downloaded files
        """
        # Default to global coverage if not specified
        self.area = area or [-89.75, -179.75, 89.75, 179.75]
        self.grid = grid or ["0.5/0.5"]
        self.data_format = data_format
        self.download_format = download_format
        self.output_dir = output_dir
        self.client = Client()

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    
    def _build_variable_map(self, variables: List[str]) -> Dict[str, str]:
        """
        Build variable name to abbreviation mapping from registry.
        
        Args:
            variables: List of ERA5 variable names
            
        Returns:
            dict: Mapping of full variable names to CBF abbreviations
        """
        variable_map = {}
        for var in variables:
            cbf_name = get_cbf_name(var)
            if cbf_name:
                # Handle case where cbf_names is a list (e.g., TMIN/TMAX for temperature)
                if isinstance(cbf_name, list):
                    # Use first CBF name as abbreviation
                    variable_map[var] = cbf_name[0]
                else:
                    variable_map[var] = cbf_name
            else:
                # If no CBF name, use variable name as-is
                variable_map[var] = var
        return variable_map

    def _monitor_and_download_jobs(self, job_info_list: List[Tuple]) -> None:
        """
        Monitor submitted jobs and download results immediately when ready.
        
        This method combines monitoring and downloading into a single phase,
        allowing downloads to begin as soon as individual jobs complete rather
        than waiting for all jobs to finish.

        Args:
            job_info_list: List of tuples (remote, filepath, metadata_dict)
                          where metadata_dict contains year, month, variable info
        """
        total_jobs = len(job_info_list)
        print(f"\nMonitoring and downloading {total_jobs} jobs...")

        pending_jobs = list(job_info_list)
        downloaded_count = 0
        failed_count = 0

        while pending_jobs:
            time.sleep(5)  # Poll every 5 seconds

            still_pending = []
            for remote, filepath, metadata in pending_jobs:
                try:
                    status = remote.status

                    if remote.results_ready:
                        # Download immediately when ready
                        print(f"⬇ [{downloaded_count + 1}/{total_jobs}] Downloading: "
                              f"{metadata['variable']} {metadata['month']:02d}/{metadata['year']}")
                        try:
                            remote.download(filepath)
                            downloaded_count += 1
                            print(f"  ✓ Saved to {filepath}")
                        except Exception as download_error:
                            failed_count += 1
                            print(f"  ✗ Download failed: {download_error}")
                            
                    elif status == 'failed':
                        failed_count += 1
                        print(f"✗ Failed: {metadata['variable']} {metadata['month']:02d}/{metadata['year']}")
                    else:
                        # Still processing by ECMWF
                        still_pending.append((remote, filepath, metadata))

                except Exception as e:
                    print(f"⚠ Error checking status for {metadata['variable']} "
                          f"{metadata['month']:02d}/{metadata['year']}: {e}")
                    still_pending.append((remote, filepath, metadata))

            pending_jobs = still_pending

            if pending_jobs:
                print(f"⏳ Progress: {downloaded_count} downloaded, "
                      f"{len(pending_jobs)} pending, {failed_count} failed", end='\r')

        print(f"\n\n✓ Complete: {downloaded_count} downloaded, {failed_count} failed")

    def download_hourly_data(self,
                           variables: Union[str, List[str]],
                           years: Union[int, List[int]],
                           months: Union[int, List[int]],
                           days: List[str] = None,
                           times: List[str] = None,
                           dataset: str = "reanalysis-era5-single-levels",
                           file_prefix: str = "ECMWF_HOURLY",
                           variable_map: Dict[str, str] = None) -> None:
        """
        Download hourly ECMWF data using batch submission.

        Args:
            variables: Variable name(s) to download
            years: Year(s) to download
            months: Month(s) to download
            days: Days to download (default: all days)
            times: Times to download (default: all hours)
            dataset: ECMWF dataset name (collection ID)
            file_prefix: Prefix for output files
            variable_map: Map full variable names to abbreviations
        """
        # Ensure lists
        if isinstance(variables, str):
            variables = [variables]
        if isinstance(years, int):
            years = [years]
        if isinstance(months, int):
            months = [months]

        # Default days and times
        if days is None:
            days = [f"{i:02d}" for i in range(1, 32)]
        if times is None:
            times = [f"{i:02d}:00" for i in range(24)]

        # Default variable map
        if variable_map is None:
            variable_map = {}

        # Phase 1: Submit all requests (non-blocking)
        print(f"\n=== Phase 1: Submitting batch requests ===")
        job_info_list = []
        skipped_count = 0

        for year in years:
            for month in months:
                for variable in variables:
                    # Use abbreviation if available, otherwise use variable name
                    var_abbr = variable_map.get(variable, variable)
                    filename = f"{file_prefix}_{var_abbr}_{month:02d}{year}.nc"
                    filepath = os.path.join(self.output_dir, filename)

                    if os.path.exists(filepath):
                        print(f"  ⊙ File '{filename}' already exists. Skipping.")
                        skipped_count += 1
                        continue

                    request = {
                        "product_type": ["reanalysis"],
                        "variable": variable,
                        "year": str(year),
                        "month": f"{month:02d}",
                        "day": days,
                        "time": times,
                        "data_format": self.data_format,
                        "grid": self.grid,
                        "download_format": self.download_format,
                        "area": self.area
                    }

                    try:
                        print(f"  → Submitting: {variable} {month:02d}/{year}")
                        remote = self.client.submit(dataset, request)

                        metadata = {
                            'variable': variable,
                            'year': year,
                            'month': month,
                            'filename': filename
                        }
                        job_info_list.append((remote, filepath, metadata))

                    except Exception as e:
                        print(f"  ✗ Error submitting {variable} {month:02d}/{year}: {e}")

        if skipped_count > 0:
            print(f"\n⊙ Skipped {skipped_count} existing files")

        if not job_info_list:
            print("\n✓ No new files to download")
            return

        print(f"\n✓ Submitted {len(job_info_list)} requests")

        # Phase 2: Monitor and download (combined)
        print(f"\n=== Phase 2: Monitoring and downloading ===")
        self._monitor_and_download_jobs(job_info_list)
    
    def download_monthly_data(self,
                            variables: Union[str, List[str]],
                            years: Union[int, List[int]],
                            months: Union[int, List[int]],
                            product_type: str = "monthly_averaged_reanalysis",
                            times: List[str] = None,
                            dataset: str = "reanalysis-era5-single-levels-monthly-means",
                            file_prefix: str = "ECMWF_MONTHLY") -> None:
        """
        Download monthly ECMWF data using batch submission.

        Args:
            variables: Variable name(s) to download
            years: Year(s) to download
            months: Month(s) to download
            product_type: Type of monthly data
            times: Times for hourly averaged data (default: ["00:00", "01:00"])
            dataset: ECMWF dataset name (collection ID)
            file_prefix: Prefix for output files
        """
        # Ensure lists
        if isinstance(variables, str):
            variables = [variables]
        if isinstance(years, int):
            years = [years]
        if isinstance(months, int):
            months = [months]

        # Default times for monthly data
        if times is None:
            if "by_hour" in product_type:
                times = [f"{i:02d}:00" for i in range(24)]
            else:
                times = ["00:00", "01:00"]

        # Phase 1: Submit all requests (non-blocking)
        print(f"\n=== Phase 1: Submitting batch requests ===")
        job_info_list = []
        skipped_count = 0

        for year in years:
            for month in months:
                for variable in variables:
                    filename = f"{file_prefix}_{variable}_{month:02d}{year}.nc"
                    filepath = os.path.join(self.output_dir, filename)

                    if os.path.exists(filepath):
                        print(f"  ⊙ File '{filename}' already exists. Skipping.")
                        skipped_count += 1
                        continue

                    request = {
                        "product_type": [product_type],
                        "variable": variable,
                        "year": str(year),
                        "month": f"{month:02d}",
                        "time": times,
                        "data_format": self.data_format,
                        "grid": self.grid,
                        "download_format": self.download_format,
                        "area": self.area
                    }

                    try:
                        print(f"  → Submitting: {variable} {month:02d}/{year}")
                        remote = self.client.submit(dataset, request)

                        metadata = {
                            'variable': variable,
                            'year': year,
                            'month': month,
                            'filename': filename
                        }
                        job_info_list.append((remote, filepath, metadata))

                    except Exception as e:
                        print(f"  ✗ Error submitting {variable} {month:02d}/{year}: {e}")

        if skipped_count > 0:
            print(f"\n⊙ Skipped {skipped_count} existing files")

        if not job_info_list:
            print("\n✓ No new files to download")
            return

        print(f"\n✓ Submitted {len(job_info_list)} requests")

        # Phase 2: Monitor and download (combined)
        print(f"\n=== Phase 2: Monitoring and downloading ===")
        self._monitor_and_download_jobs(job_info_list)


# Example usage configurations
def download_cardamom_hourly_drivers():
    """Download hourly drivers for CARDAMOM (CONUS region)"""
    # CONUS region
    conus_area = [60, -130, 20, -50]

    # Get hourly variables from registry (for CARDAMOM hourly drivers, we use a specific subset)
    hourly_driver_variables = ["skin_temperature", "surface_solar_radiation_downwards"]

    downloader = ECMWFDownloader(area=conus_area, output_dir="./hourly_data")

    # Build variable map from registry
    variable_map = downloader._build_variable_map(hourly_driver_variables)

    downloader.download_hourly_data(
        variables=hourly_driver_variables,
        years=list(range(2015, 2021)),
        months=list(range(1, 13)),
        file_prefix="ECMWF_CARDAMOM_HOURLY_DRIVER",
        variable_map=variable_map
    )


def download_cardamom_monthly_drivers():
    """Download monthly drivers for CARDAMOM (Global)"""
    # Global area
    global_area = [-89.75, -179.75, 89.75, 179.75]

    # Get variables for different processing types from registry
    hourly_quantities = get_variables_by_product_type("monthly_averaged_reanalysis_by_hour_of_day")
    monthly_quantities = get_variables_by_product_type("monthly_averaged_reanalysis")

    downloader = ECMWFDownloader(area=global_area, output_dir="./monthly_data")

    # Download hourly averaged monthly data
    downloader.download_monthly_data(
        variables=hourly_quantities,
        years=list(range(2001, 2025)),
        months=list(range(1, 13)),
        product_type="monthly_averaged_reanalysis_by_hour_of_day",
        file_prefix="ECMWF_CARDAMOM_DRIVER"
    )

    # Download monthly averaged data
    downloader.download_monthly_data(
        variables=monthly_quantities,
        years=list(range(2001, 2025)),
        months=list(range(1, 13)),
        product_type="monthly_averaged_reanalysis",
        file_prefix="ECMWF_CARDAMOM_DRIVER"
    )


def parse_range(value):
    """Parse range string like '2020-2022' or single value '2020'"""
    if '-' in value:
        start, end = map(int, value.split('-'))
        return list(range(start, end + 1))
    else:
        return [int(value)]


def parse_list(value):
    """Parse comma-separated list"""
    return [item.strip() for item in value.split(',')]


def create_parser():
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="Download ECMWF ERA5 data for CARDAMOM preprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download monthly temperature data for 2020-2021
  python ecmwf_downloader.py monthly -v 2m_temperature -y 2020-2021 -m 1-12
  
  # Download hourly precipitation for summer 2020 (CONUS)
  python ecmwf_downloader.py hourly -v total_precipitation -y 2020 -m 6-8 --area 60,-130,20,-50
  
  # Use predefined CARDAMOM configurations
  python ecmwf_downloader.py cardamom-hourly
  python ecmwf_downloader.py cardamom-monthly
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Download type')
    
    # Common arguments
    def add_common_args(parser):
        parser.add_argument('-o', '--output-dir', default='./ecmwf_data',
                          help='Output directory (default: ./ecmwf_data)')
        parser.add_argument('--area', type=str,
                          help='Area bounds as N,W,S,E (default: global)')
        parser.add_argument('--grid', default='0.5/0.5',
                          help='Grid resolution (default: 0.5/0.5)')
        parser.add_argument('--format', choices=['netcdf', 'grib'], default='netcdf',
                          help='Data format (default: netcdf)')
    
    # Hourly data subcommand
    hourly_parser = subparsers.add_parser('hourly', help='Download hourly data')
    add_common_args(hourly_parser)
    hourly_parser.add_argument('-v', '--variables', required=True, type=parse_list,
                              help='Comma-separated list of variables')
    hourly_parser.add_argument('-y', '--years', required=True, type=parse_range,
                              help='Years (single: 2020, range: 2020-2022)')
    hourly_parser.add_argument('-m', '--months', required=True, type=parse_range,
                              help='Months (single: 6, range: 6-8)')
    hourly_parser.add_argument('--dataset', default='reanalysis-era5-single-levels',
                              help='ECMWF dataset name')
    hourly_parser.add_argument('--prefix', default='ECMWF_HOURLY',
                              help='File prefix (default: ECMWF_HOURLY)')
    hourly_parser.add_argument('--var-map', type=str,
                              help='JSON file with variable name mappings')
    
    # Monthly data subcommand
    monthly_parser = subparsers.add_parser('monthly', help='Download monthly data')
    add_common_args(monthly_parser)
    monthly_parser.add_argument('-v', '--variables', required=True, type=parse_list,
                               help='Comma-separated list of variables')
    monthly_parser.add_argument('-y', '--years', required=True, type=parse_range,
                               help='Years (single: 2020, range: 2020-2022)')
    monthly_parser.add_argument('-m', '--months', required=True, type=parse_range,
                               help='Months (single: 6, range: 6-8)')
    monthly_parser.add_argument('--product-type', 
                               choices=['monthly_averaged_reanalysis', 
                                       'monthly_averaged_reanalysis_by_hour_of_day'],
                               default='monthly_averaged_reanalysis',
                               help='Product type (default: monthly_averaged_reanalysis)')
    monthly_parser.add_argument('--dataset', default='reanalysis-era5-single-levels-monthly-means',
                               help='ECMWF dataset name')
    monthly_parser.add_argument('--prefix', default='ECMWF_MONTHLY',
                               help='File prefix (default: ECMWF_MONTHLY)')
    
    # Predefined configurations
    cardamom_hourly_parser = subparsers.add_parser('cardamom-hourly',
                                                  help='Download CARDAMOM hourly drivers (CONUS)')
    cardamom_hourly_parser.add_argument('-o', '--output-dir', default='./hourly_data',
                                       help='Output directory (default: ./hourly_data)')
    cardamom_hourly_parser.add_argument('-y', '--years', type=parse_range, default=[2015, 2016, 2017, 2018, 2019, 2020],
                                       help='Years (default: 2015-2020)')
    
    cardamom_monthly_parser = subparsers.add_parser('cardamom-monthly',
                                                   help='Download CARDAMOM monthly drivers (Global)')
    cardamom_monthly_parser.add_argument('-o', '--output-dir', default='./monthly_data',
                                        help='Output directory (default: ./monthly_data)')
    cardamom_monthly_parser.add_argument('-y', '--years', type=parse_range, default=list(range(2001, 2025)),
                                        help='Years (default: 2001-2024)')
    
    return parser


def main():
    """Main CLI function"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'hourly':
            # Parse area if provided
            area = None
            if args.area:
                area = [float(x.strip()) for x in args.area.split(',')]
            
            # Load variable mapping if provided
            variable_map = {}
            if args.var_map:
                with open(args.var_map, 'r') as f:
                    variable_map = json.load(f)
            
            downloader = ECMWFDownloader(
                area=area,
                grid=[args.grid],
                data_format=args.format,
                output_dir=args.output_dir
            )
            
            downloader.download_hourly_data(
                variables=args.variables,
                years=args.years,
                months=args.months,
                dataset=args.dataset,
                file_prefix=args.prefix,
                variable_map=variable_map
            )
            
        elif args.command == 'monthly':
            # Parse area if provided
            area = None
            if args.area:
                area = [float(x.strip()) for x in args.area.split(',')]
            
            downloader = ECMWFDownloader(
                area=area,
                grid=[args.grid],
                data_format=args.format,
                output_dir=args.output_dir
            )
            
            downloader.download_monthly_data(
                variables=args.variables,
                years=args.years,
                months=args.months,
                product_type=args.product_type,
                dataset=args.dataset,
                file_prefix=args.prefix
            )
            
        elif args.command == 'cardamom-hourly':
            print("Downloading CARDAMOM hourly drivers (CONUS region)...")
            # CONUS region
            conus_area = [60, -130, 20, -50]

            # Get hourly driver variables (specific subset for CARDAMOM hourly)
            hourly_driver_variables = ["skin_temperature", "surface_solar_radiation_downwards"]

            downloader = ECMWFDownloader(area=conus_area, output_dir=args.output_dir)

            # Build variable map from registry
            variable_map = downloader._build_variable_map(hourly_driver_variables)

            downloader.download_hourly_data(
                variables=hourly_driver_variables,
                years=args.years,
                months=list(range(1, 13)),
                file_prefix="ECMWF_CARDAMOM_HOURLY_DRIVER",
                variable_map=variable_map
            )
            
        elif args.command == 'cardamom-monthly':
            print("Downloading CARDAMOM monthly drivers (Global region)...")
            # Global area
            global_area = [-89.75, -179.75, 89.75, 179.75]

            # Get variables for different processing types from registry
            hourly_quantities = get_variables_by_product_type("monthly_averaged_reanalysis_by_hour_of_day")
            monthly_quantities = get_variables_by_product_type("monthly_averaged_reanalysis")

            downloader = ECMWFDownloader(area=global_area, output_dir=args.output_dir)

            # Download hourly averaged monthly data
            print("Downloading hourly averaged monthly data...")
            downloader.download_monthly_data(
                variables=hourly_quantities,
                years=args.years,
                months=list(range(1, 13)),
                product_type="monthly_averaged_reanalysis_by_hour_of_day",
                file_prefix="ECMWF_CARDAMOM_DRIVER"
            )

            # Download monthly averaged data
            print("Downloading monthly averaged data...")
            downloader.download_monthly_data(
                variables=monthly_quantities,
                years=args.years,
                months=list(range(1, 13)),
                product_type="monthly_averaged_reanalysis",
                file_prefix="ECMWF_CARDAMOM_DRIVER"
            )
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    print("Download completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())