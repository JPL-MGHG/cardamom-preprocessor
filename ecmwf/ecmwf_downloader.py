#!/usr/bin/env python3
"""
Generic ECMWF Data Downloader for CARDAMOM
Combines functionality from existing scripts with configurable parameters
"""

import cdsapi
import os
import argparse
import json
from typing import List, Dict, Optional, Union


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
        self.client = cdsapi.Client()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
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
        Download hourly ECMWF data
        
        Args:
            variables: Variable name(s) to download
            years: Year(s) to download
            months: Month(s) to download
            days: Days to download (default: all days)
            times: Times to download (default: all hours)
            dataset: ECMWF dataset name
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
        
        for year in years:
            for month in months:
                for variable in variables:
                    # Use abbreviation if available, otherwise use variable name
                    var_abbr = variable_map.get(variable, variable)
                    filename = f"{file_prefix}_{var_abbr}_{month:02d}{year}.nc"
                    filepath = os.path.join(self.output_dir, filename)
                    
                    if os.path.exists(filepath):
                        print(f"File '{filename}' already exists. Skipping download.")
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
                    
                    print(f"Downloading {variable} for {month:02d}/{year}...")
                    self.client.retrieve(dataset, request).download(filepath)
                    print(f"Saved to {filepath}")
    
    def download_monthly_data(self,
                            variables: Union[str, List[str]],
                            years: Union[int, List[int]],
                            months: Union[int, List[int]],
                            product_type: str = "monthly_averaged_reanalysis",
                            times: List[str] = None,
                            dataset: str = "reanalysis-era5-single-levels-monthly-means",
                            file_prefix: str = "ECMWF_MONTHLY") -> None:
        """
        Download monthly ECMWF data
        
        Args:
            variables: Variable name(s) to download
            years: Year(s) to download
            months: Month(s) to download
            product_type: Type of monthly data
            times: Times for hourly averaged data (default: ["00:00", "01:00"])
            dataset: ECMWF dataset name
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
        
        for year in years:
            for month in months:
                for variable in variables:
                    filename = f"{file_prefix}_{variable}_{month:02d}{year}.nc"
                    filepath = os.path.join(self.output_dir, filename)
                    
                    if os.path.exists(filepath):
                        print(f"File '{filename}' already exists. Skipping download.")
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
                    
                    print(f"Downloading {variable} for {month:02d}/{year}...")
                    self.client.retrieve(dataset, request).download(filepath)
                    print(f"Saved to {filepath}")


# Example usage configurations
def download_cardamom_hourly_drivers():
    """Download hourly drivers for CARDAMOM (CONUS region)"""
    # CONUS region
    conus_area = [60, -130, 20, -50]
    
    # Variable mapping for abbreviations
    variable_map = {
        "skin_temperature": "SKT",
        "surface_solar_radiation_downwards": "SSRD"
    }
    
    downloader = ECMWFDownloader(area=conus_area, output_dir="./hourly_data")
    
    downloader.download_hourly_data(
        variables=list(variable_map.keys()),
        years=list(range(2015, 2021)),
        months=list(range(1, 13)),
        file_prefix="ECMWF_CARDAMOM_HOURLY_DRIVER",
        variable_map=variable_map
    )


def download_cardamom_monthly_drivers():
    """Download monthly drivers for CARDAMOM (Global)"""
    # Global area
    global_area = [-89.75, -179.75, 89.75, 179.75]
    
    # Variables for different processing types
    hourly_quantities = ["2m_temperature", "2m_dewpoint_temperature"]
    monthly_quantities = ["total_precipitation", "skin_temperature", 
                         "surface_solar_radiation_downwards", "snowfall"]
    
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
            
            # Variable mapping for abbreviations
            variable_map = {
                "skin_temperature": "SKT",
                "surface_solar_radiation_downwards": "SSRD"
            }
            
            downloader = ECMWFDownloader(area=conus_area, output_dir=args.output_dir)
            
            downloader.download_hourly_data(
                variables=list(variable_map.keys()),
                years=args.years,
                months=list(range(1, 13)),
                file_prefix="ECMWF_CARDAMOM_HOURLY_DRIVER",
                variable_map=variable_map
            )
            
        elif args.command == 'cardamom-monthly':
            print("Downloading CARDAMOM monthly drivers (Global region)...")
            # Global area
            global_area = [-89.75, -179.75, 89.75, 179.75]
            
            # Variables for different processing types
            hourly_quantities = ["2m_temperature", "2m_dewpoint_temperature"]
            monthly_quantities = ["total_precipitation", "skin_temperature", 
                                 "surface_solar_radiation_downwards", "snowfall"]
            
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