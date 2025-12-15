"""
Base Downloader Class for CARDAMOM Preprocessor

This module provides an abstract base class that all specific downloaders
(ECMWF, NOAA, GFED) inherit from to ensure consistent behavior and output formats.

Scientific Context:
A common base class ensures all downloaders produce:
1. NetCDF files with standardized dimensions and coordinates
2. CF-1.8 convention compliance
3. STAC metadata in consistent format
4. CARDAMOM variable naming conventions

This reduces code duplication and makes the system more maintainable.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime
import logging
import xarray as xr
import numpy as np
from src.stac_utils import create_stac_collection, create_stac_item, write_stac_output

# Configure logging for downloaders
logger = logging.getLogger(__name__)


class BaseDownloader(ABC):
    """
    Abstract base class for CARDAMOM data downloaders.

    Each concrete downloader (ECMWF, NOAA, GFED) inherits from this class
    and implements data source-specific download and processing logic.

    Attributes:
        output_directory (Path): Root directory for output files
        keep_raw_files (bool): Whether to retain raw/intermediate files
        data_subdir (str): Subdirectory name for processed data files
        stac_subdir (str): Subdirectory name for STAC metadata
        raw_subdir (str): Subdirectory name for raw/intermediate files
    """

    # Subdirectory names (can be overridden by subclasses)
    data_subdir = 'data'
    stac_subdir = 'stac'
    raw_subdir = 'raw'

    def __init__(
        self,
        output_directory: str,
        keep_raw_files: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize the base downloader.

        Args:
            output_directory (str): Root directory for all output files
            keep_raw_files (bool): If True, retain intermediate/raw files.
                Default: False (clean up after processing)
            verbose (bool): If True, print debug information. Default: False
        """

        self.output_directory = Path(output_directory)
        self.keep_raw_files = keep_raw_files
        self.verbose = verbose

        # Create output subdirectories
        self._setup_output_directories()

        # Configure logging
        if self.verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        logger.info(f"Initialized {self.__class__.__name__}")
        logger.info(f"Output directory: {self.output_directory}")

    def _setup_output_directories(self) -> None:
        """Create output directory structure."""

        (self.output_directory / self.data_subdir).mkdir(parents=True, exist_ok=True)
        (self.output_directory / self.stac_subdir).mkdir(parents=True, exist_ok=True)

        if self.keep_raw_files:
            (self.output_directory / self.raw_subdir).mkdir(parents=True, exist_ok=True)

        logger.debug(f"Created output directories under {self.output_directory}")

    @abstractmethod
    def download_and_process(
        self,
        year: int,
        month: int,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Download and process data for a specific year and month.

        This method must be implemented by each concrete downloader class.

        Args:
            year (int): Year to download (e.g., 2020)
            month (int): Month to download (1-12)
            **kwargs: Additional arguments passed to specific downloader

        Returns:
            Dict[str, Any]: Results dictionary containing:
                - 'output_files': List of generated NetCDF file paths
                - 'stac_items': List of created STAC Item objects
                - 'collection_id': STAC Collection ID
                - 'success': bool indicating overall success

        Raises:
            NotImplementedError: Must be implemented by subclasses
            ValueError: If year/month parameters are invalid
        """

        raise NotImplementedError(
            f"{self.__class__.__name__} must implement download_and_process()"
        )

    def validate_temporal_parameters(self, year: int, month: int) -> None:
        """
        Validate year and month parameters for physical reasonableness.

        Args:
            year (int): Year to validate
            month (int): Month to validate (1-12)

        Raises:
            ValueError: If parameters are outside valid ranges
        """

        if not isinstance(year, int) or year < 1900 or year > 2100:
            raise ValueError(
                f"Year must be integer between 1900 and 2100. Got: {year}"
            )

        if not isinstance(month, int) or month < 1 or month > 12:
            raise ValueError(
                f"Month must be integer between 1 and 12. Got: {month}"
            )

        logger.debug(f"Validated temporal parameters: {year}-{month:02d}")

    def create_standard_netcdf_dataset(
        self,
        variable_data: Dict[str, np.ndarray],
        latitude_extent: tuple = (-89.75, 89.75),
        longitude_extent: tuple = (-179.75, 179.75),
        resolution_degrees: float = 0.5,
        year: int = None,
        month: int = None,
    ) -> xr.Dataset:
        """
        Create a standardized xarray Dataset for CARDAMOM.

        This ensures all downloaders produce NetCDF files with consistent
        dimensions, coordinates, and encoding.

        Scientific Context:
        Standard grid coordinates allow CARDAMOM and its downstream tools
        to reliably combine data from multiple sources without reprojection.

        Args:
            variable_data (Dict[str, np.ndarray]): Dictionary mapping variable names
                to 2D arrays [latitude, longitude]
            latitude_extent (tuple): (min_lat, max_lat) in decimal degrees.
                Default: global -89.75 to 89.75
            longitude_extent (tuple): (min_lon, max_lon) in decimal degrees.
                Default: global -179.75 to 179.75
            resolution_degrees (float): Grid resolution in degrees.
                Default: 0.5 (CARDAMOM standard)
            year (int): Year for time coordinate
            month (int): Month for time coordinate

        Returns:
            xr.Dataset: Standardized xarray Dataset with CF-1.8 conventions

        Example:
            >>> downloader = ECMWFDownloader('./output')
            >>> temp_data = {'t2m': np.random.rand(360, 720)}
            >>> ds = downloader.create_standard_netcdf_dataset(
            ...     temp_data, year=2020, month=1
            ... )
        """

        # Create coordinate arrays
        num_lat = int((latitude_extent[1] - latitude_extent[0]) / resolution_degrees) + 1
        num_lon = int((longitude_extent[1] - longitude_extent[0]) / resolution_degrees) + 1

        latitude_coordinates = np.linspace(
            latitude_extent[0], latitude_extent[1], num_lat
        )
        longitude_coordinates = np.linspace(
            longitude_extent[0], longitude_extent[1], num_lon
        )

        # Create time coordinate if year/month provided
        time_coordinates = None
        if year is not None and month is not None:
            time_coordinates = [datetime(year, month, 1)]

        # Create data arrays for each variable
        data_arrays = {}
        for var_name, data_array in variable_data.items():
            if time_coordinates:
                data_arrays[var_name] = xr.DataArray(
                    data=np.expand_dims(data_array, axis=0),  # Add time dimension
                    coords={
                        'time': time_coordinates,
                        'latitude': latitude_coordinates,
                        'longitude': longitude_coordinates,
                    },
                    dims=['time', 'latitude', 'longitude'],
                    name=var_name,
                )
            else:
                data_arrays[var_name] = xr.DataArray(
                    data=data_array,
                    coords={
                        'latitude': latitude_coordinates,
                        'longitude': longitude_coordinates,
                    },
                    dims=['latitude', 'longitude'],
                    name=var_name,
                )

        # Combine into dataset
        dataset = xr.Dataset(data_arrays)

        # Add CF-1.8 convention metadata
        dataset.attrs['Conventions'] = 'CF-1.8'
        dataset.attrs['history'] = f'Created by CARDAMOM preprocessor at {datetime.now().isoformat()}'
        dataset.attrs['source'] = self.__class__.__name__

        # Set proper encoding
        if time_coordinates:
            dataset['time'].encoding = {
                'units': 'days since 2001-01-01',
                'calendar': 'standard',
            }

        logger.debug(
            f"Created standardized NetCDF dataset with shape "
            f"{dataset[list(data_arrays.keys())[0]].shape}"
        )

        return dataset

    def write_netcdf_file(
        self,
        dataset: xr.Dataset,
        filename: str,
        variable_units: Optional[Dict[str, str]] = None,
        fill_value: float = -9999.0,
    ) -> Path:
        """
        Write xarray Dataset to NetCDF file with standard CARDAMOM encoding.

        Args:
            dataset (xr.Dataset): Data to write
            filename (str): Output filename (without path)
            variable_units (Optional[Dict]): Dictionary mapping variable names to units
            fill_value (float): NetCDF fill value for missing data.
                Default: -9999.0

        Returns:
            Path: Full path to written file

        Example:
            >>> output_file = downloader.write_netcdf_file(
            ...     dataset, 't2m_min_2020_01.nc',
            ...     variable_units={'t2m': 'K'}
            ... )
        """

        output_file = self.output_directory / self.data_subdir / filename

        # Set standard encoding for all variables
        encoding_dict = {}
        for var in dataset.data_vars:
            encoding_dict[var] = {
                'dtype': 'float32',
                '_FillValue': fill_value,
                'zlib': True,  # Enable compression
                'complevel': 4,
            }

        # Add units to variable attributes
        if variable_units:
            for var_name, units in variable_units.items():
                if var_name in dataset.data_vars:
                    dataset[var_name].attrs['units'] = units

        # Write to NetCDF
        dataset.to_netcdf(output_file, encoding=encoding_dict)

        logger.info(f"Wrote NetCDF file: {output_file}")
        return output_file

    def create_and_write_stac_metadata(
        self,
        collection_id: str,
        collection_description: str,
        collection_keywords: List[str],
        items_data: List[Dict[str, Any]],
        temporal_start: datetime,
        temporal_end: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Create STAC Collection and Items, then write to filesystem.

        Args:
            collection_id (str): Unique collection identifier
            collection_description (str): Human-readable collection description
            collection_keywords (List[str]): Keywords for searching the collection
            items_data (List[Dict]): List of item metadata dictionaries, each containing:
                - 'variable_name': str
                - 'year': int
                - 'month': int
                - 'data_file_path': str
                - 'bbox': Optional list [min_lon, min_lat, max_lon, max_lat]
                - 'properties': Optional dict of additional properties
            temporal_start (datetime): Start of temporal coverage
            temporal_end (Optional[datetime]): End of temporal coverage

        Returns:
            Dict[str, Any]: Dictionary with keys:
                - 'collection': pystac.Collection object
                - 'items': List of pystac.Item objects
                - 'stac_output_dir': Path where STAC files were written
        """

        # Create STAC Collection
        collection = create_stac_collection(
            collection_id=collection_id,
            description=collection_description,
            keywords=collection_keywords,
            temporal_start=temporal_start,
            temporal_end=temporal_end,
        )

        # Create STAC Items
        items = []
        for item_data in items_data:
            item = create_stac_item(
                variable_name=item_data['variable_name'],
                year=item_data['year'],
                month=item_data['month'],
                data_file_path=item_data['data_file_path'],
                collection_id=collection_id,
                bbox=item_data.get('bbox'),
                properties=item_data.get('properties'),
            )
            items.append(item)

        # Write STAC files
        stac_output_dir = self.output_directory / self.stac_subdir / collection_id
        write_stac_output(collection, items, str(stac_output_dir))

        return {
            'collection': collection,
            'items': items,
            'stac_output_dir': stac_output_dir,
        }

    def cleanup_raw_files(self, file_paths: List[Path]) -> None:
        """
        Delete intermediate/raw files if keep_raw_files is False.

        Args:
            file_paths (List[Path]): Paths to files to delete
        """

        if self.keep_raw_files:
            logger.info("Keeping raw files (--keep-raw flag set)")
            return

        for file_path in file_paths:
            try:
                Path(file_path).unlink()
                logger.debug(f"Deleted raw file: {file_path}")
            except FileNotFoundError:
                logger.warning(f"File not found for deletion: {file_path}")
            except Exception as e:
                logger.error(f"Error deleting {file_path}: {e}")
