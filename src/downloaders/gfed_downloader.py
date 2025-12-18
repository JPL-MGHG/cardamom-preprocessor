"""
GFED Burned Area Data Downloader for CARDAMOM Preprocessing

This module downloads Global Fire Emissions Database (GFED) data and extracts
burned area fractions for CARDAMOM carbon cycle modeling.

Scientific Context:
GFED provides spatially-explicit fire disturbance data including burned area and
emissions. Burned area represents the fraction of grid cells affected by fire in
a given month, critical for modeling fire-driven carbon losses in CARDAMOM.

References:
- GFED4.1 Database: https://daac.ornl.gov/GFED/guides/GFED4.1_Gridded_Burned_Area.html
- Giglio et al. (2013): Global Fire Emissions and the contribution of deforestation
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import numpy as np
import xarray as xr
import h5py
import requests
import tempfile

from .base import BaseDownloader

logger = logging.getLogger(__name__)

# GFED4.1 data repository
GFED_BASE_URL = 'https://daac.ornl.gov/daacdata/Global_Fire_Emissions_Database/'


class GFEDDownloader(BaseDownloader):
    """
    Download and process GFED burned area data for CARDAMOM.

    GFED data is provided as annual HDF5 files; this downloader:
    1. Downloads yearly GFED4.1 HDF5 file
    2. Extracts monthly burned_area variable
    3. Converts to NetCDF at CARDAMOM resolution
    4. Generates STAC metadata

    Attributes:
        gfed_base_url (str): Base URL for GFED data repository
        cache_yearly_data (bool): Cache downloaded HDF5 files for reuse
    """

    def __init__(
        self,
        output_directory: str,
        keep_raw_files: bool = False,
        verbose: bool = False,
        gfed_base_url: Optional[str] = None,
        cache_yearly_data: bool = True,
    ):
        """
        Initialize GFED downloader.

        Args:
            output_directory (str): Root output directory path
            keep_raw_files (bool): Retain raw HDF5 files. Default: False
            verbose (bool): Print debug messages. Default: False
            gfed_base_url (Optional[str]): Base URL for GFED repository
            cache_yearly_data (bool): Cache yearly HDF5 files. Default: True
        """

        super().__init__(output_directory, keep_raw_files, verbose)

        self.gfed_base_url = gfed_base_url or GFED_BASE_URL
        self.cache_yearly_data = cache_yearly_data
        self._yearly_cache = {}  # In-memory cache of HDF5 data

        logger.info("GFED downloader initialized")

    def _download_gfed_yearly_file(self, year: int) -> Path:
        """
        Download annual GFED4.1 HDF5 file.

        Returns:
            Path: Path to downloaded HDF5 file
        """

        # GFED file naming convention: GFED4.1s_YYYY.hdf5
        filename = f'GFED4.1s_{year}.hdf5'
        url = f'{self.gfed_base_url}GFED4.1_Gridded_Burned_Area/HDF5/{filename}'

        # Check if already in cache
        if year in self._yearly_cache:
            logger.debug(f"Using cached GFED data for {year}")
            return None  # Data is in memory

        # Create temporary file for download
        output_file = self.output_directory / self.raw_subdir / filename

        logger.info(f"Downloading GFED file from {url}")

        try:
            response = requests.get(url, timeout=300)  # 5 minute timeout
            response.raise_for_status()

            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'wb') as f:
                f.write(response.content)

            logger.info(f"Downloaded: {output_file}")

            return output_file

        except requests.RequestException as e:
            # GFED data may not be freely available on web; try local fallback
            logger.warning(f"Failed to download from {url}: {e}")
            logger.warning("Attempting local filesystem fallback")

            # Try common local paths
            local_paths = [
                Path('/data/gfed/') / filename,
                Path.home() / 'data' / 'gfed' / filename,
                Path('/tmp/gfed/') / filename,
            ]

            for local_path in local_paths:
                if local_path.exists():
                    logger.info(f"Found local GFED file: {local_path}")
                    return local_path

            raise RuntimeError(
                f"Could not obtain GFED file for {year}. "
                f"Download failed and no local copy found."
            ) from e

    def _extract_burned_area_from_hdf5(
        self,
        hdf5_file: Path,
        year: int,
        month: int,
    ) -> np.ndarray:
        """
        Extract monthly burned area fraction from GFED HDF5 file.

        GFED HDF5 structure:
        /burned_area/[month_name]/burned_area
        where month_name is like 'January', 'February', etc.

        Args:
            hdf5_file (Path): Path to GFED HDF5 file
            year (int): Year (for logging)
            month (int): Month (1-12)

        Returns:
            np.ndarray: 2D burned area fraction [latitude, longitude]

        Raises:
            RuntimeError: If HDF5 structure doesn't match expected format
        """

        month_names = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]

        month_name = month_names[month - 1]

        logger.debug(f"Extracting {month_name} {year} from {hdf5_file}")

        try:
            with h5py.File(hdf5_file, 'r') as hdf5:
                # Navigate HDF5 structure
                if 'burned_area' not in hdf5:
                    raise RuntimeError(
                        f"'burned_area' group not found in {hdf5_file}. "
                        f"Available groups: {list(hdf5.keys())}"
                    )

                ba_group = hdf5['burned_area']

                if month_name not in ba_group:
                    available_months = list(ba_group.keys())
                    raise RuntimeError(
                        f"'{month_name}' not found in burned_area. "
                        f"Available: {available_months}"
                    )

                month_group = ba_group[month_name]

                if 'burned_area' not in month_group:
                    available_vars = list(month_group.keys())
                    raise RuntimeError(
                        f"'burned_area' variable not found in {month_name}. "
                        f"Available: {available_vars}"
                    )

                burned_area_data = month_group['burned_area'][:]

                logger.debug(
                    f"Extracted {month_name}: shape={burned_area_data.shape}, "
                    f"range=[{burned_area_data.min():.6f}, {burned_area_data.max():.6f}]"
                )

                return burned_area_data

        except Exception as e:
            raise RuntimeError(
                f"Failed to extract burned_area from {hdf5_file}: {e}"
            ) from e

    def _regrid_gfed_to_cardamom(
        self,
        gfed_data: np.ndarray,
        gfed_resolution_degrees: float = 0.25,  # GFED is 0.25°
        cardamom_resolution_degrees: float = 0.5,  # CARDAMOM is 0.5°
    ) -> np.ndarray:
        """
        Regrid GFED data from native 0.25° to CARDAMOM 0.5° resolution.

        Simple averaging approach: each CARDAMOM cell is the mean of 4 GFED cells.

        Args:
            gfed_data (np.ndarray): GFED data at native resolution
            gfed_resolution_degrees (float): Native GFED resolution
            cardamom_resolution_degrees (float): Target CARDAMOM resolution

        Returns:
            np.ndarray: Regridded data at CARDAMOM resolution
        """

        if gfed_resolution_degrees == cardamom_resolution_degrees:
            return gfed_data

        factor = int(cardamom_resolution_degrees / gfed_resolution_degrees)

        if factor not in [1, 2, 4]:
            raise ValueError(
                f"Unsupported regridding factor: {factor}. "
                f"GFED resolution must be finer than CARDAMOM resolution."
            )

        logger.debug(f"Regridding GFED from {gfed_resolution_degrees}° "
                    f"to {cardamom_resolution_degrees}° (factor={factor})")

        # Simple averaging regridding
        original_shape = gfed_data.shape
        new_shape = (
            original_shape[0] // factor,
            original_shape[1] // factor,
        )

        regridded = np.zeros(new_shape, dtype=np.float32)

        for i in range(new_shape[0]):
            for j in range(new_shape[1]):
                gfed_i_start = i * factor
                gfed_j_start = j * factor
                gfed_i_end = gfed_i_start + factor
                gfed_j_end = gfed_j_start + factor

                regridded[i, j] = np.mean(
                    gfed_data[gfed_i_start:gfed_i_end, gfed_j_start:gfed_j_end]
                )

        logger.debug(f"Regridded shape: {original_shape} → {regridded.shape}")

        return regridded

    def download_and_process(
        self,
        year: int,
        month: int,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Download and process GFED burned area data for a specific month.

        Workflow:
        1. Download annual GFED HDF5 file (cached if requested)
        2. Extract monthly burned area
        3. Regrid to CARDAMOM resolution
        4. Write to NetCDF with STAC metadata

        Args:
            year (int): Year to process
            month (int): Month to process (1-12)
            **kwargs: Additional arguments (unused, for compatibility)

        Returns:
            Dict[str, Any]: Results dictionary with keys:
                - 'output_files': List of generated NetCDF paths
                - 'stac_items': List of STAC Item objects
                - 'collection_id': STAC Collection ID
                - 'success': bool

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If download fails
        """

        self.validate_temporal_parameters(year, month)

        logger.info(f"Starting GFED download for {year}-{month:02d}")

        # Step 1: Download yearly GFED file
        try:
            gfed_file = self._download_gfed_yearly_file(year)
        except RuntimeError as e:
            logger.error(f"Cannot proceed without GFED data: {e}")
            raise

        # Step 2: Extract monthly burned area
        burned_area_native = self._extract_burned_area_from_hdf5(
            gfed_file, year, month
        )

        # Step 3: Regrid to CARDAMOM resolution
        burned_area_cardamom = self._regrid_gfed_to_cardamom(burned_area_native)

        # Step 4: Create standard NetCDF dataset
        dataset = self.create_standard_netcdf_dataset(
            {'BURNED_AREA': burned_area_cardamom},
            year=year,
            month=month,
        )

        # Step 5: Write to NetCDF
        output_filename = f"burned_area_{year}_{month:02d}.nc"
        output_file = self.write_netcdf_file(
            dataset,
            output_filename,
            variable_units={'BURNED_AREA': 'fraction'},
        )

        # Step 6: Generate STAC metadata
        stac_result = self.create_and_write_stac_metadata(
            collection_id='cardamom-burned-area',
            collection_description=(
                'Monthly burned area fraction from GFED4.1s fire emissions database'
            ),
            collection_keywords=['fire', 'burned-area', 'gfed', 'disturbance'],
            items_data=[
                {
                    'variable_name': 'BURNED_AREA',
                    'year': year,
                    'month': month,
                    'data_file_path': f'data/{output_filename}',
                    'properties': {
                        'cardamom:units': 'fraction',
                        'cardamom:source': 'gfed4.1s',
                        'gfed:spatial_coverage': '0.25° native',
                        'gfed:regridding_method': 'mean aggregation to 0.5°',
                    },
                }
            ],
            temporal_start=datetime(year, month, 1),
            incremental=kwargs.get('incremental', True),
            duplicate_policy=kwargs.get('duplicate_policy', 'update'),
        )

        # Step 7: Clean up raw files if requested
        if gfed_file and not self.keep_raw_files:
            self.cleanup_raw_files([gfed_file])

        logger.info(f"Successfully created burned area NetCDF: {output_file}")

        return {
            'output_files': [output_file],
            'stac_items': stac_result['items'],
            'collection_id': 'cardamom-burned-area',
            'success': True,
        }
