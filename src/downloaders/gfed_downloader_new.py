"""
GFED Fire Data Downloader for CARDAMOM Preprocessing (REWRITTEN DEC 2025)

This module downloads Global Fire Emissions Database (GFED) v4.1s data via SFTP
and processes it for CARDAMOM carbon cycle modeling.

Scientific Context:
GFED provides spatially-explicit fire disturbance data including burned area and
CO2 emissions. This downloader implements the complete MATLAB preprocessing pipeline
including land-sea masking, NaN-aware regridding, and climatological gap-filling
for post-2016 burned area reconstruction.

Key Features:
- SFTP-based download from globalfiredata.org
- Burned area extraction (2001-2016) and reconstruction (2017-2025)
- Fire CO2 emissions extraction (all years)
- Land-sea masking with 0.5 threshold
- NaN-aware 0.25° → 0.5° regridding
- Unit conversion: gC/m²/month → gC/m²/day

References:
- GFED4.1s: Giglio et al. (2013) Global Fire Emissions and the contribution of deforestation
- Climatology method: CARDAMOM_MAPS_READ_GFED_DEC25.m

Security:
- SFTP credentials should be provided via environment variables or constructor args
- Never commit credentials to version control
"""

import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import os
import numpy as np
import xarray as xr
import h5py
import paramiko
import time

from .base import BaseDownloader

logger = logging.getLogger(__name__)

# GFED SFTP configuration (defaults - can be overridden)
GFED_SFTP_HOST = 'ftp.prd.dip.wur.nl'
GFED_SFTP_PORT = 1022
# Credentials should be provided via environment variables or constructor arguments
# NEVER commit credentials to version control!


class GFEDDownloader(BaseDownloader):
    """
    Download and process GFED v4.1s fire data for CARDAMOM.

    This downloader replicates the MATLAB preprocessing pipeline in
    CARDAMOM_MAPS_READ_GFED_DEC25.m, including:
    - SFTP download from globalfiredata.org
    - Burned area extraction (2001-2016) or reconstruction (2017-2025)
    - Fire emissions extraction (all years)
    - Land-sea masking
    - NaN-aware regridding from 0.25° to 0.5°
    - Climatological gap-filling for post-2016 burned area

    Attributes:
        sftp_host (str): SFTP server hostname
        sftp_port (int): SFTP server port
        sftp_username (str): SFTP username
        sftp_password (str): SFTP password
        land_mask_threshold (float): Land fraction threshold (default: 0.5)
    """

    def __init__(
        self,
        output_directory: str,
        keep_raw_files: bool = False,
        verbose: bool = False,
        sftp_host: str = GFED_SFTP_HOST,
        sftp_port: int = GFED_SFTP_PORT,
        sftp_username: Optional[str] = None,
        sftp_password: Optional[str] = None,
        land_mask_threshold: float = 0.5,
    ):
        """
        Initialize GFED downloader.

        SFTP credentials can be provided via:
        1. Constructor arguments (sftp_username, sftp_password)
        2. Environment variables (GFED_SFTP_USERNAME, GFED_SFTP_PASSWORD)

        Args:
            output_directory (str): Root output directory path
            keep_raw_files (bool): Retain raw HDF5 files. Default: False
            verbose (bool): Print debug messages. Default: False
            sftp_host (str): SFTP server hostname. Default: ftp.prd.dip.wur.nl
            sftp_port (int): SFTP server port. Default: 1022
            sftp_username (Optional[str]): SFTP username. If None, reads from GFED_SFTP_USERNAME env var
            sftp_password (Optional[str]): SFTP password. If None, reads from GFED_SFTP_PASSWORD env var
            land_mask_threshold (float): Land fraction threshold. Default: 0.5

        Raises:
            ValueError: If SFTP credentials are not provided
        """

        super().__init__(output_directory, keep_raw_files, verbose)

        self.sftp_host = sftp_host
        self.sftp_port = sftp_port

        # Get credentials from arguments or environment variables
        self.sftp_username = sftp_username or os.getenv('GFED_SFTP_USERNAME')
        self.sftp_password = sftp_password or os.getenv('GFED_SFTP_PASSWORD')

        if not self.sftp_username or not self.sftp_password:
            raise ValueError(
                "SFTP credentials required. Provide via constructor arguments "
                "(sftp_username, sftp_password) or environment variables "
                "(GFED_SFTP_USERNAME, GFED_SFTP_PASSWORD)"
            )

        self.land_mask_threshold = land_mask_threshold

        # Caching
        self._land_mask_cache = None
        self._climatology_cache = None

        logger.info("GFED downloader initialized (SFTP mode)")

    def _download_gfed_yearly_file_sftp(self, year: int) -> Path:
        """
        Download annual GFED4.1s HDF5 file via SFTP.

        Args:
            year (int): Year to download

        Returns:
            Path: Path to downloaded HDF5 file

        Raises:
            ConnectionError: If SFTP connection fails
            FileNotFoundError: If file not found on server
        """

        # Determine filename based on year
        if year <= 2016:
            filename = f'GFED4.1s_{year}.hdf5'
        else:
            filename = f'GFED4.1s_{year}_beta.hdf5'

        output_file = self.output_directory / self.raw_subdir / filename

        # Check if already downloaded
        if output_file.exists():
            logger.info(f"Using cached GFED file: {output_file}")
            return output_file

        logger.info(f"Downloading {filename} from SFTP server {self.sftp_host}:{self.sftp_port}")

        # SFTP download with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Create SFTP connection
                transport = paramiko.Transport((self.sftp_host, self.sftp_port))
                transport.connect(username=self.sftp_username, password=self.sftp_password)
                sftp = paramiko.SFTPClient.from_transport(transport)

                # Navigate to GFED directory
                # Files are in GFED4s directory on the server
                possible_paths = [
                    f'GFED4s/{filename}',
                    f'/GFED4s/{filename}',
                ]

                remote_file = None
                for path in possible_paths:
                    try:
                        sftp.stat(path)  # Check if file exists
                        remote_file = path
                        logger.debug(f"Found file at {path}")
                        break
                    except FileNotFoundError:
                        continue

                if remote_file is None:
                    raise FileNotFoundError(
                        f"Could not find {filename} on SFTP server. "
                        f"Tried paths: {possible_paths}"
                    )

                # Download file
                output_file.parent.mkdir(parents=True, exist_ok=True)
                sftp.get(remote_file, str(output_file))

                # Close connection
                sftp.close()
                transport.close()

                logger.info(f"Successfully downloaded: {output_file}")
                return output_file

            except Exception as e:
                logger.warning(f"SFTP download attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise ConnectionError(
                        f"Failed to download {filename} after {max_retries} attempts"
                    ) from e

    def validate_temporal_parameters(self, year: int, month: int) -> None:
        """
        Validate year and month parameters for GFED4.1s data availability.

        GFED4.1s Temporal Coverage:
        - Years: 2001-2025
        - 2025: Only through October
        - Burned area direct: 2001-2016
        - Burned area reconstructed: 2017-2025
        - Fire emissions: 2001-2025

        Args:
            year (int): Year to validate
            month (int): Month to validate (1-12)

        Raises:
            ValueError: If parameters outside valid ranges
        """

        if not isinstance(year, int) or year < 2001 or year > 2025:
            raise ValueError(
                f"Year must be between 2001 and 2025 for GFED4.1s data. Got: {year}"
            )

        # Special handling for 2025: only up to October
        if year == 2025 and month > 10:
            raise ValueError(
                f"GFED4.1s data for 2025 only available through October. "
                f"Requested: {year}-{month:02d}"
            )

        if not isinstance(month, int) or month < 1 or month > 12:
            raise ValueError(
                f"Month must be integer between 1 and 12. Got: {month}"
            )

        logger.debug(f"Validated temporal parameters: {year}-{month:02d}")

    def _extract_gfed_variables_from_hdf5(
        self,
        hdf5_file: Path,
        year: int,
        month: int,
    ) -> Dict[str, Any]:
        """
        Extract burned area and fire CO2 emissions from GFED HDF5 file.

        HDF5 Structure (year-dependent):
        - 2001-2016 files: /burned_area/{month:02d}/burned_fraction + /emissions/{month:02d}/C
        - 2017-2025 beta files: /emissions/{month:02d}/C only (NO burned_area group!)

        Args:
            hdf5_file (Path): Path to GFED HDF5 file
            year (int): Year being processed
            month (int): Month being processed (1-12)

        Returns:
            Dict[str, Any]: Dictionary with keys:
                - 'burned_area': 2D array [lat, lon] in fraction (None if not available)
                - 'fire_emissions': 2D array [lat, lon] in gC/m²/month
                - 'has_burned_area': bool indicating if BA data was directly available
                - 'latitude': 1D array of latitude coordinates
                - 'longitude': 1D array of longitude coordinates

        Raises:
            FileNotFoundError: If HDF5 file doesn't exist
            KeyError: If expected data paths not found
        """

        if not hdf5_file.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_file}")

        logger.info(f"Extracting data from {hdf5_file.name} for month {month:02d}")

        # Month as zero-padded string (e.g., '01', '02', ..., '12')
        month_str = f"{month:02d}"

        with h5py.File(hdf5_file, 'r') as hf:
            # Extract coordinate arrays (present in all files)
            # GFED provides 2D meshgrid coordinates, extract as 1D vectors
            lat_2d = np.array(hf['lat'])
            lon_2d = np.array(hf['lon'])

            # Extract 1D coordinate vectors from meshgrid
            latitude = lat_2d[:, 0]  # Latitude varies along first dimension
            longitude = lon_2d[0, :]  # Longitude varies along second dimension

            logger.debug(
                f"Loaded coordinates: lat=({latitude.shape[0]},) "
                f"[{latitude.min():.2f}, {latitude.max():.2f}], "
                f"lon=({longitude.shape[0]},) [{longitude.min():.2f}, {longitude.max():.2f}]"
            )

            # ALWAYS extract fire emissions (available in all years)
            emissions_path = f'emissions/{month_str}/C'
            try:
                fire_emissions = np.array(hf[emissions_path])
                logger.info(f"Extracted fire emissions from {emissions_path}: shape={fire_emissions.shape}")
            except KeyError:
                raise KeyError(
                    f"Fire emissions not found at path: {emissions_path}. "
                    f"Available groups in HDF5: {list(hf.keys())}"
                )

            # Extract burned area (only available for 2001-2016)
            burned_area = None
            has_burned_area = False

            if year <= 2016:
                burned_area_path = f'burned_area/{month_str}/burned_fraction'
                try:
                    burned_area = np.array(hf[burned_area_path])
                    has_burned_area = True
                    logger.info(
                        f"Extracted burned area from {burned_area_path}: shape={burned_area.shape}"
                    )
                except KeyError:
                    logger.warning(
                        f"Burned area not found at {burned_area_path} for year {year}. "
                        f"This is unexpected for pre-2017 files. Available groups: {list(hf.keys())}"
                    )
            else:
                logger.info(
                    f"Year {year} is post-2016 beta file. Burned area will be reconstructed "
                    f"from climatology (not directly available in HDF5)."
                )

        # Validate data shapes
        expected_shape = (720, 1440)  # 0.25° global resolution
        if fire_emissions.shape != expected_shape:
            logger.warning(
                f"Unexpected fire emissions shape: {fire_emissions.shape}. "
                f"Expected: {expected_shape}"
            )

        if burned_area is not None and burned_area.shape != expected_shape:
            logger.warning(
                f"Unexpected burned area shape: {burned_area.shape}. "
                f"Expected: {expected_shape}"
            )

        return {
            'burned_area': burned_area,
            'fire_emissions': fire_emissions,
            'has_burned_area': has_burned_area,
            'latitude': latitude,
            'longitude': longitude,
        }

    def _regrid_025_to_05_with_nan_handling(
        self,
        data_025: np.ndarray,
    ) -> np.ndarray:
        """
        Regrid from 0.25° to 0.5° resolution with NaN-aware averaging.

        Implements MATLAB logic from CARDAMOM_MAPS_READ_GFED_DEC25.m lines 32-45:
        - Sum values from 2x2 blocks, treating NaN as zero
        - Count non-NaN cells in each block
        - Divide sum by count to get mean (preserves NaN for all-NaN blocks)

        Args:
            data_025 (np.ndarray): Input data at 0.25° resolution (720, 1440)

        Returns:
            np.ndarray: Regridded data at 0.5° resolution (360, 720)
        """

        if data_025.shape != (720, 1440):
            raise ValueError(
                f"Expected input shape (720, 1440) for 0.25° data, got {data_025.shape}"
            )

        # Initialize output arrays (0.5° resolution = half the dimensions)
        output_shape = (360, 720)
        sum_array = np.zeros(output_shape)
        count_array = np.zeros(output_shape)

        # Sample every 2nd pixel starting from different offsets to cover all 2x2 blocks
        for row_offset in range(2):
            for col_offset in range(2):
                # Extract subset with stride 2
                subset = data_025[row_offset::2, col_offset::2]

                # Ensure subset matches output shape (should always be true for 2x2 downsampling)
                if subset.shape != output_shape:
                    # Handle edge case where dimensions don't divide evenly
                    subset = subset[:output_shape[0], :output_shape[1]]

                # Add non-NaN values to sum (NaN becomes 0)
                sum_array += np.nan_to_num(subset, nan=0.0)

                # Count non-NaN cells
                count_array += (~np.isnan(subset)).astype(float)

        # Calculate mean: sum / count
        # Where count=0 (all NaN), result will be inf, then set to NaN
        with np.errstate(divide='ignore', invalid='ignore'):
            regridded = sum_array / count_array

        # Set all-NaN blocks back to NaN
        regridded[count_array == 0] = np.nan

        logger.debug(
            f"Regridded from {data_025.shape} to {regridded.shape}. "
            f"NaN fraction: {np.isnan(regridded).sum() / regridded.size:.3f}"
        )

        return regridded

    def _convert_fire_emissions_monthly_to_daily(
        self,
        fire_emissions_monthly: np.ndarray,
    ) -> np.ndarray:
        """
        Convert fire emissions from monthly totals to daily rates.

        Implements MATLAB conversion from line 45:
        GFED.FireC = GFED.FireC * 12/365.25

        Args:
            fire_emissions_monthly (np.ndarray): Monthly emissions in gC/m²/month

        Returns:
            np.ndarray: Daily emissions in gC/m²/day
        """

        conversion_factor = 12.0 / 365.25
        fire_emissions_daily = fire_emissions_monthly * conversion_factor

        logger.debug(
            f"Converted fire emissions: monthly range [{fire_emissions_monthly.min():.3f}, "
            f"{fire_emissions_monthly.max():.3f}] → "
            f"daily range [{fire_emissions_daily.min():.3f}, {fire_emissions_daily.max():.3f}] gC/m²/day"
        )

        return fire_emissions_daily

    def download_and_process(
        self,
        year: int,
        month: int,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Download and process GFED fire data for a specific month.

        Workflow:
        1. Download annual GFED HDF5 file via SFTP
        2. Extract fire emissions (always available)
        3. Extract burned area (2001-2016) or reconstruct (2017-2025)
        4. Apply land-sea masking
        5. Regrid to CARDAMOM 0.5° resolution
        6. Convert emissions units (monthly → daily)
        7. Write NetCDF files with STAC metadata

        Args:
            year (int): Year to process
            month (int): Month to process (1-12)
            **kwargs: Additional arguments

        Returns:
            Dict[str, Any]: Results dictionary with keys:
                - 'output_files': List of generated NetCDF paths
                - 'stac_items': List of STAC Item objects
                - 'collection_ids': List of STAC Collection IDs
                - 'success': bool

        Raises:
            ValueError: If parameters are invalid
            ConnectionError: If SFTP download fails
        """

        self.validate_temporal_parameters(year, month)

        logger.info(f"Starting GFED download and processing for {year}-{month:02d}")

        # Step 1: Download yearly GFED file via SFTP
        try:
            gfed_file = self._download_gfed_yearly_file_sftp(year)
        except (ConnectionError, FileNotFoundError) as e:
            logger.error(f"Cannot proceed without GFED data: {e}")
            raise

        # Step 2-3: Extract variables from HDF5
        variables = self._extract_gfed_variables_from_hdf5(gfed_file, year, month)

        logger.info(
            f"Extracted variables: fire_emissions={variables['fire_emissions'].shape}, "
            f"burned_area={'available' if variables['has_burned_area'] else 'missing (will reconstruct)'}"
        )

        # Step 4-7: Preprocessing and output
        # TODO: Implement land-sea masking
        #   BLOCKER: Need 0.25° land-sea fraction mask
        #   MATLAB calls: loadlandseamask(0.25) but function not found
        #   Available: CARDAMOM-MAPS_05deg_LAND_SEA_FRAC.nc (0.5° only)
        #   Question: Should we upsample 0.5° → 0.25°, or is there a 0.25° mask file?

        # TODO: Implement regridding (0.25° → 0.5° with NaN-aware averaging)
        # TODO: Implement unit conversion (fire emissions: monthly → daily)
        # TODO: Implement climatological gap-filling for post-2016 BA
        #   Note: Requires batch processing 2001-2016 to compute ratios
        # TODO: Write NetCDF + STAC metadata

        # Placeholder return
        return {
            'output_files': [],
            'stac_items': [],
            'collection_ids': [],
            'success': False,
            'message': 'HDF5 extraction complete, preprocessing pipeline in progress'
        }
