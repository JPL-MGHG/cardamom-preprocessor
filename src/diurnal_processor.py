"""
Diurnal Flux Processing Module

Process CONUS carbon fluxes from monthly to diurnal (hourly) resolution.
Based on MATLAB PROJSCRIPT_DIURNAL_CMS_C_FLUXES_AUG25.m

This module handles downscaling of GPP, REC, FIR, NEE, and NBE fluxes using:
- Solar radiation patterns for GPP
- Temperature patterns for respiration (REC)
- GFED fire timing for fire emissions (FIR)

Scientific Context:
Diurnal processing creates hourly carbon flux patterns from monthly CMS data
by applying meteorological drivers and fire timing patterns. This downscaling
preserves monthly totals while adding realistic within-day variability.
"""

import os
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from config_manager import CardamomConfig
from coordinate_systems import CoordinateGrid, StandardGrids
from logging_utils import ProcessingLogger


@dataclass
class DiurnalFluxData:
    """
    Container for diurnal flux processing results.

    Attributes:
        monthly_fluxes: Dictionary with monthly flux arrays
        hourly_fluxes: Dictionary with hourly flux arrays
        auxiliary_data: Grid and coordinate information
        processing_metadata: Information about processing parameters
    """
    monthly_fluxes: Dict[str, np.ndarray]
    hourly_fluxes: Dict[str, np.ndarray]
    auxiliary_data: Dict[str, Any]
    processing_metadata: Dict[str, Any]


class DiurnalProcessor:
    """
    Process CONUS carbon fluxes from monthly to diurnal (hourly) resolution.
    Based on MATLAB PROJSCRIPT_DIURNAL_CMS_C_FLUXES_AUG25.

    Handles downscaling of GPP, REC, FIR, NEE, and NBE fluxes using:
    - Solar radiation patterns for GPP
    - Temperature patterns for respiration (REC)
    - GFED fire timing for fire emissions (FIR)

    MATLAB Reference: PROJSCRIPT_DIURNAL_CMS_C_FLUXES_AUG25.m
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize diurnal processor.

        Args:
            config_file: Optional configuration file path
        """
        self.config = CardamomConfig(config_file) if config_file else CardamomConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.processing_logger = ProcessingLogger()

        # Default processing parameters - MATLAB equivalent setup
        self.years_range = (2015, 2020)  # MATLAB: years_to_process default
        self.region_bounds = [60, -130, 20, -50]  # CONUS: N, W, S, E

        # Setup auxiliary data structure - MATLAB: AUX structure setup
        self.aux_data = self._setup_auxiliary_data()

        # Initialize coordinate system for CONUS
        self.coordinate_grid = StandardGrids.create_conus_05deg_grid()

    def _setup_auxiliary_data(self) -> Dict[str, Any]:
        """
        Setup auxiliary data paths and coordinate systems.
        Equivalent to MATLAB AUX structure setup.

        MATLAB Reference: Lines setting up AUX structure in MATLAB script

        Returns:
            dict: Auxiliary data configuration
        """
        return {
            'destination_path': {
                1: 'DUMPFILES/CARDAMOM_CONUS_DIURNAL_FLUXES_JUL25_EXP1/',
                2: 'DUMPFILES/CARDAMOM_CONUS_DIURNAL_FLUXES_JUL25/'
            },
            'lon_range': [-124.7500, -65.2500],  # MATLAB: AUX.lon_range
            'lat_range': [24.7500, 60.2500],     # MATLAB: AUX.lat_range
            'grid_resolution': 0.5               # 0.5 degree resolution
        }

    def process_diurnal_fluxes(self,
                             experiment_number: int,
                             years: Optional[List[int]] = None,
                             months: Optional[List[int]] = None,
                             output_dir: Optional[str] = None) -> DiurnalFluxData:
        """
        Main processing function for diurnal flux creation.

        MATLAB Reference: Main processing loop in PROJSCRIPT_DIURNAL_CMS_C_FLUXES_AUG25.m

        Args:
            experiment_number: 1 or 2 (different CMS experiments)
            years: List of years to process (default: 2015-2020)
            months: List of months to process (default: 1-12)
            output_dir: Optional output directory override

        Returns:
            DiurnalFluxData: Processed flux data with monthly and hourly components
        """
        if years is None:
            years = list(range(self.years_range[0], self.years_range[1] + 1))
        if months is None:
            months = list(range(1, 13))

        self.logger.info(f"Starting diurnal flux processing for experiment {experiment_number}")
        self.logger.info(f"Processing years: {years}, months: {months}")

        # Load monthly CMS fluxes - MATLAB: loading CMS data
        monthly_fluxes = self._load_cms_monthly_fluxes(experiment_number)

        # Initialize storage for hourly fluxes
        hourly_fluxes = {}

        # Process each year-month combination
        for year in years:
            for month in months:
                self.logger.info(f"Processing Month {month}, Year {year}")

                # Calculate month index for data arrays
                month_index = month - 1 + (year - 2001) * 12  # MATLAB: time indexing

                # Process single month
                monthly_result, hourly_result = self._process_single_month(
                    monthly_fluxes, year, month, month_index, experiment_number, output_dir
                )

                # Store results
                if year not in hourly_fluxes:
                    hourly_fluxes[year] = {}
                hourly_fluxes[year][month] = hourly_result

        # Create result object
        result = DiurnalFluxData(
            monthly_fluxes=monthly_fluxes,
            hourly_fluxes=hourly_fluxes,
            auxiliary_data=self.aux_data,
            processing_metadata={
                'experiment_number': experiment_number,
                'years_processed': years,
                'months_processed': months,
                'grid_info': self.coordinate_grid.get_grid_info()
            }
        )

        self.logger.info("Diurnal flux processing completed successfully")
        return result

    def _load_cms_monthly_fluxes(self, experiment_number: int) -> Dict[str, np.ndarray]:
        """
        Load monthly CMS flux data for specified experiment.

        MATLAB Reference: Loading CMS monthly flux files in MATLAB script

        Args:
            experiment_number: CMS experiment number (1 or 2)

        Returns:
            dict: Monthly flux data arrays
        """
        # Import CMS flux loader (will be implemented next)
        from .cms_flux_loader import CMSFluxLoader

        loader = CMSFluxLoader()
        monthly_fluxes = loader.load_monthly_fluxes(experiment_number)

        self.logger.info(f"Loaded monthly CMS fluxes for experiment {experiment_number}")
        return monthly_fluxes

    def _process_single_month(self,
                            fluxes: Dict[str, np.ndarray],
                            year: int,
                            month: int,
                            month_index: int,
                            experiment_number: int,
                            output_dir: Optional[str] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Process diurnal fluxes for a single month.

        MATLAB Reference: Monthly processing loop in MATLAB script

        Args:
            fluxes: Monthly flux data
            year: Processing year
            month: Processing month (1-12)
            month_index: Time index in flux arrays
            experiment_number: CMS experiment number
            output_dir: Optional output directory

        Returns:
            tuple: (monthly_processed, hourly_processed) flux dictionaries
        """
        # Step 1: Write monthly fluxes - MATLAB: write_monthlyflux_to_geoschem_format calls
        monthly_processed = self._write_monthly_fluxes(
            fluxes, year, month, month_index, experiment_number, output_dir
        )

        # Step 2: Generate diurnal patterns - MATLAB: diurnal pattern generation
        hourly_processed = self._generate_diurnal_patterns(
            fluxes, year, month, month_index, experiment_number, output_dir
        )

        return monthly_processed, hourly_processed

    def _write_monthly_fluxes(self,
                            fluxes: Dict[str, np.ndarray],
                            year: int,
                            month: int,
                            month_index: int,
                            experiment_number: int,
                            output_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Write monthly flux files with uncertainties.
        Equivalent to MATLAB write_monthlyflux_to_geoschem_format calls.

        MATLAB Reference: Calls to write_monthlyflux_to_geoschem_format in main loop

        Args:
            fluxes: Monthly flux data arrays
            year: Processing year
            month: Processing month
            month_index: Time index in arrays
            experiment_number: CMS experiment number
            output_dir: Optional output directory

        Returns:
            dict: Processed monthly flux data
        """
        # Standard CARDAMOM flux types
        flux_types = ['GPP', 'NBE', 'NEE', 'REC', 'FIR']

        monthly_output = {}

        for flux_type in flux_types:
            # Extract flux data and uncertainty
            flux_data = fluxes[flux_type][:, :, month_index] * 1e3 / 24 / 3600  # Unit conversion: gC/m²/day to Kg C/Km²/sec
            flux_unc = fluxes[f'{flux_type}unc'][:, :, month_index] * 1e3 / 24 / 3600

            # Store processed data
            monthly_output[flux_type] = flux_data
            monthly_output[f'{flux_type}_uncertainty'] = flux_unc

            # Write to NetCDF using output writers (will be implemented)
            if output_dir:
                self._write_monthly_netcdf(
                    flux_data, flux_unc, year, month, flux_type, experiment_number, output_dir
                )

        self.logger.debug(f"Processed monthly fluxes for {year}-{month:02d}")
        return monthly_output

    def _generate_diurnal_patterns(self,
                                 fluxes: Dict[str, np.ndarray],
                                 year: int,
                                 month: int,
                                 month_index: int,
                                 experiment_number: int,
                                 output_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Generate hourly diurnal patterns for the month.

        MATLAB Reference: Diurnal pattern generation section in MATLAB script

        Args:
            fluxes: Monthly flux data
            year: Processing year
            month: Processing month
            month_index: Time index in arrays
            experiment_number: CMS experiment number
            output_dir: Optional output directory

        Returns:
            dict: Hourly flux patterns
        """
        # Step 1: Load meteorological drivers - MATLAB: loading ERA5 diurnal fields
        ssrd, skt = self._load_era5_diurnal_fields(month, year)

        # Step 2: Load fire diurnal patterns - MATLAB: loading GFED diurnal fields
        co2_diurnal = self._load_gfed_diurnal_fields(month, year)

        # Step 3: Calculate diurnal fluxes - MATLAB: diurnal flux calculations
        diurnal_fluxes = self._calculate_diurnal_fluxes(
            fluxes, month_index, ssrd, skt, co2_diurnal
        )

        # Step 4: Write hourly files - MATLAB: writing hourly output files
        if output_dir:
            self._write_hourly_fluxes(diurnal_fluxes, year, month, experiment_number, output_dir)

        self.logger.debug(f"Generated diurnal patterns for {year}-{month:02d}")
        return diurnal_fluxes

    def _load_era5_diurnal_fields(self, month: int, year: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load ERA5 diurnal meteorological fields.

        MATLAB Reference: load_era5_diurnal_fields_new function call

        Args:
            month: Month to load (1-12)
            year: Year to load

        Returns:
            tuple: (SSRD, SKT) arrays with hourly data
        """
        # Import ERA5 loader (will be implemented)
        from .met_driver_loader import ERA5DiurnalLoader

        loader = ERA5DiurnalLoader()
        ssrd, skt = loader.load_diurnal_fields(month, year)

        return ssrd, skt

    def _load_gfed_diurnal_fields(self, month: int, year: int) -> np.ndarray:
        """
        Load GFED diurnal fire patterns.

        MATLAB Reference: load_gfed_diurnal_fields_05deg function call

        Args:
            month: Month to load (1-12)
            year: Year to load

        Returns:
            ndarray: CO2 diurnal emissions array
        """
        # Import GFED diurnal loader (will be implemented)
        from .gfed_diurnal_loader import GFEDDiurnalLoader

        loader = GFEDDiurnalLoader()
        co2_diurnal = loader.load_diurnal_fields(month, year)

        return co2_diurnal

    def _calculate_diurnal_fluxes(self,
                                monthly_fluxes: Dict[str, np.ndarray],
                                month_index: int,
                                ssrd: np.ndarray,
                                skt: np.ndarray,
                                co2_diurnal: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate hourly flux patterns from monthly means and drivers.

        MATLAB Reference: Diurnal flux calculation section in MATLAB script

        Args:
            monthly_fluxes: Monthly flux data
            month_index: Time index for current month
            ssrd: Solar radiation diurnal patterns
            skt: Skin temperature diurnal patterns
            co2_diurnal: Fire diurnal patterns from GFED

        Returns:
            dict: Hourly flux patterns
        """
        # Import diurnal calculator (will be implemented)
        from .diurnal_calculator import DiurnalCalculator

        calculator = DiurnalCalculator()
        diurnal_fluxes = calculator.calculate_diurnal_fluxes(
            monthly_fluxes, month_index, ssrd, skt, co2_diurnal
        )

        return diurnal_fluxes

    def _write_monthly_netcdf(self,
                            flux_data: np.ndarray,
                            uncertainty_data: np.ndarray,
                            year: int,
                            month: int,
                            flux_name: str,
                            experiment: int,
                            output_dir: str) -> None:
        """
        Write monthly flux NetCDF file.

        MATLAB Reference: write_monthlyflux_to_geoschem_format function

        Args:
            flux_data: Monthly flux array
            uncertainty_data: Uncertainty array
            year: Year for filename
            month: Month for filename
            flux_name: Type of flux (GPP, REC, etc.)
            experiment: CMS experiment number
            output_dir: Output directory path
        """
        # Import output writers (will be implemented)
        from .diurnal_output_writers import DiurnalFluxWriter

        writer = DiurnalFluxWriter(output_dir)
        writer.write_monthly_flux_to_geoschem_format(
            flux_data, uncertainty_data, year, month, self.aux_data, flux_name, experiment
        )

    def _write_hourly_fluxes(self,
                           diurnal_fluxes: Dict[str, np.ndarray],
                           year: int,
                           month: int,
                           experiment: int,
                           output_dir: str) -> None:
        """
        Write hourly flux files.

        MATLAB Reference: Writing hourly output files in MATLAB script

        Args:
            diurnal_fluxes: Hourly flux data
            year: Year for filename
            month: Month for filename
            experiment: CMS experiment number
            output_dir: Output directory path
        """
        # Import output writers (will be implemented)
        from .diurnal_output_writers import DiurnalFluxWriter

        writer = DiurnalFluxWriter(output_dir)

        # Write each flux type
        for flux_type, flux_data in diurnal_fluxes.items():
            writer.write_hourly_flux_to_geoschem_format(
                flux_data, year, month, self.aux_data, flux_type, experiment
            )