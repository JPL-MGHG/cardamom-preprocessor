"""
CARDAMOM Binary Format (CBF) Generator - Main Entry Point

This module orchestrates CBF generation by combining:
- STAC meteorological data (via stac_met_loader)
- User-provided observational constraints and static data
- Pixel-specific CBF file generation

The main difference from erens_cbf_code.py:
- Meteorological data loaded from STAC catalog (not static file)
- STAC configuration (URL, date range) instead of file path
- Observational data with NaN-filling for missing values
- Reuses all pixel processing logic from erens_cbf_code.py

Scientific Context:
CBF files are CARDAMOM's standard input format. They contain:
- Meteorological forcing (temperature, precipitation, radiation) - REQUIRED
- Observational constraints (LAI, biomass, water stress) - OPTIONAL
- Assimilation parameters and MCMC configuration

Workflow:
1. Load meteorology from STAC catalog (FAIL if incomplete)
2. Load user-provided obs constraints and static data (fill gaps with NaN)
3. Load scaffold template for metadata/attributes
4. Find land pixels above threshold using land fraction mask
5. For each pixel: extract data, set forcing/obs, apply constraints, save CBF
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
import xarray as xr

# Import STAC meteorology loader
from stac_met_loader import load_met_data_from_stac

# Import obs data handler
from cbf_obs_handler import load_observational_data_with_nan_fill

# Import helper functions from erens_cbf_code
sys.path.insert(0, str(Path(__file__).parent.parent / "matlab-migration"))
from erens_cbf_code import (
    load_and_preprocess_land_fraction,
    find_land_pixels,
    generate_output_filepaths,
    load_scaffold_data,
    get_pixel_data,
    set_forcing_variables,
    set_observation_constraints,
    set_single_value_constraints,
    adjust_assimilation_attributes,
    set_mcmc_attributes,
    finalize_and_save,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration Constants ---

# STAC Configuration
STAC_API_URL = 'file:///path/to/stac/catalog.json'  # User must set this
START_DATE = '2001-01'  # Meteorology start date
END_DATE = '2021-12'    # Meteorology end date

# User-provided Input Files (non-STAC)
LAND_FRAC_FILE = 'input/CARDAMOM-MAPS_05deg_LAND_SEA_FRAC.nc'
OBS_DRIVER_FILE = 'input/AlltsObs05x05newbiomass_LFmasked.nc'
SOM_FILE = 'input/CARDAMOM-MAPS_05deg_HWSD_PEQ_iniSOM.nc'
FIR_FILE = 'input/CARDAMOM-MAPS_05deg_GFED4_Mean_FIR.nc'
SCAFFOLD_FILE = 'input/fluxval_US-NR1_1100_LAI.cbf.nc'

# Output Configuration
OUTPUT_DIR_TEMPLATE = 'output/CBF_{exp_id}/'
EXPERIMENT_ID = '001'

# Geographic Boundaries for CONUS+
LAT_RANGE = np.arange(229, 301)
LON_RANGE = np.arange(110, 230)
LAND_THRESHOLD = 0.5

# Variable Names (STAC conventions - no renaming needed)
MET_VARS_TO_KEEP = [
    'VPD',
    'TOTAL_PREC',
    'T2M_MIN',
    'T2M_MAX',
    'STRD',
    'SSRD',
    'SNOWFALL',
    'CO2',
    'BURNED_AREA',
    'SKT',
]

# Variables requiring positive values
POSITIVE_FORCING_VARS = [
    'STRD',
    'SSRD',
    'SNOWFALL',
    'CO2',
    'BURNED_AREA',
    'SKT',
    'TOTAL_PREC',
    'VPD',
]

# Variables for setting observational constraints
OBS_CONSTRAINT_VARS = ['SCF', 'GPP', 'ABGB', 'EWT']

# Variables needing attributes copied from scaffold
OBS_ATTR_COPY_VARS = ['ABGB', 'GPP', 'SCF']

# MCMC Settings
MCMC_ITERATIONS = 500000.0
MCMC_SAMPLES = 20.0


def generate_cbf_files(
    stac_source: str,
    start_date: str,
    end_date: str,
    output_directory: str,
    land_frac_file: str = None,
    obs_driver_file: str = None,
    som_file: str = None,
    fir_file: str = None,
    scaffold_file: str = None,
    experiment_id: str = "001",
    lat_range: np.ndarray = None,
    lon_range: np.ndarray = None,
    land_threshold: float = 0.5,
):
    """
    Generate CBF files from STAC meteorology and observational data.

    Args:
        stac_source (str): Path to local STAC catalog.json or URL to remote STAC API
        start_date (str): Start date in 'YYYY-MM' format (e.g., '2020-01')
        end_date (str): End date in 'YYYY-MM' format (e.g., '2020-12')
        output_directory (str): Output directory for CBF files
        land_frac_file (str): Path to land fraction NetCDF file (optional, uses default if None)
        obs_driver_file (str): Path to observational data NetCDF (optional, uses default if None)
        som_file (str): Path to soil organic matter NetCDF (optional, uses default if None)
        fir_file (str): Path to fire emissions NetCDF (optional, uses default if None)
        scaffold_file (str): Path to scaffold CBF template (optional, uses default if None)
        experiment_id (str): Experiment ID for output files (default: '001')
        lat_range (np.ndarray): Latitude indices to process (optional, uses default if None)
        lon_range (np.ndarray): Longitude indices to process (optional, uses default if None)
        land_threshold (float): Minimum land fraction threshold (default: 0.5)

    Workflow:
    1. Load meteorology from STAC (FAIL if incomplete)
    2. Load observational and static data (NaN-fill for gaps)
    3. Load scaffold template
    4. Find land pixels
    5. Generate CBF file for each pixel
    """
    # Use defaults if not provided
    if land_frac_file is None:
        land_frac_file = LAND_FRAC_FILE
    if obs_driver_file is None:
        obs_driver_file = OBS_DRIVER_FILE
    if som_file is None:
        som_file = SOM_FILE
    if fir_file is None:
        fir_file = FIR_FILE
    if scaffold_file is None:
        scaffold_file = SCAFFOLD_FILE
    if lat_range is None:
        lat_range = LAT_RANGE
    if lon_range is None:
        lon_range = LON_RANGE

    logger.info("=" * 80)
    logger.info("Starting CBF generation from STAC meteorology")
    logger.info("=" * 80)

    # Step 1: Load Meteorological Data from STAC
    # This will FAIL if any required variable is missing for any month
    logger.info(f"Step 1: Loading meteorology from STAC catalog")
    logger.info(f"  STAC Source: {stac_source}")
    logger.info(f"  Date range: {start_date} to {end_date}")

    try:
        met_data = load_met_data_from_stac(
            stac_source=stac_source,
            start_date=start_date,
            end_date=end_date,
        )
        logger.info(f"✓ Meteorology loaded: {list(met_data.data_vars)}")
    except Exception as e:
        logger.error(f"✗ Failed to load meteorology: {e}")
        logger.error("Meteorological data is CRITICAL - cannot continue without it")
        return

    # Step 2: Load Land Fraction Data
    logger.info(f"Step 2: Loading land fraction mask")

    land_frac_ds = load_and_preprocess_land_fraction(land_frac_file)
    if land_frac_ds is None:
        logger.error("Failed to load land fraction data - cannot identify land pixels")
        return

    logger.info("✓ Land fraction loaded")

    # Step 3: Find Land Pixels
    logger.info(f"Step 3: Finding land pixels (threshold={land_threshold})")

    latlon_coords, _ = find_land_pixels(
        land_frac_ds, lat_range, lon_range, land_threshold
    )

    if not latlon_coords:
        logger.error("No land pixels found meeting criteria")
        return

    logger.info(f"✓ Found {len(latlon_coords)} land pixels")

    # Step 4: Generate Output Paths
    logger.info(f"Step 4: Generating output file paths")

    output_dir_template = f"{output_directory}/CBF_{{exp_id}}/"
    output_filepaths = generate_output_filepaths(
        latlon_coords, output_dir_template, experiment_id
    )

    logger.info(f"✓ Output directory: {output_dir_template.format(exp_id=experiment_id)}")

    # Step 5: Load Observational and Static Data (user-provided with NaN-fill)
    logger.info(f"Step 5: Loading observational constraints and static data")

    obs_data = load_observational_data_with_nan_fill(
        obs_filepath=obs_driver_file,
        som_filepath=som_file,
        fir_filepath=fir_file,
    )

    if obs_data is None:
        logger.warning(
            "No observational data available. "
            "Continuing in forward-only mode (no observational constraints)."
        )
    else:
        logger.info(f"✓ Observational data loaded with {len(obs_data.data_vars)} variables")

    # Step 6: Load Scaffold Template
    logger.info(f"Step 6: Loading scaffold template")

    scaffold_ds = load_scaffold_data(scaffold_file)
    if scaffold_ds is None:
        logger.error("Failed to load scaffold template - cannot generate CBF files")
        return

    logger.info(f"✓ Scaffold loaded")

    # Step 7: Process Each Pixel
    logger.info(f"Step 7: Generating CBF files for {len(latlon_coords)} pixels")

    successful_pixels = 0
    failed_pixels = 0

    for i, (real_lat, real_lon) in enumerate(latlon_coords):
        output_file = output_filepaths[i]

        try:
            logger.debug(f"Processing pixel {i + 1}/{len(latlon_coords)}: "
                        f"Lat={real_lat:.2f}, Lon={real_lon:.2f}")

            # Create pixel-specific CBF dataset
            # a. Copy scaffold and drop unwanted variables
            vars_to_drop = [
                v
                for v in scaffold_ds.data_vars
                if any(x in v for x in ['val', 'ET', 'NBE', 'LAI', 'Ceff'])
            ]
            pixel_ds = scaffold_ds.copy().drop_vars(vars_to_drop)

            # b. Set basic metadata
            pixel_ds['LAT'] = real_lat
            pixel_ds['LON'] = real_lon

            # c. Set forcing variables from meteorology
            set_forcing_variables(
                pixel_ds, met_data, real_lat, real_lon, scaffold_ds, POSITIVE_FORCING_VARS
            )

            # d. Set observation constraints (graceful degradation if missing)
            if obs_data is not None:
                try:
                    set_observation_constraints(
                        pixel_ds,
                        obs_data,
                        real_lat,
                        real_lon,
                        scaffold_ds,
                        OBS_CONSTRAINT_VARS,
                        OBS_ATTR_COPY_VARS,
                    )
                except Exception as e:
                    logger.debug(
                        f"Could not set obs constraints for ({real_lat}, {real_lon}): {e}"
                    )
            else:
                logger.debug(f"Skipping obs constraints for ({real_lat}, {real_lon}) - no obs data")

            # e. Set single-value constraints (graceful degradation if missing)
            if obs_data is not None:
                try:
                    set_single_value_constraints(
                        pixel_ds, obs_data, real_lat, real_lon, scaffold_ds
                    )
                except Exception as e:
                    logger.debug(
                        f"Could not set single-value constraints for ({real_lat}, {real_lon}): {e}"
                    )

            # f. Adjust assimilation attributes
            adjust_assimilation_attributes(pixel_ds)

            # g. Set MCMC attributes
            set_mcmc_attributes(pixel_ds, MCMC_ITERATIONS, MCMC_SAMPLES)

            # h. Finalize and save
            finalize_and_save(pixel_ds, str(output_file))

            successful_pixels += 1

            # Progress logging
            if (i + 1) % 100 == 0:
                logger.info(
                    f"Progress: {i + 1}/{len(latlon_coords)} pixels "
                    f"({successful_pixels} successful, {failed_pixels} failed)"
                )

            pixel_ds.close()

        except Exception as e:
            logger.warning(f"Failed to generate CBF for ({real_lat:.2f}, {real_lon:.2f}): {e}")
            failed_pixels += 1
            continue

    # Step 8: Summary
    logger.info("=" * 80)
    logger.info(f"CBF Generation Complete")
    logger.info(f"  Successful: {successful_pixels}")
    logger.info(f"  Failed: {failed_pixels}")
    logger.info(f"  Output: {output_dir_template.format(exp_id=experiment_id)}")
    logger.info("=" * 80)

    # Close datasets
    land_frac_ds.close()
    met_data.close()
    if obs_data is not None:
        obs_data.close()
    scaffold_ds.close()

    return {
        'successful_pixels': successful_pixels,
        'failed_pixels': failed_pixels,
        'output_directory': output_dir_template.format(exp_id=experiment_id),
    }


def main():
    """
    Backward-compatible main function using hardcoded configuration constants.

    For programmatic use, prefer calling generate_cbf_files() directly with parameters.
    """
    return generate_cbf_files(
        stac_source=STAC_API_URL,
        start_date=START_DATE,
        end_date=END_DATE,
        output_directory=OUTPUT_DIR_TEMPLATE.replace('CBF_{exp_id}/', '').rstrip('/'),
        experiment_id=EXPERIMENT_ID,
    )


if __name__ == "__main__":
    main()
