"""
Observational Data Handler for CARDAMOM CBF Generation

This module handles loading and processing observational constraint data
with graceful handling of missing values (fills with NaN instead of failing).

Scientific Context:
Observational constraints (LAI, biomass, water stress, photosynthesis) are
OPTIONAL for CBF generation. Forward-only model runs don't require obs data.
However, data assimilation runs benefit significantly from observational
constraints to reduce parameter uncertainty.

Data Strategy:
- Missing obs variables: Fill entire array with NaN
- Missing obs time steps: Fill with NaN for that period
- Missing obs regions: Fill spatial pixels with NaN
- Process continues gracefully with incomplete obs data

This contrasts with meteorological data, which MUST be complete.
"""

import logging
from typing import Optional
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def load_observational_data_with_nan_fill(
    obs_filepath: str,
    som_filepath: str,
    fir_filepath: str,
) -> Optional[xr.Dataset]:
    """
    Load observational data files with graceful NaN-filling for missing values.

    Scientific Rationale:
    Observational constraints are optional for CBF generation. If obs data
    is incomplete or missing, we fill with NaN and continue processing.
    This enables forward-only model runs that don't require observations.

    Args:
        obs_filepath (str): Path to observational constraints file
            (e.g., AlltsObs05x05newbiomass_LFmasked.nc)
        som_filepath (str): Path to soil organic matter file
        fir_filepath (str): Path to mean fire emissions file

    Returns:
        Optional[xr.Dataset]: Combined observational dataset, or None if
            all files fail to load

    Example:
        ```python
        obs_data = load_observational_data_with_nan_fill(
            obs_filepath='input/AlltsObs05x05newbiomass_LFmasked.nc',
            som_filepath='input/CARDAMOM-MAPS_05deg_HWSD_PEQ_iniSOM.nc',
            fir_filepath='input/CARDAMOM-MAPS_05deg_GFED4_Mean_FIR.nc'
        )
        ```
    """

    logger.info("Loading observational constraint data")

    # Track which files loaded successfully
    loaded_datasets = {}
    failed_files = []

    # Try to load main observational data
    try:
        logger.debug(f"Loading obs file: {obs_filepath}")
        ds_obs = xr.open_dataset(obs_filepath)
        loaded_datasets['obs'] = ds_obs
        logger.info(f"✓ Loaded observational constraints: {list(ds_obs.data_vars)}")
    except FileNotFoundError:
        logger.warning(f"✗ Observational file not found: {obs_filepath}")
        failed_files.append(obs_filepath)
    except Exception as e:
        logger.warning(f"✗ Could not load observational file: {e}")
        failed_files.append(obs_filepath)

    # Try to load SOM data
    try:
        logger.debug(f"Loading SOM file: {som_filepath}")
        ds_som = xr.open_dataset(som_filepath)
        loaded_datasets['som'] = ds_som
        logger.info(f"✓ Loaded SOM data")
    except FileNotFoundError:
        logger.warning(f"✗ SOM file not found: {som_filepath}")
        failed_files.append(som_filepath)
    except Exception as e:
        logger.warning(f"✗ Could not load SOM file: {e}")
        failed_files.append(som_filepath)

    # Try to load FIR data
    try:
        logger.debug(f"Loading FIR file: {fir_filepath}")
        ds_fir = xr.open_dataset(fir_filepath)
        loaded_datasets['fir'] = ds_fir
        logger.info(f"✓ Loaded Mean FIR data")
    except FileNotFoundError:
        logger.warning(f"✗ FIR file not found: {fir_filepath}")
        failed_files.append(fir_filepath)
    except Exception as e:
        logger.warning(f"✗ Could not load FIR file: {e}")
        failed_files.append(fir_filepath)

    # Check if anything loaded
    if not loaded_datasets:
        logger.warning(
            "Could not load any observational data files. "
            "Continuing with NaN-filled obs data (forward-only mode)."
        )
        return None

    # Assemble loaded data
    if 'obs' not in loaded_datasets:
        logger.warning("Main obs file unavailable - using SOM and FIR only")
        combined_dataset = xr.Dataset()
    else:
        combined_dataset = loaded_datasets['obs'].copy()

    # Add SOM data if available
    if 'som' in loaded_datasets:
        ds_som = loaded_datasets['som']
        try:
            # Get SOM variable (usually called 'data')
            som_var_name = list(ds_som.data_vars)[0]
            if 'SOM' not in combined_dataset:
                combined_dataset['SOM'] = ds_som[som_var_name]
                logger.debug("Added SOM variable to obs dataset")
        except Exception as e:
            logger.warning(f"Could not add SOM to obs dataset: {e}")

    # Add FIR data if available
    if 'fir' in loaded_datasets:
        ds_fir = loaded_datasets['fir']
        try:
            # Get FIR variable (usually called 'data')
            fir_var_name = list(ds_fir.data_vars)[0]
            if 'Mean_FIR' not in combined_dataset:
                combined_dataset['Mean_FIR'] = ds_fir[fir_var_name]
                logger.debug("Added Mean_FIR variable to obs dataset")
        except Exception as e:
            logger.warning(f"Could not add Mean_FIR to obs dataset: {e}")

    # Apply variable renaming to match CBF generation expectations
    # Original obs data has 'ModLAI' and 'GPPFluxSat' but CBF code expects 'LAI' and 'GPP'
    obs_rename_map = {
        'ModLAI': 'LAI',
        'GPPFluxSat': 'GPP'
    }

    variables_to_rename = {}
    for old_name, new_name in obs_rename_map.items():
        if old_name in combined_dataset.data_vars:
            variables_to_rename[old_name] = new_name
            logger.debug(f"Renaming {old_name} -> {new_name}")

    if variables_to_rename:
        combined_dataset = combined_dataset.rename(variables_to_rename)

    logger.info(f"Observational data ready: {list(combined_dataset.data_vars)}")

    return combined_dataset


def fill_missing_obs_variable(
    dataset: xr.Dataset,
    variable_name: str,
    time_length: int,
) -> np.ndarray:
    """
    Get obs variable data or return NaN-filled array if missing.

    This function handles missing obs variables gracefully by returning
    an array of NaN values instead of failing. This allows CBF generation
    to proceed even with incomplete observational data.

    Args:
        dataset (xr.Dataset): Observational dataset
        variable_name (str): Variable name to retrieve
        time_length (int): Length of time dimension for NaN array

    Returns:
        np.ndarray: Variable data (if present) or NaN array (if missing)
    """

    if dataset is None:
        logger.debug(f"Dataset is None - returning NaN for {variable_name}")
        return np.full(time_length, np.nan)

    if variable_name not in dataset:
        logger.debug(f"Variable {variable_name} missing from dataset - using NaN")
        return np.full(time_length, np.nan)

    try:
        var_data = dataset[variable_name].values
        # Check if all values are NaN
        if np.all(np.isnan(var_data)):
            logger.debug(f"Variable {variable_name} is all NaN - returning NaN array")
            return np.full(time_length, np.nan)
        return var_data
    except Exception as e:
        logger.warning(f"Could not extract {variable_name}: {e} - using NaN")
        return np.full(time_length, np.nan)


def get_pixel_obs_value_with_nan_fallback(
    dataset: xr.Dataset,
    variable_name: str,
    lat: float,
    lon: float,
    fallback_value: float = np.nan,
) -> float:
    """
    Get scalar obs value for a pixel, or return fallback value if missing.

    For single-value constraints (SOM, Mean_FIR, Mean_LAI), this retrieves
    the pixel value or returns NaN if the variable is missing.

    Args:
        dataset (xr.Dataset): Observational dataset
        variable_name (str): Variable name to retrieve
        lat (float): Latitude value
        lon (float): Longitude value
        fallback_value (float): Value to return if variable missing (default: NaN)

    Returns:
        float: Pixel value or fallback value
    """

    if dataset is None:
        logger.debug(f"Dataset is None - returning fallback for {variable_name}")
        return fallback_value

    if variable_name not in dataset:
        logger.debug(f"Variable {variable_name} missing - returning fallback value")
        return fallback_value

    try:
        # Try to find exact lat/lon match
        data = dataset[variable_name].sel(latitude=lat, longitude=lon, method='nearest')
        pixel_value = float(data.values)

        if np.isnan(pixel_value):
            logger.debug(f"{variable_name} at ({lat}, {lon}) is NaN")
            return fallback_value

        return pixel_value

    except Exception as e:
        logger.debug(f"Could not get {variable_name} for ({lat}, {lon}): {e}")
        return fallback_value
