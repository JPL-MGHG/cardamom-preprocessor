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
import logging
import numpy as np
import pandas as pd
import xarray as xr

# Import STAC meteorology loader
from stac_met_loader import load_met_data_from_stac

# Import obs data handler
from cbf_obs_handler import load_observational_data_with_nan_fill

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
# CRITICAL: LAI must be included as time series (not just scalar Mean_LAI)
OBS_CONSTRAINT_VARS = ['LAI', 'SCF', 'GPP', 'ABGB', 'EWT']

# Variables needing attributes copied from scaffold
OBS_ATTR_COPY_VARS = ['LAI', 'ABGB', 'GPP', 'SCF']

# MCMC Settings
MCMC_ITERATIONS = 500000.0
MCMC_SAMPLES = 20.0


# --- Helper Functions (formerly from matlab-migration/erens_cbf_code.py) ---


def load_and_preprocess_land_fraction(filepath):
    """Loads and transposes the land fraction dataset."""
    try:
        ds = xr.load_dataset(filepath).transpose()
        logger.info(f"Successfully loaded land fraction data from: {filepath}")
        return ds
    except FileNotFoundError:
        logger.error(f"Error: Land fraction file not found at {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error loading land fraction data: {e}")
        return None


def find_land_pixels(land_frac_ds, lat_range, lon_range, threshold):
    """Identifies land pixels above a threshold within lat/lon ranges."""
    latlon_coords = []
    latlon_indices = []
    if land_frac_ds is None:
        return latlon_coords, latlon_indices

    # Ensure dataset is transposed correctly if needed (assuming lat, lon dimensions)
    # The original code transposes, so we assume it's needed. Check dimension names if issues arise.
    land_frac_data = land_frac_ds['data']  # Adjust 'data' if variable name differs

    for i in lat_range:
        for j in lon_range:
            # Use .sel for label-based indexing if coordinates are monotonic and regularly spaced
            # Using .isel requires knowing the integer indices corresponding to the range
            # The original code uses integer indices directly, so we replicate that.
            # Ensure the indices i, j are valid for the dataset dimensions.
            try:
                pixel_value = land_frac_data.isel(latitude=i, longitude=j).item()  # Use .item() to get scalar value
                if pixel_value > threshold:
                    lat = land_frac_ds.latitude[i].item()
                    lon = land_frac_ds.longitude[j].item()
                    latlon_coords.append([lat, lon])
                    latlon_indices.append(
                        [i, j])  # Store original indices if needed later, though not used in current script
            except IndexError:
                # Handle cases where i or j might be out of bounds for the loaded data
                continue
            except Exception as e:
                logger.warning(f"Warning: Error accessing land fraction data at index ({i}, {j}): {e}")
                continue

    logger.info(f"Found {len(latlon_coords)} land pixels meeting criteria.")
    return latlon_coords, latlon_indices


def generate_output_filepaths(latlon_coords, output_dir_template, exp_id):
    """Generates output filenames and paths based on lat/lon coordinates."""
    filepaths = []
    output_dir = output_dir_template.format(exp_id=exp_id)
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    for lat, lon in latlon_coords:
        # Format latitude and longitude strings carefully as per original logic
        # Example: lat=35.25, lon=-105.75
        lat_str_deg = str(lat).split('.')[0]  # '35'
        lat_str_dec = str(lat).split('.')[1][:2].ljust(2, '0')  # '25'
        # Longitude needs careful handling for negative sign and formatting
        lon_abs = abs(lon)  # 105.75
        lon_str_deg = str(lon_abs).split('.')[0]  # '105'
        lon_str_dec = str(lon_abs).split('.')[1][:2].ljust(2, '0')  # '75'

        # Ensure degree parts have leading zeros if needed (original didn't seem to, but good practice)
        # lat_str_deg = lat_str_deg.zfill(2) # e.g., '05' if lat was 5.25
        # lon_str_deg = lon_str_deg.zfill(3) # e.g., '095' if lon was -95.75

        # Original format: site{lat_deg}_{lat_dec}N{lon_deg}_{lon_dec}W_ID{exp_id}exp0.cbf.nc
        # Example based on original: site35_25N105_75W_ID1100exp0.cbf.nc
        # Note: Original code used slicing like str(latlon[i][1])[1:-3] which seems complex and potentially fragile.
        # Let's try to replicate the *intent* based on the example filename structure.
        # Assuming West longitude (negative) and North latitude (positive).
        filename = f"site{lat_str_deg}_{lat_str_dec}N{lon_str_deg}_{lon_str_dec}W_ID{exp_id}exp0.cbf.nc"
        filepath = os.path.join(output_dir, filename)
        filepaths.append(filepath)

    return filepaths


def load_scaffold_data(filepath):
    """Loads the scaffold dataset."""
    try:
        ds = xr.load_dataset(filepath)
        logger.info(f"Successfully loaded scaffold data from: {filepath}")
        return ds
    except FileNotFoundError:
        logger.error(f"Error: Scaffold file not found at {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error loading scaffold data: {e}")
        return None


def get_pixel_data(dataset, lat, lon, var_name):
    """Extracts data for a specific variable at a given lat/lon."""
    # Using .sel method assumes coordinates are indexed.
    # If coordinates are not precisely aligned, 'nearest' method might be needed.
    # The original code used np.where, implying exact matches were expected or handled.
    # Replicating np.where logic for robustness if .sel fails:
    try:
        lat_idx = np.where(dataset.latitude == lat)[0][0]
        lon_idx = np.where(dataset.longitude == lon)[0][0]
        return dataset[var_name].isel(latitude=lat_idx, longitude=lon_idx).squeeze()
    except IndexError:
        # Fallback to .sel with nearest neighbor if exact match fails or coords differ slightly
        try:
            logger.warning(f"Warning: Exact coordinate match failed for {var_name} at ({lat}, {lon}). Trying nearest neighbor.")
            return dataset[var_name].sel(latitude=lat, longitude=lon, method="nearest").squeeze()
        except Exception as e:
            logger.error(f"Error extracting data for {var_name} at ({lat}, {lon}): {e}")
            return None
    except Exception as e:
        logger.error(f"Error extracting data for {var_name} at ({lat}, {lon}): {e}")
        return None


def set_forcing_variables(target_ds, source_met_ds, lat, lon, scaffold_ds, positive_vars):
    """Sets forcing variables in the target dataset using source met data."""
    # Identify forcing variables present in the scaffold (excluding time, lat, lon etc.)
    forcing_vars_in_scaffold = [v for v in MET_VARS_TO_KEEP if v in scaffold_ds.data_vars]

    # Use the correct time coordinate (target_ds now already has meteorological time)
    time_coord = target_ds.time

    for var_name in forcing_vars_in_scaffold:
        pixel_data = get_pixel_data(source_met_ds, lat, lon, var_name)
        if pixel_data is not None:
            # Create DataArray with meteorological data's time coordinate
            target_ds[var_name] = xr.DataArray(
                data=pixel_data.data,
                coords={'time': time_coord},
                dims=['time'],
                attrs=scaffold_ds[var_name].attrs if var_name in scaffold_ds else {}  # Copy attributes
            )
            # Ensure positivity constraints
            if var_name in positive_vars:
                target_ds[var_name] = target_ds[var_name].where(target_ds[var_name] > 0, other=0)
        else:
            logger.warning(f"Warning: Could not retrieve data for forcing variable {var_name}. Skipping.")

    # Set missing but required forcing variables to zero
    num_time_steps = len(time_coord)
    if 'DISTURBANCE_FLUX' not in target_ds:
        target_ds['DISTURBANCE_FLUX'] = xr.DataArray(data=np.zeros(num_time_steps), coords={'time': time_coord},
                                                     dims=['time'])
    if 'YIELD' not in target_ds:
        target_ds['YIELD'] = xr.DataArray(data=np.zeros(num_time_steps), coords={'time': time_coord}, dims=['time'])


def parse_time_coordinates(dataset):
    """
    Parse time coordinates from dataset and convert to datetime objects.

    Scientific Context:
    Time coordinates in meteorological and observational datasets may use various formats
    (e.g., minutes since reference date, days since reference date, datetime64, etc.).
    This function normalizes them to pandas DatetimeIndex for comparison.

    Args:
        dataset: xarray Dataset with time coordinate

    Returns:
        pandas DatetimeIndex: Datetime objects representing each time step
        None: If dataset has no time coordinate or parsing fails
    """
    if 'time' not in dataset.coords:
        return None

    try:
        # Use pandas to handle various time formats (CF conventions, etc.)
        times = pd.to_datetime(dataset.time.values)
        return times
    except Exception as e:
        logger.warning(f"Could not parse time coordinates: {e}")
        return None


def match_time_coordinates(obs_times, met_times):
    """
    Match observation time coordinates to meteorology time coordinates by year-month.

    Scientific Context:
    This function enables intelligent temporal alignment between observational data
    (which may be from different time periods) and meteorological forcing data.
    Matching is done at the month level to handle datasets with different day-of-month values.

    Algorithm:
    1. For each meteorological month (year, month), search for matching obs month
    2. Return a mapping of met_index -> obs_index (or None if no match found)
    3. This allows:
       - Full use when periods overlap exactly (e.g., both have 2020 data)
       - Partial use when periods overlap partially (e.g., met=2020-2024, obs=2001-2021)
       - NaN-filling when periods don't overlap (e.g., met=2024, obs=2001-2021)

    Args:
        obs_times: pandas DatetimeIndex from observational data
        met_times: pandas DatetimeIndex from meteorological data

    Returns:
        dict: Mapping {met_index: obs_index} where obs_index is None if no temporal match
    """
    time_mapping = {}

    for met_idx, met_time in enumerate(met_times):
        # Extract year-month tuple for matching
        met_year_month = (met_time.year, met_time.month)

        match_found = False
        for obs_idx, obs_time in enumerate(obs_times):
            obs_year_month = (obs_time.year, obs_time.month)

            if met_year_month == obs_year_month:
                time_mapping[met_idx] = obs_idx
                match_found = True
                logger.debug(f"Matched meteorological {met_time.strftime('%Y-%m')} to observational {obs_time.strftime('%Y-%m')}")
                break

        if not match_found:
            time_mapping[met_idx] = None
            logger.debug(f"No observational data found for meteorological {met_time.strftime('%Y-%m')}")

    return time_mapping


def set_observation_constraints(target_ds, source_obs_ds, lat, lon, scaffold_ds, constraint_vars, attr_copy_vars):
    """
    Sets observational constraints in the target dataset with intelligent temporal matching.

    Scientific Context:
    Observational constraints are optional and may come from different time periods than
    the meteorological forcing data. This function intelligently matches them by (year, month)
    to ensure we only use temporally-appropriate observational constraints:
    - Uses obs data where time periods overlap
    - Fills with NaN where no temporal match exists
    - Always maintains the meteorological time dimension

    Example behaviors:
    - Met: 2024 (12 months), Obs: 2001-2021 → No overlap → All NaN
    - Met: 2020 (12 months), Obs: 2001-2021 → Full overlap → All matched
    - Met: 2020-2024 (60 months), Obs: 2001-2021 → Partial → 2020-2021 matched, 2022-2024 NaN
    """
    # Use the updated time coordinate from target_ds (which now matches meteorological data)
    time_coord = target_ds.time
    met_time_steps = len(time_coord)

    # Parse time coordinates from both datasets for intelligent matching
    met_times = parse_time_coordinates(target_ds)
    obs_times = parse_time_coordinates(source_obs_ds) if source_obs_ds is not None else None

    for var_name in constraint_vars:
        if var_name in source_obs_ds:
            pixel_data = get_pixel_data(source_obs_ds, lat, lon, var_name)
            if pixel_data is not None:
                obs_data_values = pixel_data.data if hasattr(pixel_data, 'data') else pixel_data.values

                # Check if obs data has time dimension
                if len(obs_data_values.shape) == 0:
                    # Scalar data - broadcast to time dimension
                    final_data = np.full(met_time_steps, obs_data_values)
                    logger.debug(f"Broadcasting scalar {var_name} to {met_time_steps} time steps")

                elif obs_times is None or met_times is None:
                    # Cannot parse time coordinates - fall back to shape-based matching
                    logger.warning(f"Cannot parse time coordinates for {var_name}. Using shape-based matching.")

                    if len(obs_data_values) == met_time_steps:
                        # Perfect match - use as is
                        final_data = obs_data_values
                    elif len(obs_data_values) < met_time_steps:
                        # Less data than needed - fill with NaNs
                        logger.warning(f"Obs data {var_name} has {len(obs_data_values)} time steps, need {met_time_steps}. Filling missing with NaN.")
                        final_data = np.full(met_time_steps, np.nan)
                        final_data[:len(obs_data_values)] = obs_data_values
                    else:
                        # More data than needed - fill with NaN (don't use mismatched data)
                        logger.warning(f"Obs data {var_name} has {len(obs_data_values)} time steps, need {met_time_steps}. "
                                     f"Cannot determine temporal alignment - filling with NaN.")
                        final_data = np.full(met_time_steps, np.nan)

                else:
                    # Intelligent temporal matching based on year-month
                    time_mapping = match_time_coordinates(obs_times, met_times)
                    final_data = np.full(met_time_steps, np.nan)

                    matched_count = 0
                    for met_idx, obs_idx in time_mapping.items():
                        if obs_idx is not None:
                            final_data[met_idx] = obs_data_values[obs_idx]
                            matched_count += 1

                    if matched_count > 0:
                        logger.info(f"Matched {matched_count}/{met_time_steps} time steps for {var_name}")
                    else:
                        logger.warning(f"No temporal overlap found for {var_name} between obs data "
                                     f"({obs_times[0].strftime('%Y-%m')} to {obs_times[-1].strftime('%Y-%m')}) and met data "
                                     f"({met_times[0].strftime('%Y-%m')} to {met_times[-1].strftime('%Y-%m')}). Using NaN.")

                target_ds[var_name] = xr.DataArray(
                    data=final_data,
                    coords={'time': time_coord},
                    dims=['time']
                )
                # Copy attributes if specified
                if var_name in attr_copy_vars and var_name in scaffold_ds:
                    target_ds[var_name].attrs = scaffold_ds[var_name].attrs
            else:
                logger.warning(f"Could not retrieve data for observation constraint {var_name}. Skipping.")
        else:
            logger.warning(f"Observation constraint variable {var_name} not found in source data.")


def set_single_value_constraints(target_ds, source_obs_ds, lat, lon, scaffold_ds):
    """
    Sets single-value constraints like initial SOM, CUE, Mean LAI, Mean FIR.

    Scientific Context:
    Prior Equivalent Constraints (PEQ) provide Bayesian priors for model parameters.
    These are essential for constraining the 89 DALEC parameters when observations alone
    are insufficient. Each PEQ parameter represents prior knowledge about ecosystem
    properties at this pixel.
    """
    # PEQ_iniSOM (Initial Soil Organic Matter)
    som_data = get_pixel_data(source_obs_ds, lat, lon, 'SOM')
    if som_data is not None:
        target_ds['PEQ_iniSOM'] = xr.DataArray(
            data=som_data.item(),  # Should be scalar
            coords={'latitude': lat, 'longitude': lon},  # Use actual lat/lon
            dims=[],  # Scalar has no dims
            attrs=scaffold_ds['PEQ_iniSOM'].attrs if 'PEQ_iniSOM' in scaffold_ds else {}
        )
        # Update specific attributes
        target_ds['PEQ_iniSOM'].attrs['units'] = 'gC/m2'
        target_ds['PEQ_iniSOM'].attrs['source'] = 'HWSD'
        target_ds['PEQ_iniSOM'].attrs['unc'] = 1.5
        target_ds['PEQ_iniSOM'].attrs['opt_unc_type'] = 1.0
    else:
        logger.warning("Warning: Could not retrieve SOM data.")

    # PEQ_CUE (Carbon Use Efficiency) - Fixed value
    target_ds['PEQ_CUE'] = xr.DataArray(
        data=0.5,
        coords={'latitude': lat, 'longitude': lon},
        dims=[]
    )
    target_ds['PEQ_CUE'].attrs = {
        'opt_unc_type': 0.0,
        'unc': 0.25,
        'description': 'Carbon use efficiency constraint'
    }

    # PEQ_Cefficiency (Carbon Efficiency) - REQUIRED but MISSING
    # TODO: Source from ecological database or model prior
    logger.warning("Warning: PEQ_Cefficiency not implemented - requires ecological prior data")

    # PEQ_NBEmrg (Net Biome Exchange merge parameter) - REQUIRED but MISSING
    # TODO: Source from literature or previous MCMC runs
    logger.warning("Warning: PEQ_NBEmrg not implemented - requires calibration data")

    # PEQ_iniSnow (Initial Snow) - REQUIRED but MISSING
    # TODO: Source from satellite or model data
    logger.warning("Warning: PEQ_iniSnow not implemented - requires snow initialization data")

    # PEQ_LCMA (Leaf Carbon to Mass Ratio) - REQUIRED but MISSING
    # TODO: Source from allometric equations or database
    logger.warning("Warning: PEQ_LCMA not implemented - requires leaf trait data")

    # PEQ_clumping (Clumping Index for LAI) - REQUIRED but MISSING
    # TODO: Source from LAI/optical properties database
    logger.warning("Warning: PEQ_clumping not implemented - requires LAI clumping data")

    # Mean_LAI (Annual average LAI)
    lai_data = get_pixel_data(source_obs_ds, lat, lon, 'LAI')
    if lai_data is not None:
        # Check if LAI has time dimension
        if hasattr(lai_data, 'dims') and len(lai_data.dims) > 0:
            # Time series LAI - calculate mean
            mean_lai_value = lai_data.mean(dim='time', skipna=True).item()
        else:
            # Scalar LAI
            mean_lai_value = lai_data.item() if hasattr(lai_data, 'item') else lai_data

        target_ds['Mean_LAI'] = xr.DataArray(
            data=mean_lai_value,
            coords={'latitude': lat, 'longitude': lon},
            dims=[]
        )
        # Attributes set later in adjust_assimilation_attributes
        logger.debug(f"Set Mean_LAI = {mean_lai_value:.2f} m2/m2")
    else:
        logger.warning("Warning: Could not retrieve LAI data.")

    # Mean_GPP (Annual average GPP) - OPTIONAL but RECOMMENDED
    gpp_data = get_pixel_data(source_obs_ds, lat, lon, 'GPP')
    if gpp_data is not None and hasattr(gpp_data, 'dims') and len(gpp_data.dims) > 0:
        mean_gpp_value = gpp_data.mean(dim='time', skipna=True).item()
        target_ds['Mean_GPP'] = xr.DataArray(
            data=mean_gpp_value,
            coords={'latitude': lat, 'longitude': lon},
            dims=[]
        )
        logger.debug(f"Set Mean_GPP = {mean_gpp_value:.2f} gC/m2/day")
    else:
        logger.debug("Mean_GPP not available in source data")

    # Mean_ABGB (Annual average Above-Ground Biomass) - OPTIONAL but RECOMMENDED
    abgb_data = get_pixel_data(source_obs_ds, lat, lon, 'ABGB')
    if abgb_data is not None and hasattr(abgb_data, 'dims') and len(abgb_data.dims) > 0:
        mean_abgb_value = abgb_data.mean(dim='time', skipna=True).item()
        target_ds['Mean_ABGB'] = xr.DataArray(
            data=mean_abgb_value,
            coords={'latitude': lat, 'longitude': lon},
            dims=[]
        )
        logger.debug(f"Set Mean_ABGB = {mean_abgb_value:.2f} gC/m2")
    else:
        logger.debug("Mean_ABGB not available in source data")

    # Mean_FIR (Annual average Fire Emissions)
    fir_data = get_pixel_data(source_obs_ds, lat, lon, 'Mean_FIR')
    if fir_data is not None:
        target_ds['Mean_FIR'] = xr.DataArray(
            data=fir_data.item(),  # Should be scalar
            coords={'latitude': lat, 'longitude': lon},
            dims=[]
        )
        # Attributes set later in adjust_assimilation_attributes
    else:
        logger.warning("Warning: Could not retrieve Mean_FIR data.")


def adjust_assimilation_attributes(target_ds):
    """Adjusts attributes related to data assimilation methods."""
    # Fire flux assimilation (Mean_FIR)
    if 'Mean_FIR' in target_ds:
        target_ds.Mean_FIR.attrs.update({
            'unc': 0.1,
            'opt_unc_type': 2.0,
            'min_threshold': 0.01,
            # 'units': 'gC/m2/day', # Original had this, but FIR is often unitless ratio? Verify.
            'source': 'GFED4 mean CO2 fire emissions'  # Original had this on Mean_LAI attrs? Correcting placement.
        })

    # LAI time series assimilation (Critical for photosynthesis constraints)
    if 'LAI' in target_ds:
        target_ds.LAI.attrs.update({
            'unc': 1.2,
            'opt_unc_type': 1.0,
            'min_threshold': 0.2,
            'units': 'm2/m2',
            'source': 'MODIS LAI',
            'long_name': 'Leaf Area Index time series'
        })

    # LAI assimilation (Mean_LAI - annual average)
    if 'Mean_LAI' in target_ds:
        target_ds.Mean_LAI.attrs.update({
            'unc': 1.2,
            'opt_unc_type': 1.0,
            'min_threshold': 0.2,
            'units': 'm2/m2',
            'source': 'MODIS LAI'
        })

    # Biomass assimilation (ABGB time series)
    # CRITICAL: Variable must be named exactly "ABGB" (not "ABGB_val")
    # CARDAMOM's READ_NETCDF_TIMESERIES_OBS_FIELDS() looks for "ABGB"
    # If renamed to ABGB_val, CARDAMOM reports "obs length: 0" and can't use observations
    if 'ABGB' in target_ds:
        target_ds.ABGB.attrs.update({
            'opt_filter': 3.0,
            'opt_unc_type': 1.0,
            'single_annual_unc': 1.1,
            'min_threshold': 10.0,
            'units': 'gC/m2',
            'source': 'Xu et al 2021'
        })

    # Photosynthesis assimilation (GPP)
    if 'GPP' in target_ds:
        target_ds.GPP.attrs.update({
            'unc': 1.30,
            'opt_unc_type': 1.0,
            'min_threshold': 1.0,
            'units': 'gC/m2/day',
            'source': 'FluxSat, Joiner et al 2019'
        })

    # Water storage assimilation (EWT)
    if 'EWT' in target_ds:
        target_ds.EWT.attrs.update({
            'opt_normalization': 1.0,
            'opt_filter': 0.0,
            'unc': 200.0,
            'units': 'mm',
            'source': 'GRACE'
        })


def set_mcmc_attributes(target_ds, iterations, samples):
    """Sets MCMC specific attributes."""
    if 'MCMCID' in target_ds:
        target_ds['MCMCID'].attrs['nITERATIONS'] = iterations
        target_ds['MCMCID'].attrs['nSAMPLES'] = samples
    else:
        logger.warning("Warning: MCMCID variable not found in target dataset. Cannot set MCMC attributes.")


def compute_day_of_year(dataset):
    """
    Compute Day of Year (DOY) variable from time coordinate.

    Scientific Context:
    Day of Year (DOY) represents the fractional day within the annual cycle,
    ranging from ~0 (January 1) to ~365 (December 31). This is used by
    CARDAMOM for phenological calculations and seasonal cycle tracking.

    The DOY is computed as a 0-indexed day (0-365) plus a fractional component
    based on the time of day (hours, minutes, seconds). January 1st at 00:00
    corresponds to DOY=0.0, while January 1st at 12:00 corresponds to DOY=0.5.

    Args:
        dataset: xarray Dataset with time coordinate

    Returns:
        numpy array: Fractional day of year for each time step
    """
    if 'time' not in dataset.coords:
        logger.warning("No time coordinate found - cannot compute DOY")
        return None

    try:
        # Convert time coordinate to datetime objects
        times = pd.to_datetime(dataset.time.values)

        # Compute day of year for each time step
        doy_values = np.zeros(len(times), dtype=np.float32)

        for i, time_val in enumerate(times):
            # Get integer day of year using 0-based indexing (0-365)
            # pandas dayofyear is 1-based, so subtract 1
            day_of_year_integer = time_val.dayofyear - 1

            # Get fractional component: (hour + minute/60 + second/3600) / 24
            fractional_day = (time_val.hour + time_val.minute / 60.0 + time_val.second / 3600.0) / 24.0

            # Combine: DOY = integer_day + fractional_part (0-indexed)
            doy_values[i] = day_of_year_integer + fractional_day

        logger.debug(f"Computed DOY for {len(times)} time steps")
        return doy_values

    except Exception as e:
        logger.error(f"Error computing day of year: {e}")
        return None


def add_doy_variable(target_ds):
    """
    Add Day of Year (DOY) variable to the target dataset.

    Scientific Context:
    DOY is essential for phenological calculations in carbon cycle models.
    It provides the seasonal cycle information that constrains photosynthesis,
    respiration, and allocation processes in CARDAMOM.

    Args:
        target_ds: xarray Dataset to which DOY will be added
    """
    if 'time' not in target_ds.coords:
        logger.warning("Cannot add DOY - no time coordinate in dataset")
        return

    doy_values = compute_day_of_year(target_ds)

    if doy_values is not None:
        time_coord = target_ds.time

        target_ds['DOY'] = xr.DataArray(
            data=doy_values,
            coords={'time': time_coord},
            dims=['time'],
            attrs={
                'units': 'day',
                'long_name': 'Day of year (fractional)',
                'description': 'Integer day of year (1-366) plus fractional day based on time of day'
            }
        )
        logger.debug(f"Added DOY variable to dataset (min={doy_values.min():.2f}, max={doy_values.max():.2f})")
    else:
        logger.warning("Could not compute DOY - variable will not be added")


def finalize_and_save(dataset, output_filepath):
    """Finalizes dataset (encoding) and saves to NetCDF.

    Scientific Context:
    Variable names must exactly match CARDAMOM's expected NetCDF schema.
    CARDAMOM reads specific variable names via READ_NETCDF_TIMESERIES_OBS_FIELDS(),
    so renaming breaks the data pipeline. The ABGB variable must remain "ABGB",
    not "ABGB_val" - the "_val" suffix causes CARDAMOM to report "obs length: 0".
    """
    final_ds = dataset.copy()

    # NOTE: Do NOT rename ABGB to ABGB_val
    # CARDAMOM expects exactly "ABGB" as variable name
    # If renamed to ABGB_val, CARDAMOM will not find the observation data
    # and will report "Preprocess ABGB: obs length: 0"
    if 'ABGB_val' in final_ds:
        logger.warning("Found 'ABGB_val' - renaming to 'ABGB' for CARDAMOM compatibility")
        final_ds = final_ds.rename({'ABGB_val': 'ABGB'})

    # Set encoding for CARDAMOM compatibility
    if 'time' in final_ds.coords:
        final_ds['time'].encoding['units'] = 'days since 2001-01-01'

    encoding_settings = {'_FillValue': -9999.0, 'dtype': 'float32'}
    var_encoding = {var: encoding_settings for var in final_ds.data_vars}

    try:
        final_ds.to_netcdf(output_filepath, encoding=var_encoding)
        logger.info(f"Successfully saved CBF file to: {output_filepath}")
    except Exception as e:
        logger.error(f"Error saving NetCDF file {output_filepath}: {e}")


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
            landmask_file_path=land_frac_file,
            land_fraction_threshold=LAND_THRESHOLD,
        )
        logger.info(f"✓ Meteorology loaded: {list(met_data.data_vars)}")

        # Write unified meteorology dataset to output directory for inspection
        os.makedirs(output_directory, exist_ok=True)
        unified_met_file = os.path.join(output_directory, "unified_meteorology.nc")
        try:
            met_data.to_netcdf(unified_met_file)
            logger.info(f"✓ Unified meteorology written to: {unified_met_file}")
        except Exception as e:
            logger.warning(f"Could not write unified meteorology to file: {e}")
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
            # CRITICAL: Use meteorological data's time coordinate, not scaffold's time coordinate

            # a. Copy scaffold but replace time coordinate with meteorological time
            # Drop variables that will be re-populated from observations
            # These include:
            # - 'val' suffix vars (ABGB_val, ABGB from old code)
            # - ET, NBE, LAI: Will be populated from observations
            # - Ceff: Will be handled as prior parameter
            vars_to_drop = [
                v
                for v in scaffold_ds.data_vars
                if any(x in v for x in ['val', 'ET', 'NBE', 'LAI', 'Ceff'])
            ]

            logger.debug(f"Dropping scaffold variables: {vars_to_drop}")

            # Create new dataset with meteorological time coordinate
            scaffold_copy = scaffold_ds.copy().drop_vars(vars_to_drop)

            # Remove all time-dependent variables and the old time coordinate
            time_dependent_vars = [v for v in scaffold_copy.data_vars if 'time' in scaffold_copy[v].dims]
            if time_dependent_vars:
                scaffold_copy = scaffold_copy.drop_vars(time_dependent_vars)

            # Create new dataset with meteorological time coordinate
            pixel_ds = xr.Dataset(
                coords={
                    'time': met_data.time,
                    **{k: v for k, v in scaffold_copy.coords.items() if k != 'time'}
                }
            )

            # Add non-time-dependent variables from scaffold
            for var_name in scaffold_copy.data_vars:
                pixel_ds[var_name] = scaffold_copy[var_name]

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

            # g. Add Day of Year (DOY) variable for phenological calculations
            add_doy_variable(pixel_ds)

            # h. Set MCMC attributes
            set_mcmc_attributes(pixel_ds, MCMC_ITERATIONS, MCMC_SAMPLES)

            # i. Finalize and save
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
