# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup Commands

**Python environment setup:**
```bash
# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate cardamom-ecmwf-downloader

# Install additional dependencies if needed
pip install cdsapi maap-py pystac pystac-client boto3
```

**Testing and validation:**
```bash
# Test STAC CLI
.venv/bin/python -m src.stac_cli --help
.venv/bin/python -m src.stac_cli ecmwf --help
.venv/bin/python -m src.stac_cli noaa --help
.venv/bin/python -m src.stac_cli gfed --help
.venv/bin/python -m src.stac_cli cbf-generate --help

# Test downloads
.venv/bin/python -m src.stac_cli ecmwf --variables t2m_min,t2m_max --year 2020 --month 1 --output ./test_output

# Run tests
.venv/bin/python -m pytest tests/ -v
```

## Architecture Overview

This repository implements a **STAC-based data preprocessing pipeline** for the CARDAMOM carbon cycle modeling framework. It provides modular tools for downloading, cataloging, and processing meteorological and observational data required for CARDAMOM carbon cycle analysis.

### Core Components

**STAC-Based Data Pipeline (`src/`):**
- `stac_cli.py`: Command-line interface for downloaders and CBF generation
- `stac_utils.py`: STAC catalog creation and management utilities
- `stac_met_loader.py`: Load meteorological data from STAC catalogs with completeness validation
- `cbf_main.py`: CBF file generation orchestration using STAC meteorology + user-provided observations
- `cbf_obs_handler.py`: Observational data loading with graceful NaN-filling for missing values
- `downloaders/` package: Modular downloaders (ECMWF, NOAA, GFED) with BaseDownloader abstract class

**Main Workflow:**
1. **Download data sources independently** → Each downloader creates STAC metadata
2. **Discover meteorology from STAC catalog** → Validate completeness (FAIL if missing data)
3. **Load user-provided observational data** → NaN-fill missing values for graceful degradation
4. **Generate pixel-specific CBF files** → CARDAMOM-ready format for carbon cycle modeling

**Environment Configuration:**
- `environment.yml`: Conda environment with cdsapi, xarray, netcdf4, pystac, pystac-client dependencies
- Designed for Python 3.9+ with NASA MAAP platform compatibility

### Data Flow Architecture

1. **Phase 1: Data Acquisition**
   - ECMWFDownloader → ERA5 meteorology → STAC catalog
   - NOAADownloader → Global CO₂ concentrations → STAC catalog
   - GFEDDownloader → Burned area & fire emissions → STAC catalog

2. **Phase 2: Meteorology Discovery**
   - `stac_met_loader.py` discovers all meteorological variables from STAC catalog
   - Validates completeness: FAILS if any required month is missing
   - Loads as unified xarray Dataset with standardized time coordinates

3. **Phase 3: Observational Data**
   - `cbf_obs_handler.py` loads user-provided observational constraints
   - NaN-fills missing observations for graceful degradation
   - Allows forward-only mode if no observations available

4. **Phase 4: CBF Generation**
   - `cbf_main.py` generates pixel-specific CBF files
   - Sets forcing variables from STAC meteorology
   - Sets observational constraints from user data (optional)
   - Outputs CARDAMOM-ready NetCDF files

### Key Features

**STAC Catalog Benefits:**
- **Discoverability**: Easy querying and filtering of available data
- **Metadata Richness**: Each file includes comprehensive spatiotemporal metadata
- **Incremental Updates**: Add new data without rebuilding entire catalogs
- **Validation**: Centralized validation ensures data completeness before CBF generation
- **Decoupling**: Download and CBF generation are independent, reusable steps

**Monthly-Only Workflow:**
- Focus on monthly meteorological data (no hourly/diurnal processing)
- Simplified processing pipeline optimized for monthly carbon cycle modeling
- Reduced computational complexity and storage requirements

### Authentication Requirements

**ECMWF CDS API credentials required for ERA5 downloads:**
- Local development: `.cdsapirc` file in home directory
- Environment variables: `ECMWF_CDS_UID` and `ECMWF_CDS_KEY`

## Development Patterns

**Adding new data sources:**
1. Create downloader class in `src/downloaders/` inheriting from `BaseDownloader`
2. Implement required methods: `download_data()`, `_create_stac_item()`
3. Add STAC metadata generation for outputs
4. Register new CLI subcommand in `src/stac_cli.py`
5. Update `CARDAMOM_VARIABLE_REGISTRY` in `src/cardamom_variables.py` if adding new variables

**Adding new variables to existing downloaders:**
1. Check data source documentation for exact variable names (e.g., ERA5 variable documentation)
2. Add variable to `CARDAMOM_VARIABLE_REGISTRY` with:
   - Scientific name, CBF name, units
   - Interpolation method (based on spatial characteristics)
   - Physical validation ranges
3. Update downloader's variable processing logic if needed
4. Update STAC metadata to include new variable

**Extending STAC workflow:**
1. STAC catalog as single source of truth for downloaded data
2. Each downloader creates/updates STAC collection and items
3. CBF generator discovers data through STAC catalog queries
4. Graceful degradation: NaN-fill for missing observational data
5. Validation: FAIL early if required meteorology is missing

## Connection to CARDAMOM Framework

This preprocessor creates meteorological inputs and processes data for the main CARDAMOM framework. The downloaded ERA5 data provides essential climate drivers for:

- **DALEC model simulations**: Photosynthesis, respiration, and carbon allocation processes
- **Bayesian parameter estimation**: Constraining ecosystem model parameters using observations
- **Model-data fusion**: MCMC algorithms for uncertainty quantification
- **CBF file generation**: Input format for CARDAMOM C framework execution

The preprocessor maintains compatibility with CARDAMOM's NetCDF-based data pipeline and CBF (CARDAMOM Binary Format) requirements.

### CBF (CARDAMOM Binary Format) File Generation

The preprocessor includes functionality to generate CBF files for CARDAMOM carbon cycle modeling. This process:

1. **Accepts user-provided NetCDF files** for:
   - Land-sea fraction masks (identifies valid land pixels for processing)
   - Meteorological driver data (temperature, precipitation, radiation, etc.)
   - Observational constraint data (LAI, GPP, biomass, soil moisture, etc.)
   - Fire emission data (burned area, fire carbon emissions)
   - Soil organic matter initialization data

2. **Spatial filtering**: Processes only pixels above a land fraction threshold within specified latitude/longitude ranges

3. **Pixel-level data extraction**: For each valid land pixel, extracts corresponding data from meteorological and observational sources

4. **Variable processing**:
   - Applies unit conversions (Kelvin to Celsius, m/s to mm/day, etc.)
   - Ensures physical constraints (positive values where appropriate)
   - Calculates derived variables (Vapor Pressure Deficit from temperature and dewpoint)

5. **CBF structure creation**: Generates NetCDF files with:
   - Meteorological forcing variables
   - Observational constraints with uncertainty specifications
   - Single-value constraints (initial soil carbon, plant carbon use efficiency, etc.)
   - MCMC configuration parameters

6. **Output format**: Creates CBF-compliant NetCDF files following CARDAMOM specifications for the carbon cycle data assimilation system

**User-provided data requirements:**
- All input files must be in NetCDF format
- Consistent spatial grids across meteorological and observational datasets
- Compatible temporal dimensions (or scalar for single-value constraints)
- Proper variable naming (documented in input data specifications)

## Scientist-Friendly Coding Standards

This project prioritizes code readability for scientists who may not be proficient in Python. All implementors must follow these guidelines to ensure the codebase remains accessible to the scientific community.

### Core Principles

**1. Scientific Clarity Over Python Cleverness**
- Write code that a domain scientist can understand, even if it's more verbose
- Avoid complex Python idioms, list comprehensions, and lambda functions
- Prefer explicit operations over implicit ones

**2. Self-Documenting Code**
- Use variable names that reflect scientific meaning and include units when relevant
- Function names should describe what scientific operation they perform
- Code structure should mirror the scientific workflow

### Variable Naming Conventions

**Good Examples:**
```python
# Clear scientific meaning with units
temperature_celsius = era5_data['2m_temperature'] - 273.15
vapor_pressure_deficit_hpa = calculate_vpd(temp_max_k, dewpoint_k)
co2_concentration_ppm = load_noaa_co2_data(year, month)
photosynthetically_active_radiation_umol_m2_s = solar_rad * 0.45

# Clear array dimensions and scientific context
spatial_grid_lat_lon = create_coordinate_grid(resolution_degrees=0.5)
monthly_precipitation_mm = aggregate_to_monthly(hourly_precip_data)
```

**Avoid:**
```python
# Unclear or abbreviated names
t = data['temp'] - 273.15
vpd = calc_vpd(tmax, td)
co2 = load_data(y, m)
par = sr * 0.45

# Generic names without scientific context
arr = create_grid(0.5)
monthly_data = aggregate(hourly_data)
```

### Function Design

**Structure functions to match scientific thinking:**
```python
def calculate_vapor_pressure_deficit(temperature_max_kelvin, dewpoint_temperature_kelvin):
    """
    Calculate Vapor Pressure Deficit from temperature and dewpoint.

    VPD represents the atmospheric moisture demand and is crucial for
    understanding plant water stress and photosynthesis rates.

    Scientific Background:
    VPD = e_sat(T_max) - e_sat(T_dewpoint)
    where e_sat is saturation vapor pressure calculated using Tetens equation.

    Args:
        temperature_max_kelvin (float or array): Daily maximum temperature in Kelvin
            Typical range: 250-320 K (-23 to 47°C)
        dewpoint_temperature_kelvin (float or array): Dewpoint temperature in Kelvin
            Typical range: 230-300 K (-43 to 27°C)

    Returns:
        float or array: Vapor pressure deficit in hectopascals (hPa)
            Typical range: 0-60 hPa
            - Low VPD (0-10 hPa): High humidity, low atmospheric demand
            - Medium VPD (10-30 hPa): Moderate atmospheric demand
            - High VPD (>30 hPa): Low humidity, high atmospheric demand

    References:
        Tetens, O. (1930). Über einige meteorologische Begriffe.
        Zeitschrift für Geophysik, 6, 297-309.
    """

    # Step 1: Calculate saturation vapor pressure at maximum temperature
    # Using Tetens equation: e_sat = 6.1078 * exp(17.27 * T_c / (T_c + 237.3))
    temp_max_celsius = temperature_max_kelvin - 273.15
    saturation_pressure_at_tmax_hpa = 6.1078 * np.exp(
        17.27 * temp_max_celsius / (temp_max_celsius + 237.3)
    )

    # Step 2: Calculate saturation vapor pressure at dewpoint temperature
    dewpoint_celsius = dewpoint_temperature_kelvin - 273.15
    saturation_pressure_at_dewpoint_hpa = 6.1078 * np.exp(
        17.27 * dewpoint_celsius / (dewpoint_celsius + 237.3)
    )

    # Step 3: Calculate VPD as the difference
    vapor_pressure_deficit_hpa = (saturation_pressure_at_tmax_hpa -
                                 saturation_pressure_at_dewpoint_hpa)

    # Step 4: Validate results are physically reasonable
    if np.any(vapor_pressure_deficit_hpa < 0):
        raise ValueError("VPD cannot be negative. Check that T_max >= T_dewpoint")

    return vapor_pressure_deficit_hpa
```

### Documentation Standards

**Every function must include:**

1. **Scientific Purpose**: What does this function do in scientific terms?
2. **Scientific Background**: Brief explanation of the underlying science
3. **Units**: Explicit units for all parameters and return values
4. **Typical Ranges**: Expected value ranges for inputs and outputs
5. **Physical Interpretation**: What do the results mean scientifically?
6. **References**: Citations to scientific literature when applicable

**Example Class Documentation:**
```python
class ECMWFDataProcessor:
    """
    Process ERA5 reanalysis data for CARDAMOM carbon cycle modeling.

    This class handles the conversion of raw ERA5 meteorological data into
    the specific format and variables required by the CARDAMOM Data Assimilation
    System for carbon cycle analysis.

    Scientific Context:
    CARDAMOM requires meteorological drivers including temperature, precipitation,
    radiation, and humidity to constrain ecosystem carbon fluxes. ERA5 provides
    these variables at high spatial and temporal resolution globally.

    Key Transformations:
    - Temperature: Convert from Kelvin to Celsius where needed
    - Precipitation: Convert from m/s to mm/day
    - Radiation: Convert from J/m² to W/m² and apply PAR conversion
    - Humidity: Calculate VPD from temperature and dewpoint

    Attributes:
        resolution_degrees (float): Spatial resolution in decimal degrees
        time_aggregation (str): Temporal aggregation ('hourly', 'daily', 'monthly')
        quality_control_enabled (bool): Whether to apply QC checks
    """
```

### Error Handling for Scientists

**Provide clear, scientifically meaningful error messages:**
```python
def validate_temperature_data(temperature_kelvin):
    """Validate temperature data for physical reasonableness."""

    if np.any(temperature_kelvin < 173):  # -100°C
        problematic_values = temperature_kelvin[temperature_kelvin < 173]
        raise ValueError(
            f"Temperature values below -100°C detected: {problematic_values}. "
            f"This is below the lowest recorded Earth temperature (-89°C). "
            f"Check data units - temperature should be in Kelvin."
        )

    if np.any(temperature_kelvin > 333):  # 60°C
        problematic_values = temperature_kelvin[temperature_kelvin > 333]
        raise ValueError(
            f"Temperature values above 60°C detected: {problematic_values}. "
            f"This exceeds typical meteorological ranges. "
            f"Check data units and spatial domain."
        )

def load_precipitation_data(file_path):
    """Load precipitation data with scientific validation."""
    try:
        precipitation_data = read_netcdf_variable(file_path, 'precipitation')
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Precipitation file not found: {file_path}. "
            f"Check that ERA5 download completed successfully. "
            f"Required variables: total_precipitation in m/s units."
        )
    except KeyError:
        available_vars = list_netcdf_variables(file_path)
        raise KeyError(
            f"Precipitation variable not found in {file_path}. "
            f"Available variables: {available_vars}. "
            f"Expected 'total_precipitation' or 'tp' in ERA5 format."
        )

    return precipitation_data
```

### Code Organization Principles

**1. Mirror Scientific Workflow:**
```python
def process_monthly_meteorology(year, month):
    """Process monthly meteorological data following scientific workflow."""

    # Step 1: Data Acquisition
    print(f"Downloading ERA5 data for {year}-{month:02d}")
    era5_data = download_era5_monthly_data(year, month)

    # Step 2: Quality Control
    print("Applying meteorological quality control checks")
    validated_data = apply_meteorological_qc(era5_data)

    # Step 3: Unit Conversions
    print("Converting to CARDAMOM standard units")
    standardized_data = convert_to_cardamom_units(validated_data)

    # Step 4: Scientific Calculations
    print("Computing derived meteorological variables")
    derived_variables = calculate_derived_meteorology(standardized_data)

    # Step 5: Spatial Processing
    print("Regridding to CARDAMOM spatial resolution")
    regridded_data = regrid_to_cardamom_grid(derived_variables)

    # Step 6: Output Generation
    print("Creating CARDAMOM-format NetCDF files")
    create_cardamom_netcdf_files(regridded_data, year, month)

    return regridded_data
```

**2. Use Intermediate Variables:**
```python
# Good: Clear intermediate steps
def calculate_net_ecosystem_exchange(gpp_flux, respiration_flux, fire_emissions):
    """Calculate net ecosystem exchange of CO2."""

    # NEE = Respiration + Fire - GPP (following sign convention: positive = source to atmosphere)
    ecosystem_respiration_component = respiration_flux
    fire_emission_component = fire_emissions
    photosynthetic_uptake_component = -gpp_flux  # Negative because GPP removes CO2

    net_ecosystem_exchange = (ecosystem_respiration_component +
                            fire_emission_component +
                            photosynthetic_uptake_component)

    return net_ecosystem_exchange

# Avoid: Complex one-liners
def calculate_nee(gpp, resp, fire):
    return resp + fire - gpp  # Unclear sign conventions and components
```

### Scientific Validation

**Include validation functions that check scientific reasonableness:**
```python
def validate_carbon_flux_data(carbon_flux_data, flux_type):
    """
    Validate carbon flux data for physical and ecological reasonableness.

    Args:
        carbon_flux_data: Carbon flux values in gC/m²/day
        flux_type: Type of flux ('GPP', 'NEE', 'respiration', 'fire')

    Returns:
        dict: Validation results with pass/fail status and diagnostics
    """

    validation_results = {
        'flux_type': flux_type,
        'data_range': {
            'min': float(np.nanmin(carbon_flux_data)),
            'max': float(np.nanmax(carbon_flux_data)),
            'mean': float(np.nanmean(carbon_flux_data))
        }
    }

    # Define expected ranges based on scientific literature
    expected_ranges = {
        'GPP': {'min': 0, 'max': 50, 'typical_max': 30},      # GPP always positive
        'respiration': {'min': 0, 'max': 30, 'typical_max': 20},  # Respiration always positive
        'NEE': {'min': -30, 'max': 20, 'typical_range': (-10, 10)},  # NEE can be negative (sink) or positive (source)
        'fire': {'min': 0, 'max': 100, 'typical_max': 10}     # Fire emissions always positive
    }

    if flux_type not in expected_ranges:
        validation_results['status'] = 'unknown_flux_type'
        return validation_results

    ranges = expected_ranges[flux_type]

    # Check for values outside physical limits
    below_min = np.sum(carbon_flux_data < ranges['min'])
    above_max = np.sum(carbon_flux_data > ranges['max'])

    validation_results['physical_check'] = {
        'values_below_minimum': below_min,
        'values_above_maximum': above_max,
        'status': 'pass' if (below_min == 0 and above_max == 0) else 'fail'
    }

    # Check for values outside typical ecological ranges
    if 'typical_max' in ranges:
        above_typical = np.sum(carbon_flux_data > ranges['typical_max'])
        validation_results['ecological_check'] = {
            'values_above_typical': above_typical,
            'status': 'pass' if above_typical < len(carbon_flux_data) * 0.01 else 'warning'  # Allow 1% outliers
        }

    return validation_results
```

### Example vs. Anti-Example Patterns

**Configuration Management:**
```python
# Good: Clear, self-documenting configuration
class CARDAMOMProcessingConfig:
    def __init__(self):
        # Spatial configuration with clear scientific meaning
        self.global_grid_resolution_degrees = 0.5
        self.global_grid_bounds_lat_lon = [-89.75, -179.75, 89.75, 179.75]  # [South, West, North, East]

        # Temporal configuration
        self.processing_time_step = 'monthly'
        self.reference_year_start = 2001
        self.reference_year_end = 2020

        # Scientific processing options
        self.apply_quality_control_checks = True
        self.calculate_derived_variables = True
        self.perform_spatial_interpolation = True

        # File organization following scientific workflow
        self.output_directory_structure = {
            'meteorology': './DATA/CARDAMOM-MAPS_05deg_MET/',
            'carbon_fluxes': './DATA/CARDAMOM-MAPS_05deg_FLUX/',
            'fire_emissions': './DATA/CARDAMOM-MAPS_05deg_FIRE/',
            'validation_reports': './REPORTS/QC/'
        }

# Avoid: Cryptic or generic configuration
class Config:
    def __init__(self):
        self.res = 0.5
        self.bounds = [-89.75, -179.75, 89.75, 179.75]
        self.step = 'monthly'
        self.yr1, self.yr2 = 2001, 2020
        self.qc = True
        self.calc_derived = True
        self.interp = True
        self.dirs = {'met': './DATA/MET/', 'flux': './DATA/FLUX/'}
```

### Testing with Scientific Context

**Write tests that validate scientific correctness:**
```python
def test_vapor_pressure_deficit_calculation():
    """Test VPD calculation against known meteorological values."""

    # Test case 1: Standard atmospheric conditions
    # Temperature: 25°C (298.15 K), Dewpoint: 15°C (288.15 K)
    # Expected VPD ≈ 11.7 hPa (calculated from Tetens equation)
    temp_max_k = 298.15
    dewpoint_k = 288.15
    expected_vpd_hpa = 11.7

    calculated_vpd = calculate_vapor_pressure_deficit(temp_max_k, dewpoint_k)

    assert abs(calculated_vpd - expected_vpd_hpa) < 0.1, (
        f"VPD calculation failed for standard conditions. "
        f"Expected: {expected_vpd_hpa} hPa, Got: {calculated_vpd} hPa"
    )

    # Test case 2: Saturated conditions (dewpoint = temperature)
    # When dewpoint equals temperature, VPD should be zero
    temp_saturated = 293.15  # 20°C
    dewpoint_saturated = 293.15  # 20°C

    vpd_saturated = calculate_vapor_pressure_deficit(temp_saturated, dewpoint_saturated)

    assert abs(vpd_saturated) < 0.01, (
        f"VPD should be near zero for saturated conditions. "
        f"Got: {vpd_saturated} hPa"
    )

    # Test case 3: Error condition (dewpoint > temperature)
    # This is physically impossible and should raise an error
    with pytest.raises(ValueError, match="VPD cannot be negative"):
        calculate_vapor_pressure_deficit(288.15, 298.15)  # T < T_dewpoint

def test_carbon_flux_unit_conversion():
    """Test carbon flux unit conversions maintain mass balance."""

    # Test conversion from gC/m²/day to µmol CO2/m²/s
    carbon_flux_gc_m2_day = 10.0  # Typical GPP value

    # Known conversion: 1 gC/m²/day = 0.9645 µmol CO2/m²/s
    expected_flux_umol_co2_m2_s = 9.645

    converted_flux = convert_carbon_flux_units(
        carbon_flux_gc_m2_day,
        from_units='gC_m2_day',
        to_units='umolCO2_m2_s'
    )

    assert abs(converted_flux - expected_flux_umol_co2_m2_s) < 0.001, (
        f"Carbon flux unit conversion failed. "
        f"Expected: {expected_flux_umol_co2_m2_s}, Got: {converted_flux}"
    )
```

These standards ensure that all code in the CARDAMOM prepr  ocessor remains accessible to scientists while maintaining technical excellence. Remember: when in doubt, choose clarity over cleverness.
- use .venv for python commands
- keep testing simple not too extensive
- remember to look at matlab-migration/cardamom-matlab for matlab functions
- do not use relative imports as this is a python package
- remember that ultimately we need to create outputs that can be used in @matlab-migration/erens_cbf_code.py
- Read matlab files from /Users/shah/Desktop/Development/ghg/CARDAMOM/MATLAB/prototypes