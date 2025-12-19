# CBF Generator - STAC Integration Implementation Summary

## Overview

Successfully integrated STAC meteorological data discovery with CARDAMOM Binary Format (CBF) generation pipeline. The implementation maintains maximum code reuse from `erens_cbf_code.py` while adding STAC-based data discovery.

## Key Design Decisions

### ✅ Data Validation Strategy

**Meteorological Data (from STAC):**
- **REQUIRED**: All 10 variables must be present for ALL required months
- **Behavior**: Program FAILS immediately if any variable/month is missing
- **Rationale**: Invalid meteorology cannot produce valid CARDAMOM simulations
- **Variables**: VPD, TOTAL_PREC, T2M_MIN, T2M_MAX, STRD, SSRD, SNOWFALL, CO2, BURNED_AREA, SKT

**Observational Data (user-provided):**
- **OPTIONAL**: Can have spatial/temporal gaps
- **Behavior**: Missing values filled with NaN, processing continues
- **Rationale**: Forward-only model runs don't require observations; obs data improves parameter estimation
- **Variables**: SCF, GPP, ABGB, EWT (LAI, SOM, Mean_FIR)

### ✅ Variable Naming

**STAC CLI Output Variables (standardized):**
- Uses consistent naming: `T2M_MIN`, `T2M_MAX`, `TOTAL_PREC`, `SNOWFALL`, `BURNED_AREA`
- NO renaming needed (unlike Eren's `TMIN`/`TMAX`/`PREC`/`SNOW`/`BURN_2`)
- Simplifies integration and reduces mapping complexity

### ✅ Code Organization

**Maximum Reuse Principle:**
- Kept ALL helper functions from `erens_cbf_code.py` unmodified
- Only replaced meteorology loading mechanism
- All pixel processing logic unchanged

## Files Implemented

### 1. **src/stac_met_loader.py** (400 lines)

**Purpose:** Discover, load, and validate meteorological data from STAC

**Key Functions:**
- `load_met_data_from_stac()` - Main entry point
  - Discovers STAC items for all 10 required variables
  - Loads monthly NetCDF files
  - Validates completeness (FAILS if gaps)
  - Assembles unified dataset

- `discover_stac_items()` - Query STAC API for variable
- `load_variable_from_stac_items()` - Load monthly files and concatenate
- `validate_meteorology_completeness()` - **CRITICAL**: Fail if incomplete
- `assemble_unified_meteorology_dataset()` - Merge variables into single dataset

**Data Flow:**
```
STAC Catalog
    ↓ discover items
[Items for 10 variables]
    ↓ load monthly files
[Dataset per variable]
    ↓ validate completeness (FAIL if gaps)
[Validated datasets]
    ↓ merge/assemble
Unified meteorology dataset
    ↓
CBF pixel extraction
```

### 2. **src/cbf_obs_handler.py** (250 lines)

**Purpose:** Load observational data with graceful NaN-filling for missing values

**Key Functions:**
- `load_observational_data_with_nan_fill()` - Load obs files
  - Tries to load main obs file, SOM, FIR
  - Continues if some files missing (logs warnings)
  - Returns None if all fail

- `fill_missing_obs_variable()` - Get variable or NaN array
- `get_pixel_obs_value_with_nan_fallback()` - Get scalar or NaN

**Behavior:**
- If obs file missing: Returns None → forward-only mode
- If obs variable missing: Returns NaN array → graceful degradation
- If obs pixel missing: Returns NaN → interpolation possible later
- Allows CBF generation to proceed without obs data

### 3. **src/cbf_main.py** (350 lines)

**Purpose:** Main orchestrator combining STAC met + user-provided obs + pixel processing

**Configuration:**
```python
# STAC meteorology
STAC_API_URL = 'file:///path/to/stac/catalog.json'
START_DATE = '2001-01'
END_DATE = '2021-12'

# User-provided files (non-STAC)
LAND_FRAC_FILE = 'input/CARDAMOM-MAPS_05deg_LAND_SEA_FRAC.nc'
OBS_DRIVER_FILE = 'input/AlltsObs05x05newbiomass_LFmasked.nc'
SOM_FILE = 'input/CARDAMOM-MAPS_05deg_HWSD_PEQ_iniSOM.nc'
FIR_FILE = 'input/CARDAMOM-MAPS_05deg_GFED4_Mean_FIR.nc'
SCAFFOLD_FILE = 'input/fluxval_US-NR1_1100_LAI.cbf.nc'
```

**Workflow (main() function):**
1. Load meteorology from STAC (FAIL if incomplete)
2. Load land fraction mask
3. Find land pixels (>0.5 threshold)
4. Load obs data (NaN-fill for gaps)
5. Load scaffold template
6. For each pixel:
   - Copy scaffold
   - Set forcing variables from met
   - Set obs constraints (if available)
   - Set single-value constraints (if available)
   - Apply assimilation attributes
   - Set MCMC parameters
   - Save CBF file

**Reused from erens_cbf_code.py:**
- `load_and_preprocess_land_fraction()`
- `find_land_pixels()`
- `generate_output_filepaths()`
- `load_scaffold_data()`
- `set_forcing_variables()`
- `set_observation_constraints()`
- `set_single_value_constraints()`
- `adjust_assimilation_attributes()`
- `set_mcmc_attributes()`
- `finalize_and_save()`

## Architecture Comparison

### Before (Eren's Approach)
```
AllMet05x05_LFmasked.nc (1 file, 17 variables, 252 months)
    ↓ load directly
Met dataset (complete)
    ↓
Pixel processing
    ↓
CBF files
```

### After (STAC Integration)
```
STAC Catalog (10 collection × 252 files)
    ↓ discover & load
t2m_min_2020_01.nc, vpd_2020_01.nc, ... (10 monthly files per variable)
    ↓ validate & assemble
Met dataset (complete or FAIL)
    ↓ (+ obs files with NaN-fill)
Pixel processing
    ↓
CBF files
```

**Key Advantages:**
- Automatic STAC discovery (no manual file management)
- Validation at load time (fail fast)
- Flexible date ranges (can download different time periods)
- Modular design (can extend to other data sources)

## Variable Mapping Summary

**STAC Naming (What we use) → Eren's Naming (Reference)**

| STAC Name | Eren's Name | Description |
|-----------|------------|-------------|
| T2M_MIN | TMIN | Min 2m temperature |
| T2M_MAX | TMAX | Max 2m temperature |
| TOTAL_PREC | PREC | Total precipitation |
| SNOWFALL | SNOW | Snowfall |
| BURNED_AREA | BURN_2 | Burned area fraction |
| VPD | VPD | Vapor pressure deficit |
| SSRD | SSRD | Surface solar radiation |
| STRD | STRD | Surface thermal radiation |
| SKT | SKT | Skin temperature |
| CO2 | CO2_2 | Atmospheric CO2 |

**No renaming needed** - STAC already has standardized names.

## Error Handling Strategy

### Meteorological Data (CRITICAL)
```python
try:
    met_data = load_met_data_from_stac(...)
except ValueError as e:
    # Missing variable/month detected
    logger.error(f"CRITICAL: {e}")
    return  # Exit - cannot generate valid CBF
```

### Observational Data (OPTIONAL)
```python
obs_data = load_observational_data_with_nan_fill(...)
if obs_data is None:
    logger.warning("No obs data available - forward-only mode")
    # Continue with NaN constraints

try:
    set_observation_constraints(...)
except:
    logger.debug("Could not set obs constraint - using NaN")
    # Continue with NaN values
```

## Testing Strategy

### Phase 1: Unit Tests
- [ ] Test STAC meteorology loader with sample catalog
- [ ] Test variable discovery for each of 10 variables
- [ ] Test validation (should fail on missing month)
- [ ] Test obs data loading (NaN-fill for missing files)

### Phase 2: Validation Tests
- [ ] Met data completeness: Should FAIL if any variable missing
- [ ] Obs data NaN-fill: Should SUCCEED with partial obs
- [ ] Variable naming: Verify all STAC names used correctly

### Phase 3: Integration Tests
- [ ] Run CBF generation on small test region (5×5 pixels)
- [ ] Verify CBF file structure matches CARDAMOM format
- [ ] Check variable attributes and dimensions
- [ ] Compare with Eren's sample CBF files using `ncdump`

### Phase 4: Full Validation
- [ ] Generate full CONUS CBF files
- [ ] Verify file counts and naming
- [ ] Spot-check pixel data values
- [ ] Run CARDAMOM model with generated CBF files

## Next Steps

1. **User Configuration**
   - Set `STAC_API_URL` to point to local STAC catalog
   - Set `START_DATE` and `END_DATE` for desired time period
   - Provide user-provided input files (obs, land mask, SOM, FIR, scaffold)

2. **Run CBF Generation**
   ```bash
   python -m src.cbf_main
   ```

3. **Validate Output**
   ```bash
   # Check file structure
   ncdump -h output/CBF_001/site35_25N105_75W_ID001exp0.cbf.nc

   # Compare with Eren's sample
   ncdump -h matlab-migration/sample-data-from-eren/fluxval_US-NR1_1100_LAI.cbf.nc
   ```

## Code Quality Notes

**Scientific Clarity:**
- All functions include scientific context and rationale
- Variable names include units where relevant
- Clear documentation of assumptions and constraints
- Proper error messages for users

**Robustness:**
- Comprehensive logging for debugging
- Graceful handling of missing optional data
- Clear failure messages for critical data
- Progress reporting during pixel processing

**Maintainability:**
- Maximum code reuse from erens_cbf_code.py
- Modular design (separate STAC loader, obs handler, main)
- Clear separation of concerns
- Minimal changes to existing working code

## References

- Original: `matlab-migration/erens_cbf_code.py`
- STAC Spec: https://stacspec.org/
- CARDAMOM: https://www2.geog.umd.edu/~sjd/cardamom.html
