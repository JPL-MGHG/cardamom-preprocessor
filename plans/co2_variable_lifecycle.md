# CO2 Variable Lifecycle Documentation

## Overview

This document traces the complete lifecycle of the atmospheric CO2 variable from its source at NOAA Global Monitoring Laboratory through its integration into CARDAMOM-compliant CBF files. The CO2 variable provides the essential atmospheric boundary condition for carbon cycle modeling in CARDAMOM.

## 1. Data Source & Acquisition

### 1.1 NOAA Data Source
- **Primary Source**: NOAA Global Monitoring Laboratory (GML)
- **Original URL**: `ftp://aftp.cmdl.noaa.gov/products/trends/co2/co2_mm_gl.txt` *(outdated)*
- **Current URL**: `https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_gl.txt`
- **Data Type**: Global monthly mean atmospheric CO2 concentrations
- **Measurement Origin**: Mauna Loa Observatory + global network
- **Units**: Parts per million (ppm)
- **Temporal Range**: 1958-present, monthly resolution

### 1.2 Data Characteristics
```
# NOAA CO2 file format (co2_mm_gl.txt):
# year  month  decimal_date  average  interpolated  trend  #days
# Missing values represented as -99.99
# Example:
2023    1     2023.042    421.08    421.08    421.58     31
2023    2     2023.125    422.13    422.13    421.71     28
```

**Key Scientific Context**: This represents the **background atmospheric CO2 concentration** that affects all terrestrial ecosystems globally. It's a single global mean value, not spatially resolved.

## 2. Python Implementation: Download & Processing

### 2.1 NOAA Downloader (`src/downloaders/noaa_downloader.py`)

**Primary Class**: `NOAADownloader(BaseDownloader)`

**Status**: ✅ **Updated** - Now uses HTTPS and downloads entire dataset by default

#### Download Process:
```python
# HTTPS Configuration (Updated)
NOAA_CO2_URL = 'https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_gl.csv'

# Download method
def _download_co2_data(self) -> Dict[int, Dict[int, float]]
    # Downloads entire CO2 dataset via HTTPS
    # Parses CSV format: year,month,decimal_date,average_observed,std_dev,trend,trend_std
```

**Key Improvements**:
- Uses HTTPS instead of FTP
- Downloads entire dataset (small ~100KB CSV file)
- Supports both full dataset and single-month modes

#### Data Parsing:
```python
# Parse NOAA CSV format
# Format: year,month,decimal_date,average_observed,std_dev,trend,trend_std
# Example: 2020,1,2020.042,411.72,0.15,410.23,0.10

for line in response.text.split('\n'):
    if not line or line.startswith('#'):
        continue

    parts = line.split(',')
    year = int(parts[0].strip())
    month = int(parts[1].strip())
    co2_ppm = float(parts[3].strip())  # Use 'average_observed' column
```

#### Spatial Replication:
```python
# For full dataset: Create 3D array [time, lat, lon]
num_lat = int((89.75 - (-89.75)) / 0.5) + 1  # 360 points
num_lon = int((179.75 - (-179.75)) / 0.5) + 1  # 720 points

co2_grid_3d = np.zeros((num_time, num_lat, num_lon), dtype=np.float32)
for t_idx, co2_val in enumerate(co2_values):
    co2_grid_3d[t_idx, :, :] = co2_val  # Replicate spatially for each time step
```

#### Two Download Modes:

**Mode 1: Full Dataset (Default - Recommended)**
```python
# Download all available CO2 data
results = downloader.download_and_process()
# Creates: co2_1958_2024.nc (single file with complete time series)
```

**Mode 2: Single Month (Backwards Compatibility)**
```python
# Download specific month
results = downloader.download_and_process(year=2020, month=1)
# Creates: co2_2020_01.nc (single month file)
```

### 2.2 NetCDF Creation Process

#### Variable Creation (Full Dataset Mode):
```python
dataset = xr.Dataset(
    {
        'CO2': (
            ['time', 'latitude', 'longitude'],
            co2_grid_3d,  # Shape: (num_months, 360, 720)
            {
                'long_name': 'Atmospheric CO2 concentration',
                'units': 'ppm',
                'standard_name': 'mole_fraction_of_carbon_dioxide_in_air',
                'description': 'Global monthly mean atmospheric CO2 concentration replicated spatially'
            }
        )
    },
    coords={
        'time': time_steps,  # List of datetime objects
        'latitude': lats,    # -89.75 to 89.75 by 0.5°
        'longitude': lons    # -179.75 to 179.75 by 0.5°
    }
)
```

#### Output Files:
- **Full Dataset Mode**:
  - Pattern: `co2_{start_year}_{end_year}.nc` (e.g., `co2_1958_2024.nc`)
  - Dimensions: `[time: ~800 months, latitude: 360, longitude: 720]`

- **Single Month Mode**:
  - Pattern: `co2_{year}_{month:02d}.nc` (e.g., `co2_2020_01.nc`)
  - Dimensions: `[time: 1, latitude: 360, longitude: 720]`

- **Directory**: `{output_dir}/data/` (default)

**Scientific Rationale**: Since atmospheric CO2 is well-mixed globally and the source file is small, downloading the entire time series is more efficient and provides complete historical context for CARDAMOM modeling.

## 3. Integration into CARDAMOM Pipeline

### 3.1 CBF Met Processor Integration (`src/cbf_met_processor.py`)

#### Variable Mapping:
```python
# Required CBF variables (line 55)
MET_VARS_TO_KEEP = ['VPD', 'TOTAL_PREC', 'T2M_MIN', 'T2M_MAX',
                    'STRD', 'SSRD', 'SNOW', 'CO2_2', 'BURN_2', 'SKT']

# CO2_2 identified as external source (line 68)
# CO2_2 and BURN_2 from external sources
```

#### Integration Process (lines 651-663):
```python
# Primary integration pathway
if co2_data_dir:
    try:
        co2_dataset = self._load_co2_data(co2_data_dir, reference_coords)
        external_variables['CO2_2'] = co2_dataset           # Line 655
        self.logger.info("CO2 data integration completed")  # Line 656
    except Exception as e:
        # Fallback to constant value
        external_variables['CO2_2'] = self._create_constant_co2_dataset(reference_coords)  # Line 660

# No CO2 directory provided - use fallback
else:
    external_variables['CO2_2'] = self._create_constant_co2_dataset(reference_coords)      # Line 663
```

#### Fallback Mechanism (lines 682-702):
```python
def _create_constant_co2_dataset(self, reference_coords):
    """Create constant CO2 dataset based on reference coordinates."""

    # Fallback value: 415 ppm (line 689)
    co2_data = xr.DataArray(
        data=np.full(time_shape, 415.0),    # Modern atmospheric CO2 concentration
        coords=reference_coords,
        attrs={
            'long_name': 'Atmospheric CO2 Concentration',
            'units': 'ppm',
            'description': 'Constant atmospheric CO2 concentration',
            'source': 'Fixed value fallback'
        }
    )
    co2_data.name = 'CO2_2'              # Line 700
    return xr.Dataset({'CO2_2': co2_data})  # Line 702
```

#### Data Loading & Interpolation (lines 725-763):
```python
def _load_co2_data(self, co2_data_dir, reference_coords):
    """Load CO2 data from directory and interpolate to reference grid."""

    # Search patterns for CO2 files (line 727)
    for pattern in ['*co2*.nc', '*CO2*.nc', '*noaa*.nc', '*NOAA*.nc']:

    # Variable name detection (line 741)
    co2_var_names = ['co2', 'CO2', 'co2_concentration', 'mole_fraction']

    # Final variable assignment (lines 761-763)
    co2_interp.name = 'CO2_2'
    return xr.Dataset({'CO2_2': co2_interp})
```

### 3.2 ECMWF Integration (`src/ecmwf_downloader.py`)

#### Optional Integration (lines 1154-1161):
```python
# Integrate CO2 data
if co2_data:
    try:
        co2_var = self._integrate_co2_data(co2_data, integrated_ds)
        integrated_ds['CO2'] = co2_var                    # Note: 'CO2' not 'CO2_2'
        self.logger.info("CO2 data integration completed")
    except Exception as e:
        self.logger.warning(f"Failed to integrate CO2 data: {e}")
```

#### Variable Name Mapping (line 1242):
```python
co2_var_names = ['co2', 'CO2', 'co2_concentration', 'mole_fraction']
```

## 4. MATLAB Reference Implementation

### 4.1 Expected MATLAB Processing Pattern

Based on the Python implementation comments, the original MATLAB processing likely follows this pattern:

```matlab
% MATLAB Reference (inferred from Python comments):
% 1. Load NOAA CO2 data
NOAACO2 = load_noaa_co2_data(year, month);

% 2. Spatial replication across global grid
% Python line 287 comment: "matches MATLAB repmat logic"
CO2_global = repmat(permute(NOAACO2.data), [lat_dim, lon_dim, 1]);

% 3. Integration with meteorological variables
% From CARDAMOM_MAPS_05deg_DATASETS_JUL24.m (referenced in multiple files)
MET_VARS = struct();
MET_VARS.CO2 = CO2_global;  % Add to meteorological variable structure
```

### 4.2 Expected Variable Transformations

```matlab
% Variable naming conventions (inferred):
% MATLAB: 'CO2' → Python: 'CO2_2' (CBF format)
% MATLAB: co2_concentration → Python: co2_mole_fraction (NetCDF standard)
```

### 4.3 MATLAB File References

Based on documentation references:
- **Primary MATLAB file**: `CARDAMOM_MAPS_05deg_DATASETS_JUL24.m` (referenced in plans/README.md:10)
- **Function location**: VPD calculation reference at line 202 (plans/README_PHASE8.md:17)
- **Integration point**: Meteorological variable creation and NetCDF writing section

## 5. CBF File Integration

### 5.1 Final CBF Variable Format (`matlab-migration/erens_cbf_code.py`)

#### Variable Requirements:
```python
# Expected in AllMet05x05_LFmasked.nc (lines referenced from erens_cbf_code.py)
MET_RENAME_MAP = {
    'CO2_2': 'CO2',           # Rename for CBF compatibility
    'PREC': 'TOTAL_PREC',
    # ... other variables
}

MET_VARS_TO_KEEP = ['VPD', 'TOTAL_PREC', 'T2M_MIN', 'T2M_MAX',
                    'STRD', 'SSRD', 'SNOWFALL', 'CO2', 'BURNED_AREA', 'SKT']
```

#### CBF Processing:
```python
def set_forcing_variables(target_ds, source_met_ds, lat, lon, scaffold_ds, positive_vars):
    """Sets forcing variables in the target dataset using source met data."""

    # CO2 becomes forcing variable for CARDAMOM model
    # Used as atmospheric boundary condition for photosynthesis calculations
    # Applied to all grid cells as global background concentration
```

### 5.2 Scientific Usage in CARDAMOM

**Purpose**: The CO2 variable provides the atmospheric boundary condition for:
1. **Photosynthesis calculations**: Affects CO2 assimilation rates in vegetation models
2. **Stomatal conductance**: Influences plant water use efficiency
3. **Carbon cycle constraints**: Provides atmospheric CO2 context for land-atmosphere exchange

**Spatial Treatment**: Since atmospheric CO2 is well-mixed globally, the single NOAA measurement is appropriately replicated across all grid cells.

## 6. Data Flow Summary

```
NOAA GML Data Source (HTTPS)
        ↓
    [co2_mm_gl.csv]                 (CSV format, ~100KB)
        ↓
NOAADownloader._download_co2_data() [src/downloaders/noaa_downloader.py]
        ↓
    Parse CSV and create time series
        ↓
NOAADownloader._download_all_data() OR _download_single_month()
        ↓
    [co2_1958_2024.nc]              (full dataset - recommended)
    OR
    [co2_YYYY_MM.nc]                (single month - backwards compatibility)
        ↓
    (spatial replication occurs during NetCDF creation)
        ↓
CBFMetProcessor._load_co2_data()     [src/cbf_met_processor.py:725-763]
        ↓
External variable integration        [src/cbf_met_processor.py:651-663]
        ↓
    [AllMet05x05_LFmasked.nc]       (CO2_2 variable)
        ↓
CBF file creation                    [matlab-migration/erens_cbf_code.py]
        ↓
    [site*.cbf.nc]                  (CO2 forcing variable)
        ↓
CARDAMOM Model Input                 (atmospheric boundary condition)
```

## 7. Recent Updates (2024)

### 7.1 Resolved Issues:
✅ **Fixed Download URL**: Updated from outdated FTP to HTTPS
   - **Old (broken)**: `ftp://aftp.cmdl.noaa.gov/products/trends/co2/co2_mm_gl.txt`
   - **New (working)**: `https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_gl.csv`

✅ **Protocol Change**: Replaced `ftplib` with `requests` for HTTPS download

✅ **Full Dataset Download**: Now downloads entire time series by default
   - More efficient than monthly downloads (source file is small)
   - Provides complete historical context
   - Single NetCDF file with all time steps

✅ **Backwards Compatibility**: Single-month download mode still available

### 7.2 Current Implementation:
```python
# Updated implementation in src/downloaders/noaa_downloader.py

# HTTPS URL
NOAA_CO2_URL = 'https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_gl.csv'

# Download entire dataset (default)
results = downloader.download_and_process()
# Creates: co2_1958_2024.nc (~60MB, complete time series)

# Or download specific month (backwards compatibility)
results = downloader.download_and_process(year=2020, month=1)
# Creates: co2_2020_01.nc (~200KB, single time step)
```

### 7.3 CLI Usage:
```bash
# Download entire CO2 dataset (recommended)
python -m src.stac_cli noaa --output ./output

# Download specific month (backwards compatibility)
python -m src.stac_cli noaa --year 2020 --month 1 --output ./output
```

## 8. Validation & Quality Control

### 8.1 Data Validation (lines 198-202):
```python
# Reasonable atmospheric CO2 range check
if min_co2 < 250 or max_co2 > 500:
    self.logger.warning(f"CO2 values outside expected range: {min_co2:.1f} - {max_co2:.1f} ppm")
```

### 8.2 Expected Value Ranges:
- **Historical (1958-1990)**: 315-355 ppm
- **Modern (1990-2020)**: 355-415 ppm
- **Current (2020-2025)**: 415-425 ppm
- **Future projections**: 425-500 ppm (depending on emission scenarios)

## 9. Testing & Verification

### 9.1 Key Test Points:
1. **Download functionality**: Verify HTTPS connection to NOAA GML
2. **Data parsing**: Ensure text format parsing handles missing values (-99.99)
3. **Spatial replication**: Verify global grid creation matches MATLAB logic
4. **CBF integration**: Confirm CO2_2 variable appears in final CBF files
5. **Value ranges**: Validate CO2 concentrations are within expected ranges

### 9.2 Integration Tests:
```python
# Test complete pipeline:
# 1. Download → 2. Parse → 3. Spatially replicate → 4. Integrate → 5. CBF output
```

This documentation provides the complete lifecycle understanding needed to maintain and improve the CO2 variable processing pipeline in the CARDAMOM preprocessing system.