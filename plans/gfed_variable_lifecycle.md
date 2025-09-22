# GFED Variable Lifecycle Documentation

## Overview

This document traces the complete lifecycle of GFED (Global Fire Emissions Database) variables from their source through integration into CARDAMOM-compliant CBF files. GFED variables provide essential fire disturbance data for carbon cycle modeling, including burned area fractions and fire emissions by vegetation type.

## 1. Data Source & Acquisition

### 1.1 GFED Data Sources

#### Primary Source (GFED4.1s)
- **Base URL**: `https://www.globalfiredata.org/data_new/`
- **Data Format**: HDF5 files
- **Coverage**: 2001-2016 (complete), 2017+ (beta versions, incomplete)
- **Spatial Resolution**: 0.25° global grid
- **Temporal Resolution**: Daily, monthly aggregations
- **Data Types**: Burned area, fire emissions by vegetation type, diurnal patterns

#### Secondary Source (GFED5 Beta Monthly Emissions)
- **Base URL**: `https://surfdrive.surf.nl/files/index.php/s/VPMEYinPeHtWVxn?path=%2FGFED5_Beta_monthly_emissions`
- **Data Format**: NetCDF files
- **Coverage**: 2002+ (monthly emissions available)
- **File Pattern**: `GFED5_Beta_monthly_{YYYY}.nc`
- **Example URL**: `https://surfdrive.surf.nl/files/index.php/s/VPMEYinPeHtWVxn/download?path=%2FGFED5_Beta_monthly_emissions&files=GFED5_Beta_monthly_2002.nc`
- **Data Types**: Monthly fire emissions, updated GFED5 methodology

### 1.2 Version Handling Strategy
```python
# Updated version availability with GFED5 Beta
available_years = {
    'gfed4_historical': list(range(2001, 2017)),   # 2001-2016 (complete GFED4.1s)
    'gfed4_beta': list(range(2017, 2025)),         # 2017+ (GFED4.1s beta, incomplete)
    'gfed5_beta': list(range(2002, 2025))          # 2002+ (GFED5 Beta monthly, alternative source)
}

# Version-specific file handling
def get_gfed_source_info(year):
    if year <= 2016:
        return {
            'version': 'GFED4.1s',
            'source': 'globalfiredata.org',
            'format': 'HDF5',
            'filename': f'GFED4.1s_{year}.hdf5'
        }
    elif year <= 2021:
        return {
            'version': 'GFED4.1s_beta',
            'source': 'globalfiredata.org',
            'format': 'HDF5',
            'filename': f'GFED4.1s_{year}_beta.hdf5'
        }
    else:  # year >= 2017
        # Note: GFED5 Beta is available from 2002+ as alternative source
        # Can be used as primary for recent years or fallback for gap-filling
        return {
            'version': 'GFED4.1s_beta',
            'source': 'globalfiredata.org',
            'format': 'HDF5',
            'filename': f'GFED4.1s_{year}_beta.hdf5',
            'alternative': {
                'version': 'GFED5_beta',
                'source': 'surfdrive.surf.nl',
                'format': 'NetCDF',
                'filename': f'GFED5_Beta_monthly_{year}.nc'
            }
        }
```

### 1.3 GFED Vegetation Types & Emission Factors
```python
# Standard vegetation categories (src/gfed_downloader.py:46-53)
vegetation_types = [
    'SAVA',  # Savanna
    'BORF',  # Boreal Forest
    'TEMF',  # Temperate Forest
    'DEFO',  # Deforestation
    'PEAT',  # Peat
    'AGRI'   # Agriculture
]

# Emission factors (g species / kg dry matter) (src/gfed_diurnal_loader.py:73-89)
emission_factors = {
    'CO2': {'SAVA': 1686, 'BORF': 1489, 'TEMF': 1520, 'DEFO': 1643, 'PEAT': 1703, 'AGRI': 1585},
    'CO': {'SAVA': 65, 'BORF': 127, 'TEMF': 88, 'DEFO': 93, 'PEAT': 210, 'AGRI': 102},
    'CH4': {'SAVA': 1.9, 'BORF': 4.7, 'TEMF': 5.2, 'DEFO': 5.7, 'PEAT': 21.0, 'AGRI': 2.3},
    'C': {'SAVA': 0.45, 'BORF': 0.45, 'TEMF': 0.45, 'DEFO': 0.45, 'PEAT': 0.45, 'AGRI': 0.45}
}
```

**Scientific Context**: GFED provides vegetation-specific fire data essential for realistic ecosystem disturbance modeling in CARDAMOM, affecting carbon pools, vegetation dynamics, and ecosystem recovery.

### 1.4 Data Continuity Strategy

The evolution of GFED data sources creates **overlapping data availability** that provides both primary and alternative sources:

```
Timeline: 2001────────2016│2017────────────────────→
GFED4.1s: Complete        │Beta (incomplete/gaps)
GFED5:    ─────────2002───────────────────────────→ (alternative/complete)
Format:   HDF5            │HDF5 Beta    │NetCDF Monthly
Method:   Direct Use      │Gap-filling OR Use GFED5 Alternative
```

**Implementation Strategy**:
- **2001-2016**: Use complete GFED4.1s data (primary source)
- **2017+**: Either gap-fill GFED4.1s beta OR use GFED5 Beta as complete alternative
- **GFED5 Advantage**: Available from 2002+ as consistent NetCDF format with updated methodology

**Implementation Priority**: GFED5 Beta provides a **complete alternative data source** from 2002 onward, potentially eliminating the need for gap-filling in the transition period (2017+) and offering updated fire emission methodology.

## 2. GFED Variable Types in CARDAMOM

### 2.1 BURN_2 (Burned Area Variable)
- **Source Variable**: `burned_area` from GFED HDF5
- **CBF Variable Name**: `BURN_2` (meteorological forcing)
- **Units**: Fraction of grid cell burned per month
- **Usage**: Provides fire disturbance forcing for ecosystem carbon pool dynamics

### 2.2 Mean_FIR (Mean Fire Emissions)
- **Source Variable**: Fire carbon emissions (calculated from emission factors)
- **CBF Variable Name**: `Mean_FIR` (observational constraint)
- **Units**: gC/m²/day (carbon emission rate)
- **Usage**: Constrains fire emission estimates in CARDAMOM data assimilation

### 2.3 Diurnal Fire Patterns (CONUS)
- **Source**: 3-hourly GFED fire patterns
- **Usage**: CONUS hourly flux downscaling in diurnal processor
- **Spatial Domain**: CONUS region only
- **Application**: GeosChem atmospheric modeling inputs

## 3. Python Implementation: Download & Processing

### 3.1 GFED Downloader (`src/gfed_downloader.py`)

**Primary Class**: `GFEDDownloader(BaseDownloader)` (line 19)

#### Configuration:
```python
# Server and data configuration (lines 42-53)
self.base_url = "https://www.globalfiredata.org/data_new/"
self.vegetation_types = ['SAVA', 'BORF', 'TEMF', 'DEFO', 'PEAT', 'AGRI']

# Year availability (lines 62-67)
def _determine_available_years(self) -> Dict[str, List[int]]:
    return {
        'historical': list(range(2001, 2017)),  # Complete GFED4.1s
        'beta': list(range(2017, 2025))         # Beta versions
    }
```

#### Download Process:
```python
# Main download method (lines 82-91)
def download_yearly_files(self, years: List[int]) -> Dict[str, Any]:
    for year in years:
        file_url = self.get_file_url(year)              # Construct download URL
        filename = self.get_filename(year)              # GFED4.1s_{year}.hdf5
        # Download HDF5 file from GFED server
```

### 3.2 GFED Processor (`src/gfed_processor.py`)

**Primary Classes**: `GFEDData` (line 23), `GFEDProcessor` (main processing class)

#### Data Structure:
```python
# MATLAB-equivalent structure (lines 23-38)
@dataclass
class GFEDData:
    burned_area: np.ndarray  # GFED.BA in MATLAB
    fire_carbon: np.ndarray  # GFED.FireC in MATLAB
    year: np.ndarray         # GFED.year in MATLAB
    month: np.ndarray        # GFED.month in MATLAB
    resolution: str
    units: Dict[str, str]
```

#### NetCDF Export Process:
```python
# Monthly file creation (lines 89-113)
def _create_monthly_files(self, output_dir: str) -> List[str]:
    for i in range(len(self.month)):
        year = int(self.year[i])
        month = int(self.month[i])

        # Extract monthly data
        ba_monthly = self.burned_area[:, :, i:i+1]
        fc_monthly = self.fire_carbon[:, :, i:i+1]

        # Create CARDAMOM NetCDF files
        ba_filename = self._get_cardamom_filename("burned_area", year, month, output_dir)
        fc_filename = self._get_cardamom_filename("fire_carbon", year, month, output_dir)
```

#### CARDAMOM File Naming:
```python
# Filename convention (lines 296-299)
if month is not None:
    # Monthly: CARDAMOM_GFED_{variable}_{resolution}_{YYYYMM}.nc
    filename = f"CARDAMOM_GFED_{variable}_{self.resolution}_{year}{month:02d}.nc"
else:
    # Yearly: CARDAMOM_GFED_{variable}_{resolution}_{YYYY}.nc
```

### 3.3 GFED Diurnal Loader (`src/gfed_diurnal_loader.py`)

**Purpose**: CONUS diurnal fire pattern processing for hourly flux downscaling

#### Core Processing:
```python
# Load diurnal patterns (lines 93-133)
def load_diurnal_fields(self, month: int, year: int, target_region: Optional[List[float]] = None):
    # Determine file path with beta handling
    beta_suffix = '_beta' if year >= 2017 else ''                    # Line 113
    gfed_file = f'GFED4.1s_{year}{beta_suffix}.hdf5'               # Line 116

    # Load monthly GFED data
    gfed_data = self._load_monthly_gfed_data(gfed_file, month)      # Line 125

    # Extract regional CO2 diurnal patterns
    co2_diurnal = self._extract_regional_co2_diurnal(gfed_data, target_region)  # Line 128
```

#### CO2 Emission Calculation:
```python
# Calculate CO2 emissions from dry matter (lines 194-239)
def _calculate_co2_emissions_from_dry_matter(self, dry_matter: np.ndarray,
                                           vegetation_fractions: np.ndarray) -> np.ndarray:
    """
    Calculate CO2 emissions from GFED dry matter and emission factors.

    MATLAB Reference: CO2 emission calculation loops in MATLAB function
    """
    co2_factors = self.emission_factors['CO2']                      # Line 212

    # Apply vegetation-specific emission factors
    for veg_idx, veg_type in enumerate(self.vegetation_types):      # Line 218
        emission_factor = co2_factors[veg_type]
        # Calculate total CO2 emissions for this timestep            # Line 222
```

## 4. MATLAB Reference Implementation

### 4.1 Primary MATLAB File
- **Main Script**: `CARDAMOM_MAPS_READ_GFED_NOV24.m` (referenced in src/gfed_processor.py:4)
- **Function**: `CARDAMOM_MAPS_READ_GFED_NOV24(RES)` (line 7)

### 4.2 Expected MATLAB Processing Pattern
```matlab
% MATLAB Reference (inferred from Python implementation):
% 1. Load GFED HDF5 data
GFED = load_gfed_hdf5_data(year, month);

% 2. Extract burned area and dry matter
GFED.BA = extract_burned_area(GFED_data);
GFED.FireC = calculate_fire_carbon(GFED_data, emission_factors);

% 3. Gap-filling for missing years (2017+)
if year >= 2017
    GFED = apply_climatology_gap_filling(GFED, climatology_data);
end

% 4. Spatial regridding to CARDAMOM grid
GFED_regridded = regrid_to_cardamom_05deg(GFED);
```

### 4.3 Gap-Filling Strategy (Post-2016)
Based on Python implementation references:
- **Years 2001-2016**: Complete GFED4.1s data available
- **Years 2017+**: Beta versions with potential gaps
- **Gap-filling**: Use climatological means from 2001-2016 period
- **Implementation**: Python gap-filling processor (referenced but not yet implemented)

## 5. Integration into CARDAMOM Pipeline

### 5.1 CBF Met Processor Integration (`src/cbf_met_processor.py`)

#### Variable Mapping:
```python
# Required CBF variables (line 55)
MET_VARS_TO_KEEP = ['VPD', 'TOTAL_PREC', 'T2M_MIN', 'T2M_MAX',
                    'STRD', 'SSRD', 'SNOW', 'CO2_2', 'BURN_2', 'SKT']

# BURN_2 identified as external source (line 68)
# CO2_2 and BURN_2 from external sources
```

#### Integration Process (lines 669-677):
```python
# Primary integration pathway
if fire_data_dir:
    try:
        fire_dataset = self._load_fire_data(fire_data_dir, reference_coords)
        external_variables['BURN_2'] = fire_dataset                 # Line 669
        self.logger.info("Fire data integration completed")
    except Exception as e:
        # Fallback to zero values
        external_variables['BURN_2'] = self._create_zero_burned_area_dataset(reference_coords)  # Line 674

# No fire directory provided - use fallback
else:
    external_variables['BURN_2'] = self._create_zero_burned_area_dataset(reference_coords)      # Line 677
```

#### Fallback Mechanism (lines 710-722):
```python
def _create_zero_burned_area_dataset(self, reference_coords):
    """Create zero burned area dataset for missing fire data."""

    # Create zero-filled burned area data
    burned_area_data = xr.DataArray(
        data=np.zeros(time_shape),                                  # Zero fallback
        coords=reference_coords,
        attrs={
            'long_name': 'Burned Area Fraction',
            'units': 'fraction',
            'description': 'Zero burned area (fallback for missing fire data)',
            'source': 'Zero fallback'
        }
    )
    burned_area_data.name = 'BURN_2'                               # Line 720
    return xr.Dataset({'BURN_2': burned_area_data})                # Line 722
```

#### Fire Data Loading & Interpolation (lines 765-804):
```python
def _load_fire_data(self, fire_data_dir, reference_coords):
    """Load fire data from directory and interpolate to reference grid."""

    # Search patterns for fire files (line 782)
    fire_var_names = ['burned_area', 'BURNED_AREA', 'burned_fraction', 'ba']

    # Final variable assignment (lines 802-804)
    fire_interp.name = 'BURN_2'
    return xr.Dataset({'BURN_2': fire_interp})
```

### 5.2 ECMWF Integration (`src/ecmwf_downloader.py`)

#### Variable Name Mapping (lines 1273-1278):
```python
fire_var_mapping = {
    'burned_area': 'BURNED_AREA',
    'BURNED_AREA': 'BURNED_AREA',
    'burned_fraction': 'BURNED_AREA',
    'fire_emissions': 'Mean_FIR',
    'Mean_FIR': 'Mean_FIR',
    'co2_emissions': 'Mean_FIR'
}
```

#### Integration Logic (lines 1289-1303):
```python
# Handle different fire variable types
if cbf_var == 'BURNED_AREA':
    fire_interp = fire_var.interp(latitude=target_coords['latitude'],
                                 longitude=target_coords['longitude'])
    fire_interp.attrs.update({
        'long_name': 'Burned Area Fraction',
        'cbf_variable': 'BURNED_AREA'                               # Line 1295
    })
elif cbf_var == 'Mean_FIR':
    fire_interp = fire_var.interp(latitude=target_coords['latitude'],
                                 longitude=target_coords['longitude'])
    fire_interp.attrs.update({
        'long_name': 'Mean Fire Emissions',
        'cbf_variable': 'Mean_FIR'                                  # Line 1303
    })
```

## 6. CBF File Integration

### 6.1 Final CBF Variable Format (`matlab-migration/erens_cbf_code.py`)

#### Variable Requirements:
```python
# Expected in meteorological files
MET_RENAME_MAP = {
    'BURN_2': 'BURNED_AREA',      # Rename for CBF compatibility
    'CO2_2': 'CO2',
    # ... other variables
}

MET_VARS_TO_KEEP = ['VPD', 'TOTAL_PREC', 'T2M_MIN', 'T2M_MAX',
                    'STRD', 'SSRD', 'SNOWFALL', 'CO2', 'BURNED_AREA', 'SKT']

# Expected in observational files
OBS_VARS_TO_KEEP = ['SCF', 'GPP', 'ABGB', 'LAI', 'EWT', 'SOM', 'Mean_FIR']
```

#### CBF Processing:
```python
def set_forcing_variables(target_ds, source_met_ds, lat, lon, scaffold_ds, positive_vars):
    """Sets forcing variables including BURNED_AREA as ecosystem disturbance."""
    # BURNED_AREA becomes forcing variable for fire disturbance effects
    # Used in CARDAMOM ecosystem model for:
    # - Carbon pool disturbance
    # - Vegetation mortality
    # - Post-fire recovery dynamics

def set_observation_constraints(target_ds, source_obs_ds, lat, lon, scaffold_ds, constraint_vars):
    """Sets observational constraints including Mean_FIR for fire emission validation."""
    # Mean_FIR provides observational constraint for fire emissions
    # Used in CARDAMOM data assimilation for:
    # - Constraining fire emission estimates
    # - Validating model fire carbon fluxes
```

### 6.2 Scientific Usage in CARDAMOM

**BURNED_AREA (BURN_2)**:
- **Purpose**: Ecosystem disturbance forcing
- **Effects**: Reduces vegetation carbon pools, triggers post-fire recovery dynamics
- **Model Impact**: Affects GPP, respiration, and carbon allocation patterns

**Mean_FIR**:
- **Purpose**: Fire emission observational constraint
- **Effects**: Constrains fire carbon flux estimates in data assimilation
- **Model Impact**: Improves fire emission accuracy and uncertainty quantification

## 7. Data Flow Summary

```
GFED4.1s Database (HDF5)
        ↓
GFEDDownloader.download_yearly_files()   [src/gfed_downloader.py:82-91]
        ↓
    [GFED4.1s_{year}.hdf5]
        ↓
GFEDProcessor.process_gfed_data()        [src/gfed_processor.py:main]
        ↓
    [CARDAMOM_GFED_{variable}_{res}_{date}.nc]  (spatial regridding)
        ↓
CBFMetProcessor._load_fire_data()        [src/cbf_met_processor.py:765-804]
        ↓
External variable integration            [src/cbf_met_processor.py:669-677]
        ↓
    [AllMet05x05_LFmasked.nc]           (BURN_2 variable)
    [AlltsObs05x05_LFmasked.nc]         (Mean_FIR variable)
        ↓
CBF file creation                        [matlab-migration/erens_cbf_code.py]
        ↓
    [site*.cbf.nc]                      (BURNED_AREA + Mean_FIR variables)
        ↓
CARDAMOM Model Input                     (fire disturbance + constraints)
```

## 8. Diurnal Processing Branch (CONUS Only)

### 8.1 Separate Data Flow:
```
GFED4.1s HDF5 Files
        ↓
GFEDDiurnalLoader.load_diurnal_fields()  [src/gfed_diurnal_loader.py:93-133]
        ↓
3-hourly CO2 emission patterns          (CONUS region)
        ↓
DiurnalProcessor integration            [src/diurnal_processor.py]
        ↓
GeosChem hourly flux files             (atmospheric modeling)
```

### 8.2 Regional Focus:
- **Spatial Domain**: CONUS region only (60°N to 20°N, -130°W to -50°W)
- **Temporal Resolution**: 3-hourly patterns for fire timing
- **Application**: Atmospheric transport modeling inputs
- **Output Format**: GeosChem-compatible NetCDF files

## 9. Current Issues & Required Updates

### 9.1 Critical Issues:
1. **Multi-Source Integration**: Need to handle overlapping GFED data sources
   - **GFED4.1s Historical (2001-2016)**: Complete HDF5 data from globalfiredata.org
   - **GFED4.1s Beta (2017+)**: Incomplete HDF5 data, requires gap-filling
   - **GFED5 Beta (2002+)**: Complete NetCDF monthly emissions from surfdrive.surf.nl (alternative)

2. **Downloader Enhancement**: Current downloader only handles GFED4.1s from globalfiredata.org
   - **Missing**: GFED5 Beta downloader for surfdrive.surf.nl source
   - **Missing**: Format conversion from NetCDF to unified internal format

3. **Integration Status**: GFED processing exists but CBF integration incomplete
   - **BURN_2 Integration**: Partially implemented in CBF met processor
   - **Mean_FIR Integration**: Referenced but needs proper observational file integration

### 9.2 Required Updates:

#### Multi-Source Downloader:
```python
# Update src/gfed_downloader.py to handle multiple sources:
class GFEDDownloader(BaseDownloader):
    def download_data(self, years: List[int]) -> Dict[str, Any]:
        results = {}
        for year in years:
            source_info = self.get_gfed_source_info(year)

            if source_info['version'].startswith('GFED4'):
                # Existing HDF5 download from globalfiredata.org
                results[year] = self._download_gfed4_hdf5(year, source_info)
            elif source_info['version'] == 'GFED5_beta':
                # New NetCDF download from surfdrive.surf.nl
                results[year] = self._download_gfed5_netcdf(year, source_info)

        return results

    def _download_gfed5_netcdf(self, year: int, source_info: dict) -> dict:
        """Download GFED5 Beta monthly NetCDF from surfdrive.surf.nl"""
        base_url = "https://surfdrive.surf.nl/files/index.php/s/VPMEYinPeHtWVxn/download"
        download_url = f"{base_url}?path=%2FGFED5_Beta_monthly_emissions&files={source_info['filename']}"
        # Implement NetCDF download and format standardization
```

#### Gap-Filling Implementation:
```python
# Enhanced strategy: Gap-filling OR GFED5 alternative for 2017+
def get_complete_gfed_data(self, year: int) -> GFEDData:
    """Get complete GFED data using best available source for the year."""
    if year <= 2016:
        # Use complete GFED4.1s data
        return self._load_gfed4_data(year)
    else:  # year >= 2017
        # Option 1: Try GFED5 Beta as complete alternative (2002+)
        if year >= 2002:
            try:
                return self._load_gfed5_beta_data(year)
            except Exception:
                # Fallback to gap-filled GFED4.1s beta
                return self._load_gap_filled_gfed4_beta(year)
        else:
            # Only GFED4.1s available for 2001
            return self._load_gfed4_data(year)
```

#### Format Standardization:
```python
# Unified format converter for different GFED versions
def standardize_gfed_format(self, raw_data: Dict, version: str) -> GFEDData:
    """Convert different GFED formats to unified GFEDData structure."""
    if version.startswith('GFED4'):
        return self._convert_hdf5_to_gfed_data(raw_data)
    elif version == 'GFED5_beta':
        return self._convert_netcdf_to_gfed_data(raw_data)
```

## 10. Validation & Quality Control

### 10.1 Data Validation Checks:
```python
# Expected value ranges for validation:
# BURNED_AREA: 0.0 - 1.0 (fraction of grid cell)
# Mean_FIR: 0.0 - 50.0 gC/m²/day (typical fire emission rates)
# Vegetation fractions: Sum should equal 1.0 for each grid cell
```

### 10.2 Spatial Consistency:
- **Global Coverage**: GFED provides global fire data (all continents)
- **Hot Spots**: Africa (savanna), Australia (bushfires), North America (forest fires)
- **Seasonal Patterns**: Should reflect regional fire seasons and climate patterns

### 10.3 Temporal Consistency:
- **Interannual Variability**: Fire activity varies with climate (El Niño, droughts)
- **Missing Data Handling**: Years 2017+ may have incomplete coverage
- **Climatology Validation**: Gap-filled data should preserve realistic patterns

## 11. Testing & Verification

### 11.1 Key Test Points:
1. **HDF5 Reading**: Verify GFED HDF5 file extraction works correctly
2. **Emission Factors**: Validate vegetation-specific emission calculations
3. **Spatial Regridding**: Ensure 0.25° to 0.5° regridding preserves totals
4. **Gap Filling**: Test climatology application for post-2016 data
5. **CBF Integration**: Confirm BURN_2 and Mean_FIR appear in final CBF files
6. **Value Ranges**: Validate fire data falls within expected ranges

### 11.2 Integration Tests:
```python
# Test complete pipeline:
# 1. Download → 2. Process → 3. Regrid → 4. Gap-fill → 5. Integrate → 6. CBF output
```

## 12. MATLAB to Python Translation Notes

### 12.1 Key Equivalencies:
```matlab
% MATLAB → Python equivalents
GFED.BA        → gfed_data.burned_area
GFED.FireC     → gfed_data.fire_carbon
GFED.year      → gfed_data.year
GFED.month     → gfed_data.month

% File naming patterns
GFED4.1s_YYYY.hdf5 → Same in Python
CARDAMOM_GFED_*    → Python naming convention maintained
```

### 12.2 Processing Logic Preservation:
- **Emission Factor Application**: Maintained same vegetation-specific factors
- **Spatial Regridding**: Conservative regridding preserves fire totals
- **Gap-Filling Strategy**: Climatology approach matches MATLAB methodology
- **NetCDF Structure**: CARDAMOM-compliant format maintained

This documentation provides the complete lifecycle understanding needed to maintain, debug, and enhance the GFED variable processing pipeline in the CARDAMOM preprocessing system, complementing the CO2 lifecycle documentation with fire-specific processing details.