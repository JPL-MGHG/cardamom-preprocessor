# Phase 3: GFED Processing Module

## Overview
Create a comprehensive processor for GFED (Global Fire Emissions Database) data that handles burned area and fire emissions with gap-filling logic. Based on `CARDAMOM_MAPS_READ_GFED_NOV24.m`.

## 3.1 Core GFED Processor (`gfed_processor.py`)

### Main GFEDProcessor Class
```python
class GFEDProcessor:
    """
    Process GFED4.1s burned area and fire emissions data.
    Handles multi-year loading, gap-filling, and resolution conversion.

    Based on MATLAB function CARDAMOM_MAPS_READ_GFED_NOV24
    """

    def __init__(self, data_dir="./DATA/GFED4/", output_dir="./DATA/PROCESSED_GFED/"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.years_range = (2001, 2023)  # Default from MATLAB
        self.land_sea_mask = None
        self.emission_factors = self._setup_emission_factors()

    def process_multi_year_data(self, years=None, target_resolution="05deg"):
        """
        Main processing function that loads multiple years of GFED data.

        Args:
            years: List of years to process (default: 2001-2023)
            target_resolution: Output resolution ('05deg', '0.25deg', 'GC4x5')

        Returns:
            Dictionary with processed burned area and fire carbon data
        """

    def _setup_emission_factors(self):
        """Setup emission factors for different vegetation types and species"""
        return EmissionFactors()

    def load_land_sea_mask(self, resolution=0.25):
        """Load land-sea mask for data filtering"""
```

### Core Processing Methods
```python
def load_yearly_gfed_data(self, year):
    """
    Load GFED data for single year from HDF5 file.
    Equivalent to MATLAB GF4=DATASCRIPT_READ_GFED4_DATA_MAY16(yr)
    """

def extract_burned_area(self, gfed_data, year):
    """Extract burned area data from GFED HDF5 structure"""

def extract_fire_emissions(self, gfed_data, year, species='C'):
    """Extract fire emissions data (carbon as default species)"""

def apply_land_sea_mask_corrections(self, data, mask):
    """
    Apply land-sea mask corrections.
    Equivalent to MATLAB lines 24-27:
    - Set sea regions to NaN
    - Set land fire-free regions to 0
    """

def aggregate_to_resolution(self, data, target_resolution):
    """
    Aggregate data to target resolution.
    Handles different resolution conversions as in MATLAB switch statement.
    """
```

## 3.2 Gap-Filling and Temporal Continuity (`gap_filling.py`)

### Gap-Filling Strategy
```python
class GFEDGapFiller:
    """
    Handle gap-filling for missing GFED data using climatology.
    Based on MATLAB logic for handling 2017+ data gaps.
    """

    def __init__(self, reference_period=(2001, 2016)):
        self.reference_period = reference_period
        self.climatology_data = None

    def create_climatology(self, burned_area_data, fire_carbon_data):
        """
        Create climatological patterns from reference period.
        Equivalent to MATLAB lines 66-68: Using 2001-2016 BA/Emissions ratio
        """

    def fill_missing_years(self, data_dict, missing_years):
        """
        Fill missing years using climatology and available emissions.

        Implements MATLAB logic:
        BAextra = sum(BA[reference]) / sum(FireC[reference]) * FireC[missing]
        """

    def calculate_ba_emission_ratio(self, burned_area, fire_carbon):
        """Calculate burned area to emission ratio for climatology"""

    def apply_gap_filling(self, target_year, target_month, emissions_data):
        """Apply gap-filling for specific year/month"""
```

### Temporal Processing
```python
def create_temporal_arrays(self, start_year, end_year):
    """
    Create temporal coordinate arrays.
    Equivalent to MATLAB lines 75-77: year and month arrays
    """
    year_dates = np.arange(start_year + 1/24, end_year + 1, 1/12)
    years = np.floor(year_dates)
    months = np.round(np.mod(year_dates * 12 - 0.5, 12) + 1)
    return years, months

def ensure_temporal_continuity(self, data_dict):
    """Ensure no temporal gaps in the final dataset"""

def handle_missing_data_flags(self, data):
    """Convert missing data flags to appropriate NaN/zero values"""
```

## 3.3 Resolution Conversion (`resolution_converter.py`)

### Multi-Resolution Support
```python
class GFEDResolutionConverter:
    """
    Convert GFED data between different spatial resolutions.
    Supports: 0.25deg (native), 0.5deg, GeosChem 4x5
    """

    def __init__(self):
        self.supported_resolutions = ['0.25deg', '05deg', 'GC4x5']
        self.conversion_matrices = self._setup_conversion_matrices()

    def convert_resolution(self, data, source_res, target_res):
        """
        Convert data from source to target resolution.

        MATLAB equivalent for each case:
        - '05deg': Aggregate 0.25deg to 0.5deg using 2:2:end indexing
        - 'GC4x5': Use GEOSChem_regular_grid_to_GC() function
        """

    def aggregate_025_to_05_degree(self, data_025):
        """
        Aggregate 0.25° data to 0.5° resolution.
        Equivalent to MATLAB: data(2:2:end, 2:2:end, :)
        """

    def convert_to_geoschem_grid(self, data):
        """
        Convert to GeosChem 4°x5° grid.
        Equivalent to MATLAB: GEOSChem_regular_grid_to_GC(data)
        """

    def _setup_conversion_matrices(self):
        """Setup spatial conversion matrices for each resolution pair"""
```

### Spatial Aggregation
```python
def spatial_average_with_weights(self, data, weights=None):
    """Perform weighted spatial averaging"""

def handle_boundary_effects(self, data, method='nearest'):
    """Handle edge effects in spatial aggregation"""

def preserve_total_emissions(self, original_data, aggregated_data):
    """Ensure total emissions are preserved during aggregation"""
```

## 3.4 Emission Factors and Species Calculations (`emission_factors.py`)

### Emission Factors Class
```python
class EmissionFactors:
    """
    Manage emission factors for different vegetation types and chemical species.
    Based on MATLAB emission_factors() function.
    """

    def __init__(self):
        self.vegetation_types = ['SAVA', 'BORF', 'TEMF', 'DEFO', 'PEAT', 'AGRI']
        self.species = ['CO2', 'CO', 'CH4', 'C']
        self.factors = self._setup_emission_factors()
        self.uncertainties = self._setup_uncertainties()

    def _setup_emission_factors(self):
        """
        Setup emission factors matrix.
        From MATLAB: EF.SP matrix with species × vegetation types
        """
        return np.array([
            [1686, 1489, 1647, 1643, 1703, 1585],  # CO2
            [63, 127, 88, 93, 210, 102],           # CO
            [1.94, 5.96, 3.36, 5.07, 20.8, 5.82], # CH4
            [488.273, 464.989, 489.416, 491.751, 570.055, 480.352]  # C
        ])

    def _setup_uncertainties(self):
        """Setup uncertainty values for emission factors"""
        return np.array([
            [38, 121, 37, 58, 58, 100],          # CO2 uncertainty
            [17, 45, 19, 27, 68, 33],            # CO uncertainty
            [0.85, 3.14, 0.91, 1.98, 11.42, 3.56], # CH4 uncertainty
            [np.nan] * 6                          # C uncertainty (not provided)
        ])

    def get_emission_factor(self, species, vegetation_type):
        """Get emission factor for specific species and vegetation type"""

    def calculate_emissions(self, dry_matter, vegetation_fractions):
        """Calculate emissions from dry matter and vegetation fractions"""
```

### Species Conversion
```python
def convert_dry_matter_to_carbon(self, dry_matter_data, vegetation_fractions):
    """Convert dry matter combustion to carbon emissions"""

def calculate_species_emissions(self, carbon_emissions, species='CO2'):
    """Calculate specific species emissions from carbon"""

def apply_emission_uncertainties(self, emissions, species, vegetation_type):
    """Apply uncertainty estimates to emission calculations"""
```

## 3.5 Data Quality and Validation (`gfed_validation.py`)

### Quality Control
```python
class GFEDValidator:
    """Validate GFED data quality and consistency"""

    def validate_temporal_consistency(self, data_dict):
        """Check for temporal gaps or inconsistencies"""

    def validate_spatial_coverage(self, data_dict):
        """Ensure appropriate spatial coverage"""

    def validate_physical_ranges(self, burned_area, fire_emissions):
        """Check if values are within physically reasonable ranges"""

    def check_land_sea_consistency(self, data, land_sea_mask):
        """Verify data consistency with land-sea mask"""

    def generate_quality_report(self, data_dict, output_path):
        """Generate comprehensive quality assessment report"""
```

### Statistical Analysis
```python
def calculate_data_statistics(self, data_dict):
    """Calculate summary statistics for processed data"""

def compare_with_reference_dataset(self, processed_data, reference_data):
    """Compare processed data against reference dataset"""

def identify_outliers(self, data, method='iqr'):
    """Identify statistical outliers in the data"""
```

## 3.6 GFED Data Structures and I/O (`gfed_io.py`)

### Data Structure Definitions
```python
@dataclass
class GFEDDataset:
    """Structure for GFED dataset with metadata"""
    burned_area: np.ndarray
    fire_carbon: np.ndarray
    years: np.ndarray
    months: np.ndarray
    resolution: str
    units: Dict[str, str]
    metadata: Dict[str, Any]

    def to_netcdf(self, filename):
        """Export to NetCDF format"""

    def to_cardamom_format(self, output_dir):
        """Export in CARDAMOM-compliant format"""

    @classmethod
    def from_netcdf(cls, filename):
        """Load from NetCDF format"""
```

### I/O Operations
```python
def read_gfed_hdf5(self, filepath, extract_variables=None):
    """Read GFED HDF5 file and extract specified variables"""

def write_processed_gfed(self, data_dict, output_path, format='netcdf'):
    """Write processed GFED data to file"""

def load_cached_gfed(self, cache_key):
    """Load previously processed GFED data from cache"""

def save_to_cache(self, data_dict, cache_key):
    """Save processed data to cache for reuse"""
```

## 3.7 Integration with CARDAMOM Framework

### CARDAMOM Output Format
```python
def create_cardamom_burned_area_files(self, gfed_data, years, output_dir):
    """
    Create CARDAMOM-format burned area files.
    Equivalent to MATLAB file creation in main script lines 182-195
    """

def create_cardamom_metadata(self, processing_info):
    """
    Create CARDAMOM-compliant metadata.
    Includes version, source, processing details, contact information.
    """

def apply_cardamom_naming_convention(self, variable, year, resolution='05deg'):
    """Apply CARDAMOM file naming convention"""
```

### Integration Points
```python
def integrate_with_pipeline(self, pipeline_config):
    """Integration with main CARDAMOM processing pipeline"""

def provide_data_for_diurnal_processing(self, years, months, region=None):
    """Provide GFED data for diurnal flux processing"""

def create_summary_for_reports(self, processed_data):
    """Create summary information for processing reports"""
```

## 3.8 Configuration and Settings

### GFED Configuration
```yaml
# config/gfed_processing.yaml
gfed:
  data_directory: "./DATA/GFED4/"
  output_directory: "./DATA/PROCESSED_GFED/"

  processing:
    default_years: [2001, 2023]
    target_resolutions: ["05deg", "0.25deg", "GC4x5"]
    gap_filling:
      reference_period: [2001, 2016]
      method: "climatology_ratio"

  quality_control:
    check_physical_ranges: true
    validate_totals: true
    generate_reports: true

  output_formats:
    - "netcdf"
    - "cardamom_compliant"
```

### Processing Options
```python
class GFEDProcessingConfig:
    """Configuration class for GFED processing options"""

    def __init__(self, config_file=None):
        self.config = self._load_config(config_file)

    def get_processing_years(self):
        """Get years to process"""

    def get_gap_filling_settings(self):
        """Get gap-filling configuration"""

    def get_output_settings(self):
        """Get output format and directory settings"""
```

## 3.9 Testing and Validation

### Test Structure
```
tests/gfed/
├── test_gfed_processor.py
├── test_gap_filling.py
├── test_resolution_converter.py
├── test_emission_factors.py
├── test_gfed_validation.py
└── fixtures/
    ├── sample_gfed_2020.hdf5
    ├── sample_land_sea_mask.nc
    └── expected_outputs/
        ├── burned_area_05deg_2020.nc
        └── fire_carbon_05deg_2020.nc
```

### Validation Against MATLAB
```python
def test_against_matlab_output(self, matlab_output_dir, python_output_dir):
    """Compare Python output against MATLAB reference results"""

def validate_gap_filling_logic(self, test_years):
    """Validate gap-filling produces expected results"""

def test_resolution_conversions(self, test_data):
    """Test all resolution conversion methods"""
```

## 3.10 Success Criteria

### Functional Requirements
- [ ] Accurately reproduce MATLAB GFED processing results
- [ ] Handle all supported resolutions (0.25°, 0.5°, GeosChem)
- [ ] Implement robust gap-filling for missing years
- [ ] Process multi-year datasets efficiently

### Data Quality Requirements
- [ ] Preserve total emissions during spatial aggregation
- [ ] Maintain temporal continuity across all years
- [ ] Apply appropriate land-sea mask corrections
- [ ] Generate data within physically reasonable ranges

### Performance Requirements
- [ ] Process 20+ years of data within reasonable time
- [ ] Efficient memory usage for large spatial arrays
- [ ] Support for parallel processing where possible
- [ ] Provide progress tracking for long operations

### Integration Requirements
- [ ] Seamless integration with CARDAMOM pipeline
- [ ] Compatible output formats for downstream processing
- [ ] Consistent metadata and documentation
- [ ] Support for different processing workflows