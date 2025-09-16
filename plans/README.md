# CARDAMOM MATLAB to Python Migration Plans

## Overview

This directory contains detailed implementation plans for converting the MATLAB CARDAMOM preprocessing system to Python. The migration is organized into 8 phases, each with specific deliverables and success criteria.

## Migration Scope

### Source MATLAB Files
1. **CARDAMOM_MAPS_05deg_DATASETS_JUL24.m** - Main data processing and NetCDF creation
2. **CARDAMOM_MAPS_READ_GFED_NOV24.m** - GFED burned area data reader with gap-filling
3. **PROJSCRIPT_DIURNAL_CMS_C_FLUXES_AUG25.m** - Diurnal flux processing for CONUS

### Target Python System
A comprehensive, modular preprocessing pipeline that:
- Provides scientifically equivalent functionality to the original MATLAB system
- Integrates with existing ECMWF downloader infrastructure
- Supports NASA MAAP platform deployment
- Provides enhanced CLI and individual component operations

## Phase Structure

### Phase 1: Core Data Processing Framework
**File**: `phase1_core_framework.md`

**Deliverables**:
- `CARDAMOMProcessor` main orchestration class
- Coordinate system management for multiple grids (0.25°, 0.5°, GeosChem)
- NetCDF infrastructure with CARDAMOM-compliant templates
- Scientific utility functions (VPD calculation, unit conversions)
- Configuration management and error handling

**Key Components**:
- `cardamom_preprocessor.py` - Main orchestration
- `coordinate_systems.py` - Grid management
- `netcdf_infrastructure.py` - NetCDF creation
- `scientific_utils.py` - Scientific calculations
- `config_manager.py` - Configuration handling

### Phase 2: Data Source-Specific Downloaders
**File**: `phase2_downloaders.md`

**Deliverables**:
- Enhanced ECMWF downloader with additional variables
- NOAA CO2 downloader with FTP access
- GFED downloader with HDF5 support and authentication
- MODIS land-sea mask downloader
- Unified downloader interface with retry logic and caching

**Key Components**:
- `ecmwf_downloader.py` - Enhanced ERA5 downloads
- `noaa_downloader.py` - NOAA CO2 data
- `gfed_downloader.py` - GFED fire data
- `modis_downloader.py` - Land-sea masks
- `base_downloader.py` - Common interface

### Phase 3: GFED Processing Module
**File**: `phase3_gfed_processor.md`

**Deliverables**:
- Comprehensive GFED data processor with gap-filling
- Multi-resolution support (0.25°, 0.5°, GeosChem 4×5)
- Climatology-based extrapolation for missing years (2017+)
- Emission factor calculations and species conversions
- Data quality validation and reporting

**Key Components**:
- `gfed_processor.py` - Main processing class
- `gap_filling.py` - Temporal gap-filling logic
- `resolution_converter.py` - Spatial aggregation
- `emission_factors.py` - Fire emission calculations
- `gfed_validation.py` - Quality control

### Phase 4: Diurnal Flux Processing
**File**: `phase4_diurnal_processor.md`

**Deliverables**:
- CONUS diurnal flux processing system
- CMS monthly flux loader with spatial interpolation
- ERA5 meteorological driver integration
- GFED diurnal fire pattern application
- Flux downscaling algorithms (GPP, REC, FIR, NEE, NBE)

**Key Components**:
- `diurnal_processor.py` - Main processing workflow
- `cms_flux_loader.py` - CMS data handling
- `met_driver_loader.py` - ERA5 diurnal data
- `diurnal_calculator.py` - Downscaling algorithms
- `diurnal_output_writers.py` - GeosChem format outputs

### Phase 5: NetCDF Template and Writing System
**File**: `phase5_netcdf_system.md`

**Status**: **CONSOLIDATED INTO PHASE 1** - NetCDF functionality has been integrated into Phase 1's core framework to eliminate duplication.

**Note**: Originally planned as a separate phase, the comprehensive NetCDF system has been consolidated into Phase 1 (`netcdf_infrastructure.py`) to address architectural concerns about redundant NetCDF planning. All NetCDF functionality is now available through Phase 1's `CARDAMOMNetCDFWriter` class and associated component managers.

**Integrated Components** (now in Phase 1):
- `CARDAMOMNetCDFWriter` - Main writing infrastructure
- `DimensionManager` - Dimension and coordinate handling
- `DataVariableManager` - Variable creation with compression
- `MetadataManager` - Comprehensive attribute management
- `TemplateGenerator` - 2D and 3D template creation

### Phase 6: Unified Processing Pipeline
**File**: `phase6_pipeline_manager.md`

**Deliverables**:
- Comprehensive pipeline orchestration system
- Component managers for downloaders, processors, and outputs
- Pipeline state management and resumability
- Error handling and recovery mechanisms
- Quality assurance and reporting system

**Key Components**:
- `pipeline_manager.py` - Main orchestrator
- `component_managers.py` - Module coordination
- `pipeline_state.py` - State tracking
- `error_handling.py` - Recovery strategies
- `qa_reporting.py` - Quality control

### Phase 7: Enhanced CLI Integration
**File**: `phase7_cli_integration.md`

**Deliverables**:
- Extended CLI maintaining backward compatibility
- Pipeline management commands
- Data source-specific commands
- Utility commands for validation and reporting
- Interactive configuration wizards

**Key Components**:
- `cardamom_cli.py` - Enhanced main CLI
- `pipeline_commands.py` - Pipeline interface
- `data_source_commands.py` - Individual source management
- `utility_commands.py` - Validation and utilities
- `interactive_cli.py` - Configuration wizards

### Phase 8: Scientific Functions Library
**File**: `phase8_scientific_utils.md`

**Deliverables**:
- Comprehensive atmospheric science calculations
- Carbon cycle and biogeochemistry functions
- Statistical and interpolation utilities
- Unit conversion and physical constants
- Data quality control and validation

**Key Components**:
- `atmospheric_science.py` - Water vapor, VPD, radiation
- `carbon_cycle.py` - Photosynthesis, respiration, fire emissions
- `statistics_utils.py` - Interpolation and time series analysis
- `units_constants.py` - Conversions and constants
- `quality_control.py` - Data validation

## Implementation Strategy

### Development Order *(Updated for Consolidated Architecture)*
1. **Phases 1 & 8** can be developed in parallel as they provide foundational infrastructure
   - **Phase 1** now includes consolidated NetCDF system (formerly Phase 5)
2. **Phase 2** follows Phase 1, extending existing downloader capabilities
3. **Phases 3 & 4** can be developed in parallel, both using Phase 1 infrastructure
4. **Phase 6** orchestrates all previous phases using unified configuration
5. **Phase 7** provides user interface to complete system

### Key Integration Points *(Updated)*
- **Phase 1** provides the foundation used by all other phases, including NetCDF infrastructure and unified configuration
- **Phase 2** downloaders feed into **Phase 6** pipeline management
- **Phases 3 & 4** processors use **Phase 1 NetCDF system** for output generation
- **Phase 6** coordinates **Phases 2-4** into unified workflows using **Phase 1 configuration system**
- **Phase 7** provides user access to **Phase 6** capabilities via **Phase 1 configuration system**
- **Phase 8** functions are used throughout **Phases 1-6**

### Testing Strategy
- Each phase includes comprehensive unit and integration tests
- Validation against MATLAB reference outputs
- Performance benchmarking for large datasets
- End-to-end pipeline testing with real data

## Success Criteria

### Functional Requirements
- [ ] Scientifically equivalent functionality to original MATLAB workflows
- [ ] Support for all MATLAB workflows (global monthly, CONUS diurnal)
- [ ] Enhanced capabilities beyond original MATLAB system
- [ ] Seamless integration with existing ECMWF infrastructure

### Performance Requirements
- [ ] Process multi-year global datasets efficiently
- [ ] Support parallel processing where applicable
- [ ] Memory-efficient handling of large arrays
- [ ] Reasonable processing times for operational use

### Quality Requirements
- [ ] Comprehensive error handling and logging
- [ ] Extensive test coverage (>90% for each phase)
- [ ] Clear documentation and user guides
- [ ] Robust data validation and quality control

### Integration Requirements
- [ ] Compatible with NASA MAAP platform
- [ ] Backward compatible CLI interface
- [ ] Configurable for different deployment scenarios
- [ ] Support for different data source combinations

## Dependencies and Environment

### Core Dependencies
```yaml
# From existing environment.yml, enhanced
dependencies:
  - python>=3.9
  - numpy>=1.20
  - scipy>=1.7
  - xarray>=0.20
  - netcdf4>=1.5
  - h5py>=3.0
  - pandas>=1.3
  - pyyaml>=5.4
  - requests>=2.25
  - cdsapi>=0.5
  - click>=8.0  # For enhanced CLI
  - tqdm>=4.60  # For progress bars
```

### Development Dependencies
```yaml
development:
  - pytest>=6.0
  - pytest-cov>=2.10
  - black>=21.0
  - flake8>=3.8
  - sphinx>=4.0
  - jupyter>=1.0
```

## Getting Started

1. **Review Plans**: Read through each phase plan to understand the scope and approach
2. **Set Up Environment**: Install dependencies and development tools
3. **Choose Starting Phase**: Begin with Phase 1 or Phase 8 depending on team expertise
4. **Iterative Development**: Implement, test, and validate each component
5. **Integration Testing**: Combine phases and test with domain expert validation
6. **Documentation**: Maintain comprehensive documentation throughout development

## Support and Collaboration

- Each phase plan includes detailed implementation specifications
- Success criteria provide clear validation targets
- Test frameworks ensure quality and compatibility
- Modular design supports parallel development by multiple developers

For questions about specific phases, refer to the individual plan files or the detailed specifications within each phase document.

## Implementation Guidelines for Scientists

This project prioritizes accessibility for scientists who may not be proficient in Python. All implementors must follow these guidelines to ensure the codebase serves the scientific community effectively.

### Core Implementation Principles

**1. Scientific Readability Over Programming Efficiency**
- Code should be immediately understandable by domain scientists
- Use descriptive variable names that include scientific context and units
- Prefer explicit, step-by-step implementations over compact Python idioms
- Include extensive comments explaining the scientific reasoning behind each operation

**2. Domain-Driven Code Organization**
- Structure code to mirror scientific workflows and thinking patterns
- Group related scientific operations together logically
- Use intermediate variables with clear scientific meaning
- Follow the natural progression from raw data to scientific products

**3. Comprehensive Scientific Documentation**
- Every function must include scientific context, not just technical specifications
- Document the physical meaning and typical ranges of all variables
- Include references to relevant scientific literature
- Provide examples using realistic atmospheric and carbon cycle values

### Implementation Standards by Phase

#### Phase 1: Core Framework - Scientific Foundation
```python
# Example: Clear scientific coordinate system definition
class CARDAMOMCoordinateGrid:
    """
    Scientific coordinate system for CARDAMOM global carbon cycle modeling.

    Implements the standard CARDAMOM spatial grid used for carbon flux analysis,
    matching the resolution and coverage required for ecosystem modeling at
    global scale while maintaining computational efficiency.
    """

    def __init__(self, resolution_degrees=0.5):
        # Define grid resolution in scientific terms
        self.spatial_resolution_degrees = resolution_degrees
        self.grid_cell_area_km2 = self._calculate_grid_cell_area()

        # Global coverage following CARDAMOM conventions
        # Note: 0.25° offset ensures grid centers align with CARDAMOM standards
        self.longitude_min_degrees = -179.75  # Western boundary
        self.longitude_max_degrees = 179.75   # Eastern boundary
        self.latitude_min_degrees = -89.75    # Southern boundary
        self.latitude_max_degrees = 89.75     # Northern boundary

        # Create coordinate arrays for scientific use
        self.longitude_centers_degrees = self._create_longitude_coordinates()
        self.latitude_centers_degrees = self._create_latitude_coordinates()
```

#### Phase 2: Downloaders - Scientific Data Access
```python
# Example: Clear scientific data download interface
def download_era5_monthly_meteorology(year, month, variables_list, spatial_bounds=None):
    """
    Download ERA5 meteorological data for CARDAMOM carbon cycle modeling.

    Retrieves essential atmospheric variables required for constraining
    ecosystem carbon fluxes in the CARDAMOM data assimilation system.

    Args:
        year (int): Year to download (1979-present)
        month (int): Month to download (1-12)
        variables_list (list): ERA5 variable names required for CARDAMOM:
            - '2m_temperature': Air temperature at 2m height (K)
            - '2m_dewpoint_temperature': Dewpoint temperature (K)
            - 'total_precipitation': Accumulated precipitation (m)
            - 'surface_solar_radiation_downwards': Downward solar radiation (J/m²)
        spatial_bounds (tuple, optional): (North, West, South, East) in degrees
            Default: Global coverage (-90, -180, 90, 180)

    Returns:
        dict: ERA5 data arrays with scientific metadata
            Each variable includes units, valid ranges, and quality flags

    Scientific Context:
        ERA5 provides the meteorological forcing data essential for CARDAMOM's
        process-based ecosystem modeling. Temperature drives photosynthesis and
        respiration rates, precipitation controls soil moisture and plant water
        availability, and solar radiation determines photosynthetic capacity.
    """
```

#### Phase 3 & 4: Processors - Scientific Calculations
```python
# Example: Scientific processing with clear methodology
def calculate_photosynthetically_active_radiation(solar_radiation_w_m2, cloud_fraction=None):
    """
    Calculate Photosynthetically Active Radiation (PAR) from solar radiation.

    PAR represents the portion of solar radiation (400-700 nm) that plants
    can use for photosynthesis. This calculation is essential for determining
    the light limitation on gross primary productivity in CARDAMOM.

    Scientific Method:
    PAR = Solar_radiation × PAR_fraction × Cloud_correction
    where PAR_fraction ≈ 0.45 for clear sky conditions

    Args:
        solar_radiation_w_m2 (array): Downward solar radiation in W/m²
            Typical range: 0-1400 W/m² (varies with latitude, season, clouds)
        cloud_fraction (array, optional): Cloud cover fraction (0-1)
            Used for improved PAR estimation under cloudy conditions

    Returns:
        array: PAR in µmol photons/m²/s
            Typical range: 0-2500 µmol/m²/s
            Light saturation for most plants: ~1500 µmol/m²/s

    References:
        Monteith, J.L. (1972). Solar radiation and productivity in tropical ecosystems.
        Journal of Applied Ecology, 9(3), 747-766.
    """

    # Standard conversion: ~45% of solar radiation is photosynthetically active
    par_fraction_clear_sky = 0.45

    # Convert W/m² to µmol photons/m²/s
    # Conversion factor: 1 W/m² PAR ≈ 4.57 µmol photons/m²/s
    watts_to_umol_conversion = 4.57

    # Calculate PAR under clear sky conditions
    par_clear_sky_umol_m2_s = (solar_radiation_w_m2 *
                              par_fraction_clear_sky *
                              watts_to_umol_conversion)

    # Apply cloud correction if available
    if cloud_fraction is not None:
        # Clouds reduce PAR but the relationship is non-linear
        cloud_transmission_factor = 1.0 - (cloud_fraction * 0.75)  # Clouds block ~75% when present
        par_actual_umol_m2_s = par_clear_sky_umol_m2_s * cloud_transmission_factor
    else:
        par_actual_umol_m2_s = par_clear_sky_umol_m2_s

    # Validate results are physically reasonable
    if np.any(par_actual_umol_m2_s < 0):
        raise ValueError("PAR cannot be negative. Check input solar radiation values.")

    if np.any(par_actual_umol_m2_s > 3000):
        raise ValueError("PAR values exceed physical maximum. Check input data units.")

    return par_actual_umol_m2_s
```

#### Phase 6: Pipeline Management - Scientific Workflow
```python
# Example: Scientific workflow management
class CARDAMOMProcessingPipeline:
    """
    Orchestrate CARDAMOM preprocessing following scientific best practices.

    Manages the complete data processing workflow from raw meteorological
    and observational data to CARDAMOM-ready input files, ensuring scientific
    consistency and traceability throughout the process.
    """

    def execute_global_monthly_workflow(self, processing_years, target_months=None):
        """
        Execute the complete global monthly preprocessing workflow.

        This method implements the standard CARDAMOM preprocessing sequence
        that transforms raw Earth observation data into the gridded, quality-
        controlled input files required for global carbon cycle analysis.

        Scientific Workflow Steps:
        1. Data Acquisition: Download meteorological and observational data
        2. Quality Control: Apply physical range checks and spatial consistency tests
        3. Unit Standardization: Convert all variables to CARDAMOM standard units
        4. Spatial Regridding: Harmonize all data to common 0.5° global grid
        5. Temporal Aggregation: Create monthly means with uncertainty estimates
        6. Scientific Validation: Check results against climatological expectations
        7. Output Generation: Create NetCDF files with complete metadata

        Args:
            processing_years (list): Years to process (e.g., [2020, 2021, 2022])
            target_months (list, optional): Months to process (1-12).
                Default: All months

        Returns:
            dict: Processing summary with scientific quality metrics
        """

        # Step 1: Initialize scientific processing environment
        print("Initializing CARDAMOM preprocessing environment")
        self._validate_scientific_configuration()
        self._setup_processing_directories()

        # Step 2: Execute processing for each year/month combination
        processing_summary = {'years_processed': [], 'scientific_validation': {}}

        for year in processing_years:
            print(f"\nProcessing year {year} for global carbon cycle analysis")

            months_to_process = target_months or list(range(1, 13))

            for month in months_to_process:
                print(f"  Processing {year}-{month:02d}: Meteorological and carbon data")

                # Execute scientific processing steps
                month_results = self._process_single_month_scientifically(year, month)

                # Validate scientific consistency
                validation_results = self._validate_monthly_outputs_scientifically(month_results)

                # Store results with scientific context
                processing_summary[f'{year}_{month:02d}'] = {
                    'data_coverage': month_results['spatial_coverage_percent'],
                    'quality_metrics': validation_results,
                    'scientific_flags': month_results['quality_flags']
                }

        # Generate final scientific assessment
        self._generate_scientific_processing_report(processing_summary)

        return processing_summary
```

### Error Handling for Scientists

**Provide scientifically meaningful error messages and recovery suggestions:**

```python
def validate_carbon_flux_physical_consistency(gpp_flux, respiration_flux, nee_flux):
    """
    Validate carbon flux data for physical and ecological consistency.

    Checks that carbon fluxes follow fundamental ecological principles
    and fall within observed ranges from flux tower measurements.
    """

    # Check sign conventions (CARDAMOM uses atmosphere perspective)
    if np.any(gpp_flux > 0):
        raise ValueError(
            "GPP values must be negative (carbon uptake from atmosphere). "
            "Found positive values suggesting incorrect sign convention. "
            "In CARDAMOM: GPP < 0 (sink), Respiration > 0 (source), NEE = Respiration - |GPP|"
        )

    # Check ecological consistency: |GPP| should generally exceed respiration
    gpp_magnitude = np.abs(gpp_flux)
    unrealistic_ratios = respiration_flux > (gpp_magnitude * 2.0)

    if np.any(unrealistic_ratios):
        problem_locations = np.where(unrealistic_ratios)
        raise ValueError(
            f"Respiration exceeds 2× GPP magnitude at {len(problem_locations[0])} locations. "
            f"This suggests ecosystem respiration is unrealistically high. "
            f"Typical ratios: Respiration/|GPP| = 0.5-1.2 for healthy ecosystems. "
            f"Check input data quality and units."
        )

    # Validate mass balance: NEE ≈ Respiration + GPP (accounting for sign conventions)
    calculated_nee = respiration_flux + gpp_flux  # GPP is negative
    nee_difference = np.abs(nee_flux - calculated_nee)

    if np.any(nee_difference > 1.0):  # 1 gC/m²/day tolerance
        raise ValueError(
            "Carbon flux mass balance violated: NEE ≠ Respiration + GPP. "
            f"Maximum difference: {np.max(nee_difference):.2f} gC/m²/day. "
            "This indicates inconsistent flux data or calculation errors. "
            "Verify flux components are from the same time period and spatial domain."
        )
```

### Testing with Scientific Validation

**Create tests that verify scientific correctness, not just code functionality:**

```python
def test_ecosystem_carbon_balance_conservation():
    """
    Test that carbon flux calculations conserve mass according to ecological principles.

    Validates that the fundamental equation of ecosystem carbon balance holds:
    NEE = Ecosystem_Respiration - GPP + Fire_Emissions
    """

    # Create realistic test data based on flux tower observations
    # GPP: Strong carbon sink during growing season
    test_gpp_gc_m2_day = -15.0  # Negative = carbon uptake

    # Ecosystem respiration: Always positive (carbon emission)
    test_respiration_gc_m2_day = 8.0

    # Fire emissions: Episodic carbon source
    test_fire_emissions_gc_m2_day = 2.0

    # Calculate NEE using scientific function
    calculated_nee = calculate_net_ecosystem_exchange(
        gpp_flux=test_gpp_gc_m2_day,
        ecosystem_respiration=test_respiration_gc_m2_day,
        fire_emissions=test_fire_emissions_gc_m2_day
    )

    # Expected NEE based on carbon balance equation
    expected_nee = test_respiration_gc_m2_day + test_fire_emissions_gc_m2_day + test_gpp_gc_m2_day
    # = 8.0 + 2.0 + (-15.0) = -5.0 gC/m²/day (net carbon sink)

    assert abs(calculated_nee - expected_nee) < 0.001, (
        f"Carbon mass balance violated. Expected NEE: {expected_nee}, "
        f"Calculated NEE: {calculated_nee}. "
        f"Check carbon flux calculation follows ecological sign conventions."
    )

    # Verify result is ecologically reasonable
    assert -50 < calculated_nee < 20, (
        f"NEE value {calculated_nee} gC/m²/day is outside typical ecosystem range. "
        f"Expected range: -50 to +20 gC/m²/day for terrestrial ecosystems."
    )

def test_vapor_pressure_deficit_meteorological_accuracy():
    """
    Test VPD calculation against published meteorological reference values.

    Uses standard atmospheric conditions from meteorological handbooks
    to verify calculation accuracy.
    """

    # Test case from meteorological literature
    # Standard conditions: 25°C, 60% relative humidity
    air_temperature_celsius = 25.0
    relative_humidity_percent = 60.0

    # Calculate dewpoint from RH (for testing purposes)
    saturation_pressure_25c = 31.69  # hPa at 25°C (from meteorological tables)
    actual_vapor_pressure = saturation_pressure_25c * (relative_humidity_percent / 100.0)

    # Convert to temperatures for function input
    air_temp_kelvin = air_temperature_celsius + 273.15
    dewpoint_kelvin = calculate_dewpoint_from_vapor_pressure(actual_vapor_pressure) + 273.15

    # Calculate VPD using our function
    calculated_vpd = calculate_vapor_pressure_deficit(air_temp_kelvin, dewpoint_kelvin)

    # Expected VPD from meteorological calculation
    expected_vpd_hpa = saturation_pressure_25c - actual_vapor_pressure  # = 31.69 - 19.01 = 12.68 hPa

    assert abs(calculated_vpd - expected_vpd_hpa) < 0.1, (
        f"VPD calculation differs from meteorological reference. "
        f"Expected: {expected_vpd_hpa:.2f} hPa, Calculated: {calculated_vpd:.2f} hPa. "
        f"Error exceeds 0.1 hPa tolerance for standard atmospheric conditions."
    )
```

### Documentation Templates

**Use these templates for consistent scientific documentation across all phases:**

```python
def template_scientific_function(input_data, scientific_parameters):
    """
    [One-line scientific description of what this function does]

    [2-3 sentences describing the scientific context and importance]

    Scientific Background:
    [Brief explanation of the underlying science, equations, or methodology]

    Args:
        input_data (type): [Description with units and typical range]
            Example: temperature_kelvin (array): Air temperature in Kelvin
                Typical range: 250-320 K (-23 to 47°C)
                Source: ERA5 reanalysis or meteorological observations
        scientific_parameters (type): [Description with scientific meaning]

    Returns:
        type: [Description with units, typical range, and scientific interpretation]
            Example: vapor_pressure_deficit_hpa (array): VPD in hectopascals
                Range: 0-60 hPa
                Interpretation: 0-10 hPa (low atmospheric demand),
                               10-30 hPa (moderate demand), >30 hPa (high demand)

    Raises:
        SpecificError: [When this occurs and what it means scientifically]

    References:
        [Relevant scientific literature or methodology sources]

    Example:
        >>> # Realistic example with scientific context
        >>> temp_k = np.array([298.15, 303.15])  # 25°C and 30°C
        >>> dewpoint_k = np.array([288.15, 293.15])  # 15°C and 20°C
        >>> vpd = template_scientific_function(temp_k, dewpoint_k)
        >>> # Expected: [11.7, 21.0] hPa for moderate atmospheric demand conditions
    """
```

### Getting Started for Scientists

**For scientists implementing any phase:**

1. **Read CLAUDE.md first** - Understand the complete coding standards
2. **Study existing examples** - Look at how similar scientific functions are implemented
3. **Focus on clarity** - Write code that you would want to read in 6 months
4. **Document scientifically** - Explain the science, not just the code
5. **Test with realistic data** - Use actual atmospheric/carbon values in tests
6. **Validate against literature** - Compare results with published values
7. **Ask for review** - Have domain experts review your scientific implementation

### Scientific Quality Checklist

Before submitting any implementation, verify:

- [ ] Variable names include scientific meaning and units where relevant
- [ ] Functions are documented with scientific context and literature references
- [ ] Error messages provide scientific guidance for troubleshooting
- [ ] Code structure follows the natural scientific workflow
- [ ] Calculations include physical validation and range checking
- [ ] Tests verify scientific correctness with realistic data
- [ ] Comments explain the scientific reasoning behind each step
- [ ] Results are interpreted in terms of physical/ecological meaning

**Remember: The goal is code that serves science, not code that impresses programmers.**