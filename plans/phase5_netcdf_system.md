# Phase 5: NetCDF Template and Writing System

## Overview
Create a comprehensive NetCDF writing system that produces CARDAMOM-compliant files matching the exact structure and metadata of the original MATLAB implementation. Based on template functions from `CARDAMOM_MAPS_05deg_DATASETS_JUL24.m`.

## 5.1 Core NetCDF Writing Infrastructure (`netcdf_writer.py`)

### Main NetCDF Writer Class
```python
class CARDAMOMNetCDFWriter:
    """
    Primary class for creating CARDAMOM-compliant NetCDF files.
    Reproduces exact structure and metadata from MATLAB templates.
    """

    def __init__(self, template_type="3D", compression=True):
        self.template_type = template_type
        self.compression = compression
        self.global_attributes = self._setup_global_attributes()
        self.dimension_registry = self._setup_dimension_registry()

    def write_2d_dataset(self, data_dict):
        """
        Write 2D dataset (spatial only).
        Equivalent to MATLAB CARDAMOM_MAPS_WRITE_2D_DATASET.
        """
        self._validate_2d_input(data_dict)

        with netCDF4.Dataset(data_dict['filename'], 'w') as nc:
            self._create_2d_dimensions(nc, data_dict)
            self._create_coordinate_variables(nc, data_dict, spatial_only=True)
            self._create_data_variables_2d(nc, data_dict)
            self._add_global_metadata(nc, data_dict)

        print(f"Done with {data_dict['filename']}")

    def write_3d_dataset(self, data_dict):
        """
        Write 3D dataset (spatial + temporal).
        Equivalent to MATLAB CARDAMOM_MAPS_WRITE_3D_DATASET.
        """
        self._validate_3d_input(data_dict)

        with netCDF4.Dataset(data_dict['filename'], 'w') as nc:
            self._create_3d_dimensions(nc, data_dict)
            self._create_coordinate_variables(nc, data_dict, spatial_only=False)
            self._create_data_variables_3d(nc, data_dict)
            self._add_global_metadata(nc, data_dict)

        print(f"Done with {data_dict['filename']}")

    def write_template_2d(self, grid_info, filename):
        """
        Create 2D template file.
        Equivalent to MATLAB CARDAMOM_MAPS_WRITE_2D_TEMPLATE.
        """

    def write_template_3d(self, grid_info, filename):
        """
        Create 3D template file.
        Equivalent to MATLAB CARDAMOM_MAPS_WRITE_3D_TEMPLATE.
        """
```

### Data Structure Validation
```python
def _validate_2d_input(self, data_dict):
    """Validate input for 2D dataset creation"""
    required_fields = ['filename', 'x', 'y', 'data', 'info']
    for field in required_fields:
        if field not in data_dict:
            raise ValueError(f"Required field '{field}' missing from data_dict")

    # Validate data dimensions
    expected_shape = (len(data_dict['y']), len(data_dict['x']))
    if data_dict['data'].shape[:2] != expected_shape:
        raise ValueError(f"Data shape {data_dict['data'].shape} does not match grid {expected_shape}")

def _validate_3d_input(self, data_dict):
    """Validate input for 3D dataset creation"""
    self._validate_2d_input(data_dict)

    if 't' not in data_dict:
        raise ValueError("Time coordinate 't' required for 3D datasets")

    expected_shape = (len(data_dict['y']), len(data_dict['x']), len(data_dict['t']))
    if data_dict['data'].shape != expected_shape:
        raise ValueError(f"Data shape {data_dict['data'].shape} does not match expected {expected_shape}")
```

## 5.2 Dimension and Coordinate Management (`coordinate_manager.py`)

### Dimension Creation
```python
class DimensionManager:
    """Manage NetCDF dimensions and coordinate variables"""

    def __init__(self):
        self.standard_dimensions = {
            'longitude': {'var_name': 'longitude', 'units': 'degrees_east'},
            'latitude': {'var_name': 'latitude', 'units': 'degrees_north'},
            'time': {'var_name': 'time', 'units': 'months since Dec of previous year'}
        }

    def create_2d_dimensions(self, nc_dataset, data_dict):
        """
        Create dimensions for 2D datasets.
        Matches MATLAB dimension creation logic.
        """
        data_shape = data_dict['data'].shape

        # Create dimensions
        nc_dataset.createDimension('latitude', data_shape[0])
        nc_dataset.createDimension('longitude', data_shape[1])

    def create_3d_dimensions(self, nc_dataset, data_dict):
        """
        Create dimensions for 3D datasets.
        Matches MATLAB dimension creation logic.
        """
        data_shape = data_dict['data'].shape

        # Create dimensions
        nc_dataset.createDimension('latitude', data_shape[0])
        nc_dataset.createDimension('longitude', data_shape[1])
        nc_dataset.createDimension('time', data_shape[2])

    def create_coordinate_variables(self, nc_dataset, data_dict, spatial_only=True):
        """
        Create coordinate variables with proper attributes.
        Replicates MATLAB coordinate variable creation.
        """

        # Create longitude variable
        lon_var = nc_dataset.createVariable('longitude', 'f4', ('longitude',))
        lon_var[:] = data_dict['x']
        lon_var.units = 'degrees'

        # Create latitude variable
        lat_var = nc_dataset.createVariable('latitude', 'f4', ('latitude',))
        lat_var[:] = data_dict['y']
        lat_var.units = 'degrees'

        # Create time variable if 3D
        if not spatial_only:
            time_var = nc_dataset.createVariable('time', 'f4', ('time',))
            time_var[:] = data_dict['t']

            # Set time units (with fallback)
            time_units = data_dict.get('timeunits', 'months since Dec of previous year')
            time_var.units = time_units
```

### Coordinate System Utilities
```python
def validate_coordinate_system(self, x_coords, y_coords):
    """Validate coordinate arrays are properly formatted"""

    # Check monotonicity
    if not np.all(np.diff(x_coords) > 0):
        raise ValueError("Longitude coordinates must be strictly increasing")

    if not np.all(np.diff(y_coords) < 0):  # Latitude typically decreasing (N to S)
        print("Warning: Latitude coordinates are not decreasing (N to S)")

def create_coordinate_attributes(self, coord_type):
    """Create standard coordinate attributes"""

    attribute_map = {
        'longitude': {
            'units': 'degrees_east',
            'long_name': 'longitude',
            'standard_name': 'longitude'
        },
        'latitude': {
            'units': 'degrees_north',
            'long_name': 'latitude',
            'standard_name': 'latitude'
        },
        'time': {
            'long_name': 'time',
            'standard_name': 'time'
        }
    }

    return attribute_map.get(coord_type, {})
```

## 5.3 Data Variable Management (`data_variable_manager.py`)

### Data Variable Creation
```python
class DataVariableManager:
    """Manage creation of data variables with proper attributes"""

    def __init__(self, compression=True):
        self.compression = compression
        self.fill_value = -9999.0

    def create_2d_data_variables(self, nc_dataset, data_dict):
        """
        Create 2D data variables.
        Handles both single and multiple variables per file.
        """

        data_array = data_dict['data']
        info_list = data_dict['info']

        # Handle single vs multiple variables
        if data_array.ndim == 2:
            # Single variable
            self._create_single_2d_variable(nc_dataset, data_array, info_list)
        elif data_array.ndim == 3:
            # Multiple variables (4th dimension for different data types)
            for var_idx in range(data_array.shape[2]):
                var_data = data_array[:, :, var_idx]
                var_info = info_list[var_idx] if isinstance(info_list, list) else info_list
                self._create_single_2d_variable(nc_dataset, var_data, var_info, var_idx)

    def create_3d_data_variables(self, nc_dataset, data_dict):
        """
        Create 3D data variables.
        Handles temporal dimension properly.
        """

        data_array = data_dict['data']
        info_list = data_dict['info']

        # Handle single vs multiple variables
        if data_array.ndim == 3:
            # Single variable
            self._create_single_3d_variable(nc_dataset, data_array, info_list)
        elif data_array.ndim == 4:
            # Multiple variables
            for var_idx in range(data_array.shape[3]):
                var_data = data_array[:, :, :, var_idx]
                var_info = info_list[var_idx] if isinstance(info_list, list) else info_list
                self._create_single_3d_variable(nc_dataset, var_data, var_info, var_idx)

    def _create_single_2d_variable(self, nc_dataset, data_array, var_info, var_idx=None):
        """Create a single 2D data variable"""

        # Determine variable name
        var_name = self._get_variable_name(var_info, var_idx)

        # Create variable with compression if enabled
        if self.compression:
            var = nc_dataset.createVariable(
                var_name, 'f4', ('latitude', 'longitude'),
                zlib=True, complevel=6, fill_value=self.fill_value
            )
        else:
            var = nc_dataset.createVariable(
                var_name, 'f4', ('latitude', 'longitude'),
                fill_value=self.fill_value
            )

        # Set data
        var[:] = data_array

        # Add variable attributes
        self._add_variable_attributes(var, var_info)

    def _create_single_3d_variable(self, nc_dataset, data_array, var_info, var_idx=None):
        """Create a single 3D data variable"""

        # Determine variable name
        var_name = self._get_variable_name(var_info, var_idx)

        # Create variable with compression if enabled
        if self.compression:
            var = nc_dataset.createVariable(
                var_name, 'f4', ('latitude', 'longitude', 'time'),
                zlib=True, complevel=6, fill_value=self.fill_value
            )
        else:
            var = nc_dataset.createVariable(
                var_name, 'f4', ('latitude', 'longitude', 'time'),
                fill_value=self.fill_value
            )

        # Set data
        var[:] = data_array

        # Add variable attributes
        self._add_variable_attributes(var, var_info)
```

### Variable Attributes
```python
def _add_variable_attributes(self, nc_variable, var_info):
    """Add attributes to data variable"""

    # Units (required)
    if hasattr(var_info, 'units'):
        nc_variable.units = var_info.units
    elif isinstance(var_info, dict) and 'units' in var_info:
        nc_variable.units = var_info['units']

    # Variable info/description
    if hasattr(var_info, 'variable_info'):
        nc_variable.variable_info = var_info.variable_info
    elif isinstance(var_info, dict) and 'variable_info' in var_info:
        nc_variable.variable_info = var_info['variable_info']

    # Long name
    if hasattr(var_info, 'long_name'):
        nc_variable.long_name = var_info.long_name
    elif hasattr(var_info, 'name'):
        nc_variable.long_name = var_info.name

def _get_variable_name(self, var_info, var_idx=None):
    """Determine appropriate variable name"""

    if hasattr(var_info, 'name'):
        return var_info.name
    elif isinstance(var_info, dict) and 'name' in var_info:
        return var_info['name']
    elif var_idx is not None:
        return f"data_{var_idx}"
    else:
        return "data"
```

## 5.4 Metadata Management (`metadata_manager.py`)

### Global Attributes Manager
```python
class MetadataManager:
    """Manage global and variable metadata for CARDAMOM NetCDF files"""

    def __init__(self):
        self.default_global_attrs = self._setup_default_global_attributes()
        self.cardamom_version = "CARDAMOM preprocessor v1.0"

    def add_global_metadata(self, nc_dataset, data_dict):
        """
        Add global attributes to NetCDF dataset.
        Replicates MATLAB global attribute structure.
        """

        # Required CARDAMOM attributes
        nc_dataset.description = self._get_description(data_dict)
        nc_dataset.details = self._get_details(data_dict)

        # Creation and version info
        nc_dataset.creation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        nc_dataset.version = self.cardamom_version

        # Contact information
        nc_dataset.contact = "CARDAMOM Development Team"

        # Data-specific attributes
        if 'Attributes' in data_dict:
            self._add_custom_attributes(nc_dataset, data_dict['Attributes'])

    def _setup_default_global_attributes(self):
        """Setup default global attributes for CARDAMOM files"""
        return {
            'title': 'CARDAMOM Preprocessed Dataset',
            'institution': 'NASA Jet Propulsion Laboratory',
            'source': 'CARDAMOM Data Assimilation System',
            'conventions': 'CF-1.6',
            'history': f'Created by CARDAMOM preprocessor on {datetime.now().isoformat()}'
        }

    def _get_description(self, data_dict):
        """Extract or generate description from data dictionary"""

        if 'Attributes' in data_dict and hasattr(data_dict['Attributes'], 'description'):
            return data_dict['Attributes'].description
        elif 'Attributes' in data_dict and 'description' in data_dict['Attributes']:
            return data_dict['Attributes']['description']
        else:
            return 'CARDAMOM-compliant dataset'

    def _get_details(self, data_dict):
        """Extract or generate details from data dictionary"""

        if 'Attributes' in data_dict and hasattr(data_dict['Attributes'], 'details'):
            return data_dict['Attributes'].details
        elif 'Attributes' in data_dict and 'details' in data_dict['Attributes']:
            return data_dict['Attributes']['details']
        else:
            return 'CARDAMOM-compliant datasets are CARDAMOM-ready data'

    def _add_custom_attributes(self, nc_dataset, attributes):
        """Add custom attributes from Attributes dictionary"""

        # Handle both object attributes and dictionary
        if hasattr(attributes, '__dict__'):
            attr_dict = attributes.__dict__
        else:
            attr_dict = attributes

        for key, value in attr_dict.items():
            if key not in ['description', 'details']:  # These are handled separately
                setattr(nc_dataset, key, str(value))
```

### Variable-Specific Metadata
```python
def create_variable_metadata(self, variable_name, data_source, processing_info=None):
    """Create metadata for specific variables"""

    metadata_templates = {
        'T2M_MIN': {
            'long_name': 'Monthly average daily minimum 2m temperature',
            'standard_name': 'air_temperature',
            'cell_methods': 'time: minimum within days time: mean over days'
        },
        'T2M_MAX': {
            'long_name': 'Monthly average daily maximum 2m temperature',
            'standard_name': 'air_temperature',
            'cell_methods': 'time: maximum within days time: mean over days'
        },
        'SSRD': {
            'long_name': 'Monthly average shortwave downward radiation',
            'standard_name': 'surface_downwelling_shortwave_flux_in_air',
            'cell_methods': 'time: mean'
        },
        'VPD': {
            'long_name': 'Vapor Pressure Deficit',
            'cell_methods': 'time: mean'
        }
    }

    return metadata_templates.get(variable_name, {})

def add_processing_history(self, nc_dataset, processing_steps):
    """Add processing history to global attributes"""

    history_string = f"Created by CARDAMOM preprocessor on {datetime.now().isoformat()}\n"

    for step in processing_steps:
        history_string += f"  - {step}\n"

    nc_dataset.history = history_string
```

## 5.5 Template File Generators (`template_generators.py`)

### Template Creation Functions
```python
class TemplateGenerator:
    """Generate template NetCDF files matching MATLAB templates"""

    def __init__(self):
        self.writer = CARDAMOMNetCDFWriter()

    def create_2d_template(self, output_path, grid_resolution=0.5):
        """
        Create 2D template file.
        Equivalent to MATLAB CARDAMOM_MAPS_WRITE_2D_TEMPLATE.
        """

        # Create coordinate arrays
        if grid_resolution == 0.5:
            lon_coords = np.arange(-179.75, 180, 0.5)  # 720 points
            lat_coords = np.arange(89.75, -90, -0.5)   # 360 points
        else:
            raise ValueError(f"Grid resolution {grid_resolution} not supported")

        # Create template data (NaN filled)
        template_data = np.full((len(lat_coords), len(lon_coords)), np.nan)

        # Create data dictionary
        data_dict = {
            'filename': output_path,
            'x': lon_coords,
            'y': lat_coords,
            'data': template_data,
            'info': {'name': 'data', 'units': 'unitless'},
            'Attributes': {
                'variable_info': 'TEMPLATE DATASET'
            }
        }

        self.writer.write_2d_dataset(data_dict)

    def create_3d_template(self, output_path, grid_resolution=0.5):
        """
        Create 3D template file.
        Equivalent to MATLAB CARDAMOM_MAPS_WRITE_3D_TEMPLATE.
        """

        # Create coordinate arrays
        if grid_resolution == 0.5:
            lon_coords = np.arange(-179.75, 180, 0.5)  # 720 points
            lat_coords = np.arange(89.75, -90, -0.5)   # 360 points
        else:
            raise ValueError(f"Grid resolution {grid_resolution} not supported")

        time_coords = np.arange(1, 13)  # 12 months

        # Create template data (NaN filled)
        template_data = np.full((len(lat_coords), len(lon_coords), len(time_coords)), np.nan)

        # Create data dictionary
        data_dict = {
            'filename': output_path,
            'x': lon_coords,
            'y': lat_coords,
            't': time_coords,
            'timeunits': 'months since Dec of previous year',
            'data': template_data,
            'info': {'name': 'data', 'units': 'unitless'},
            'Attributes': {
                'variable_info': 'TEMPLATE DATASET'
            }
        }

        self.writer.write_3d_dataset(data_dict)
```

### Variable-Specific Templates
```python
def create_era5_temperature_template(self, output_path, variable_type='min'):
    """Create template for ERA5 temperature variables"""

    template_info = {
        'min': {
            'name': 'T2M_MIN',
            'units': 'deg C',
            'variable_info': 'ERA5 0.5 degree dataset: monthly average daily minimum temperature'
        },
        'max': {
            'name': 'T2M_MAX',
            'units': 'deg C',
            'variable_info': 'ERA5 0.5 degree dataset: monthly average daily maximum temperature'
        }
    }

    info = template_info[variable_type]

    # Use base 3D template with specific metadata
    self.create_3d_template(output_path)

    # Update with variable-specific metadata
    with netCDF4.Dataset(output_path, 'a') as nc:
        nc.variables['data'].setncattr('variable_info', info['variable_info'])
        nc.variables['data'].units = info['units']

def create_co2_template(self, output_path):
    """Create template for NOAA CO2 data"""

    # Base 3D template
    self.create_3d_template(output_path)

    # Update with CO2-specific metadata
    with netCDF4.Dataset(output_path, 'a') as nc:
        nc.variables['data'].setncattr('variable_info',
            'NOAA global mean surface atmospheric CO2 (not spatially resolved, mean replicated everywhere)')
        nc.variables['data'].units = 'CO2 [ppm]'
        nc.setncattr('source', 'ftp://aftp.cmdl.noaa.gov/products/trends/co2/co2_mm_gl.txt')
```

## 5.6 Data Format Converters (`format_converters.py`)

### MATLAB to Python Data Structure Conversion
```python
class MATLABDataConverter:
    """Convert MATLAB-style data structures to Python NetCDF format"""

    def __init__(self):
        self.required_fields = ['filename', 'x', 'y', 'data']
        self.optional_fields = ['t', 'timeunits', 'info', 'Attributes']

    def convert_matlab_data_structure(self, matlab_data):
        """
        Convert MATLAB DATA structure to Python dictionary.

        MATLAB structure fields:
        - DATA.filename
        - DATA.x, DATA.y, DATA.t
        - DATA.data
        - DATA.info(n).name, DATA.info(n).units
        - DATA.Attributes.description, etc.
        """

        python_data = {}

        # Required fields
        for field in self.required_fields:
            if hasattr(matlab_data, field):
                python_data[field] = getattr(matlab_data, field)
            else:
                raise ValueError(f"Required field '{field}' missing from MATLAB data")

        # Optional fields
        for field in self.optional_fields:
            if hasattr(matlab_data, field):
                python_data[field] = getattr(matlab_data, field)

        # Convert info structure
        if 'info' in python_data:
            python_data['info'] = self._convert_info_structure(python_data['info'])

        return python_data

    def _convert_info_structure(self, info_struct):
        """Convert MATLAB info structure to Python format"""

        if hasattr(info_struct, '__len__') and not isinstance(info_struct, str):
            # Multiple info entries
            converted_info = []
            for info_entry in info_struct:
                converted_info.append(self._convert_single_info(info_entry))
            return converted_info
        else:
            # Single info entry
            return self._convert_single_info(info_struct)

    def _convert_single_info(self, info_entry):
        """Convert single info entry"""

        if hasattr(info_entry, 'name') and hasattr(info_entry, 'units'):
            return {
                'name': info_entry.name,
                'units': info_entry.units
            }
        elif isinstance(info_entry, dict):
            return info_entry
        else:
            return {'name': 'data', 'units': 'unknown'}
```

### Unit Conversion Utilities
```python
def apply_unit_conversions(self, data_dict, variable_type):
    """Apply unit conversions as specified in MATLAB"""

    conversion_map = {
        'precipitation': {
            'scale_factor': 1e3,
            'source_units': 'm/s',
            'target_units': 'mm/day'
        },
        'temperature': {
            'offset': -273.15,  # K to C
            'source_units': 'K',
            'target_units': 'deg C'
        },
        'vpd': {
            'scale_factor': 10,  # Convert to hPa
            'source_units': 'Pa',
            'target_units': 'hPa'
        }
    }

    if variable_type in conversion_map:
        conversion = conversion_map[variable_type]

        if 'scale_factor' in conversion:
            data_dict['data'] *= conversion['scale_factor']

        if 'offset' in conversion:
            data_dict['data'] += conversion['offset']

        # Update units in info
        if isinstance(data_dict['info'], dict):
            data_dict['info']['units'] = conversion['target_units']
        elif hasattr(data_dict['info'], 'units'):
            data_dict['info'].units = conversion['target_units']

    return data_dict
```

## 5.7 Quality Control and Validation (`netcdf_validation.py`)

### File Validation
```python
class NetCDFValidator:
    """Validate NetCDF files for CARDAMOM compliance"""

    def __init__(self):
        self.cf_checker = None  # Could integrate CF compliance checker

    def validate_file_structure(self, filepath):
        """Validate basic NetCDF file structure"""

        with netCDF4.Dataset(filepath, 'r') as nc:
            # Check required dimensions
            required_dims = ['latitude', 'longitude']
            for dim in required_dims:
                if dim not in nc.dimensions:
                    raise ValueError(f"Required dimension '{dim}' missing")

            # Check coordinate variables
            for coord in required_dims:
                if coord not in nc.variables:
                    raise ValueError(f"Coordinate variable '{coord}' missing")

            # Validate coordinate attributes
            self._validate_coordinate_attributes(nc)

    def validate_cardamom_compliance(self, filepath):
        """Validate CARDAMOM-specific requirements"""

        with netCDF4.Dataset(filepath, 'r') as nc:
            # Check required global attributes
            required_attrs = ['description', 'creation_date']
            for attr in required_attrs:
                if not hasattr(nc, attr):
                    print(f"Warning: Missing global attribute '{attr}'")

            # Check data variable attributes
            for var_name in nc.variables:
                if var_name not in ['latitude', 'longitude', 'time']:
                    self._validate_data_variable(nc.variables[var_name])

    def _validate_coordinate_attributes(self, nc_dataset):
        """Validate coordinate variable attributes"""

        # Longitude checks
        if 'longitude' in nc_dataset.variables:
            lon_var = nc_dataset.variables['longitude']
            if not hasattr(lon_var, 'units'):
                raise ValueError("Longitude variable missing 'units' attribute")

        # Latitude checks
        if 'latitude' in nc_dataset.variables:
            lat_var = nc_dataset.variables['latitude']
            if not hasattr(lat_var, 'units'):
                raise ValueError("Latitude variable missing 'units' attribute")

    def _validate_data_variable(self, data_variable):
        """Validate data variable attributes"""

        if not hasattr(data_variable, 'units'):
            print(f"Warning: Data variable '{data_variable.name}' missing 'units' attribute")
```

### Data Quality Checks
```python
def check_data_ranges(self, filepath, variable_name, expected_range=None):
    """Check if data values are within expected ranges"""

    with netCDF4.Dataset(filepath, 'r') as nc:
        if variable_name in nc.variables:
            data = nc.variables[variable_name][:]

            # Basic checks
            if np.all(np.isnan(data)):
                print(f"Warning: All values are NaN for {variable_name}")

            if expected_range:
                min_val, max_val = expected_range
                if np.any(data < min_val) or np.any(data > max_val):
                    print(f"Warning: {variable_name} values outside expected range [{min_val}, {max_val}]")

def compare_with_matlab_output(self, python_file, matlab_file, tolerance=1e-6):
    """Compare Python-generated file with MATLAB reference"""

    with netCDF4.Dataset(python_file, 'r') as py_nc:
        with netCDF4.Dataset(matlab_file, 'r') as mat_nc:
            # Compare dimensions
            for dim_name in py_nc.dimensions:
                if dim_name in mat_nc.dimensions:
                    if len(py_nc.dimensions[dim_name]) != len(mat_nc.dimensions[dim_name]):
                        raise ValueError(f"Dimension '{dim_name}' size mismatch")

            # Compare data variables
            for var_name in py_nc.variables:
                if var_name in mat_nc.variables:
                    py_data = py_nc.variables[var_name][:]
                    mat_data = mat_nc.variables[var_name][:]

                    if not np.allclose(py_data, mat_data, rtol=tolerance, equal_nan=True):
                        raise ValueError(f"Data mismatch in variable '{var_name}'")
```

## 5.8 Testing Framework

### Test Structure
```
tests/netcdf/
├── test_netcdf_writer.py
├── test_coordinate_manager.py
├── test_data_variable_manager.py
├── test_metadata_manager.py
├── test_template_generators.py
├── test_format_converters.py
├── test_netcdf_validation.py
└── fixtures/
    ├── sample_matlab_output/
    ├── template_files/
    └── test_data/
```

### Integration Tests
```python
def test_complete_workflow(self):
    """Test complete NetCDF creation workflow"""

def test_matlab_compatibility(self):
    """Test compatibility with MATLAB-generated files"""

def test_cf_compliance(self):
    """Test Climate and Forecast (CF) convention compliance"""
```

## 5.9 Success Criteria

### Functional Requirements
- [ ] Exactly reproduce MATLAB NetCDF file structure
- [ ] Support both 2D and 3D dataset creation
- [ ] Handle multiple variables per file
- [ ] Generate proper coordinate systems

### Metadata Requirements
- [ ] Include all CARDAMOM-required global attributes
- [ ] Maintain variable-specific metadata
- [ ] Support custom attribute addition
- [ ] Preserve processing history

### Quality Requirements
- [ ] Pass CF convention compliance checks
- [ ] Validate against MATLAB reference files
- [ ] Handle edge cases and error conditions
- [ ] Provide comprehensive logging and validation

### Performance Requirements
- [ ] Efficient writing of large datasets
- [ ] Optional compression support
- [ ] Memory-efficient processing
- [ ] Progress tracking for large files