"""
Comprehensive NetCDF Infrastructure for CARDAMOM Preprocessing

This module provides complete NetCDF file creation and management capabilities
that replicate the exact structure and metadata from MATLAB CARDAMOM templates.
It consolidates all NetCDF functionality with proper dimension management,
data variable handling, and metadata management.
"""

import numpy as np
import netCDF4
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path


class CARDAMOMNetCDFWriter:
    """
    Comprehensive class for creating CARDAMOM-compliant NetCDF files.

    Reproduces exact structure and metadata from MATLAB templates.
    Consolidates all NetCDF functionality into a single, unified system.
    """

    def __init__(self, template_type: str = "3D", compression: bool = True):
        """
        Initialize NetCDF writer with specified template and compression settings.

        Args:
            template_type: Type of NetCDF template ("2D" or "3D")
            compression: Whether to apply compression to data variables
        """
        self.template_type = template_type
        self.compression = compression
        self.global_attributes = self._setup_global_attributes()

        # Initialize component managers
        self.dimension_manager = DimensionManager()
        self.data_variable_manager = DataVariableManager(compression)
        self.metadata_manager = MetadataManager()

    def write_2d_dataset(self, data_dict: Dict[str, Any]) -> None:
        """
        Write 2D dataset (spatial only).

        Equivalent to MATLAB CARDAMOM_MAPS_WRITE_2D_DATASET.

        Args:
            data_dict: Dictionary containing:
                - 'filename': Output NetCDF file path
                - 'x': Longitude coordinates array
                - 'y': Latitude coordinates array
                - 'data': Data array with shape (lat, lon) or (lat, lon, variables)
                - 'info': Variable information (dict or list of dicts)
                - 'Attributes': Optional global attributes dictionary
        """
        self._validate_2d_input(data_dict)

        with netCDF4.Dataset(data_dict['filename'], 'w') as nc_dataset:
            # Create dimensions and coordinate variables
            self.dimension_manager.create_2d_dimensions(nc_dataset, data_dict)
            self.dimension_manager.create_coordinate_variables(nc_dataset, data_dict, spatial_only=True)

            # Create data variables
            self.data_variable_manager.create_2d_data_variables(nc_dataset, data_dict)

            # Add metadata
            self.metadata_manager.add_global_metadata(nc_dataset, data_dict)

        print(f"Done with {data_dict['filename']}")

    def write_3d_dataset(self, data_dict: Dict[str, Any]) -> None:
        """
        Write 3D dataset (spatial + temporal).

        Equivalent to MATLAB CARDAMOM_MAPS_WRITE_3D_DATASET.

        Args:
            data_dict: Dictionary containing:
                - 'filename': Output NetCDF file path
                - 'x': Longitude coordinates array
                - 'y': Latitude coordinates array
                - 't': Time coordinates array
                - 'timeunits': Time units string
                - 'data': Data array with shape (lat, lon, time) or (lat, lon, time, variables)
                - 'info': Variable information (dict or list of dicts)
                - 'Attributes': Optional global attributes dictionary
        """
        self._validate_3d_input(data_dict)

        with netCDF4.Dataset(data_dict['filename'], 'w') as nc_dataset:
            # Create dimensions and coordinate variables
            self.dimension_manager.create_3d_dimensions(nc_dataset, data_dict)
            self.dimension_manager.create_coordinate_variables(nc_dataset, data_dict, spatial_only=False)

            # Create data variables
            self.data_variable_manager.create_3d_data_variables(nc_dataset, data_dict)

            # Add metadata
            self.metadata_manager.add_global_metadata(nc_dataset, data_dict)

        print(f"Done with {data_dict['filename']}")

    def write_template_2d(self, grid_info: Dict[str, Any], filename: str) -> None:
        """
        Create 2D template file.

        Equivalent to MATLAB CARDAMOM_MAPS_WRITE_2D_TEMPLATE.

        Args:
            grid_info: Dictionary with grid information
            filename: Output template file path
        """
        template_generator = TemplateGenerator(self)
        template_generator.create_2d_template(filename, grid_info)

    def write_template_3d(self, grid_info: Dict[str, Any], filename: str) -> None:
        """
        Create 3D template file.

        Equivalent to MATLAB CARDAMOM_MAPS_WRITE_3D_TEMPLATE.

        Args:
            grid_info: Dictionary with grid information
            filename: Output template file path
        """
        template_generator = TemplateGenerator(self)
        template_generator.create_3d_template(filename, grid_info)

    def _validate_2d_input(self, data_dict: Dict[str, Any]) -> None:
        """Validate input for 2D dataset creation"""
        required_fields = ['filename', 'x', 'y', 'data', 'info']
        for field in required_fields:
            if field not in data_dict:
                raise ValueError(f"Required field '{field}' missing from data_dict")

        # Validate data dimensions
        expected_shape = (len(data_dict['y']), len(data_dict['x']))
        if data_dict['data'].shape[:2] != expected_shape:
            raise ValueError(f"Data shape {data_dict['data'].shape} does not match grid {expected_shape}")

    def _validate_3d_input(self, data_dict: Dict[str, Any]) -> None:
        """Validate input for 3D dataset creation"""
        self._validate_2d_input(data_dict)

        if 't' not in data_dict:
            raise ValueError("Time coordinate 't' required for 3D datasets")

        # For 3D data, check if we have a time dimension
        if data_dict['data'].ndim >= 3:
            expected_time_length = len(data_dict['t'])
            if data_dict['data'].shape[2] != expected_time_length:
                raise ValueError(f"Data time dimension {data_dict['data'].shape[2]} does not match time coordinate length {expected_time_length}")

    def _setup_global_attributes(self) -> Dict[str, Any]:
        """Setup default global attributes for CARDAMOM files"""
        return {
            'title': 'CARDAMOM Preprocessed Dataset',
            'institution': 'NASA Jet Propulsion Laboratory',
            'source': 'CARDAMOM Data Assimilation System',
            'conventions': 'CF-1.6',
            'history': f'Created by CARDAMOM preprocessor on {datetime.now().isoformat()}'
        }


class DimensionManager:
    """Manage NetCDF dimensions and coordinate variables"""

    def __init__(self):
        """Initialize dimension manager with standard dimension definitions"""
        self.standard_dimensions = {
            'longitude': {'var_name': 'longitude', 'units': 'degrees_east'},
            'latitude': {'var_name': 'latitude', 'units': 'degrees_north'},
            'time': {'var_name': 'time', 'units': 'months since Dec of previous year'}
        }

    def create_2d_dimensions(self, nc_dataset: netCDF4.Dataset, data_dict: Dict[str, Any]) -> None:
        """
        Create dimensions for 2D datasets.

        Matches MATLAB dimension creation logic.

        Args:
            nc_dataset: NetCDF dataset object
            data_dict: Data dictionary with coordinate information
        """
        data_shape = data_dict['data'].shape
        nc_dataset.createDimension('latitude', data_shape[0])
        nc_dataset.createDimension('longitude', data_shape[1])

    def create_3d_dimensions(self, nc_dataset: netCDF4.Dataset, data_dict: Dict[str, Any]) -> None:
        """
        Create dimensions for 3D datasets.

        Matches MATLAB dimension creation logic.

        Args:
            nc_dataset: NetCDF dataset object
            data_dict: Data dictionary with coordinate information
        """
        data_shape = data_dict['data'].shape
        nc_dataset.createDimension('latitude', data_shape[0])
        nc_dataset.createDimension('longitude', data_shape[1])
        nc_dataset.createDimension('time', data_shape[2])

    def create_coordinate_variables(self, nc_dataset: netCDF4.Dataset,
                                  data_dict: Dict[str, Any],
                                  spatial_only: bool = True) -> None:
        """
        Create coordinate variables with proper attributes.

        Replicates MATLAB coordinate variable creation.

        Args:
            nc_dataset: NetCDF dataset object
            data_dict: Data dictionary with coordinate arrays
            spatial_only: If True, only create spatial coordinates (lat/lon)
        """
        # Create longitude variable
        lon_var = nc_dataset.createVariable('longitude', 'f4', ('longitude',))
        lon_var[:] = data_dict['x']
        lon_var.units = 'degrees_east'
        lon_var.long_name = 'longitude'
        lon_var.standard_name = 'longitude'

        # Create latitude variable
        lat_var = nc_dataset.createVariable('latitude', 'f4', ('latitude',))
        lat_var[:] = data_dict['y']
        lat_var.units = 'degrees_north'
        lat_var.long_name = 'latitude'
        lat_var.standard_name = 'latitude'

        # Create time variable if 3D
        if not spatial_only and 't' in data_dict:
            time_var = nc_dataset.createVariable('time', 'f4', ('time',))
            time_var[:] = data_dict['t']
            time_units = data_dict.get('timeunits', 'months since Dec of previous year')
            time_var.units = time_units
            time_var.long_name = 'time'
            time_var.standard_name = 'time'


class DataVariableManager:
    """Manage creation of data variables with proper attributes"""

    def __init__(self, compression: bool = True):
        """
        Initialize data variable manager.

        Args:
            compression: Whether to apply compression to data variables
        """
        self.compression = compression
        self.fill_value = -9999.0

    def create_2d_data_variables(self, nc_dataset: netCDF4.Dataset, data_dict: Dict[str, Any]) -> None:
        """
        Create 2D data variables.

        Handles both single and multiple variables per file.

        Args:
            nc_dataset: NetCDF dataset object
            data_dict: Data dictionary with data array and variable info
        """
        data_array = data_dict['data']
        info_list = data_dict['info']

        if data_array.ndim == 2:
            # Single variable
            self._create_single_2d_variable(nc_dataset, data_array, info_list)
        elif data_array.ndim == 3:
            # Multiple variables (third dimension is variables)
            for var_idx in range(data_array.shape[2]):
                var_data = data_array[:, :, var_idx]
                var_info = info_list[var_idx] if isinstance(info_list, list) else info_list
                self._create_single_2d_variable(nc_dataset, var_data, var_info, var_idx)

    def create_3d_data_variables(self, nc_dataset: netCDF4.Dataset, data_dict: Dict[str, Any]) -> None:
        """
        Create 3D data variables.

        Handles temporal dimension properly.

        Args:
            nc_dataset: NetCDF dataset object
            data_dict: Data dictionary with data array and variable info
        """
        data_array = data_dict['data']
        info_list = data_dict['info']

        if data_array.ndim == 3:
            # Single variable with time dimension
            self._create_single_3d_variable(nc_dataset, data_array, info_list)
        elif data_array.ndim == 4:
            # Multiple variables with time dimension (fourth dimension is variables)
            for var_idx in range(data_array.shape[3]):
                var_data = data_array[:, :, :, var_idx]
                var_info = info_list[var_idx] if isinstance(info_list, list) else info_list
                self._create_single_3d_variable(nc_dataset, var_data, var_info, var_idx)

    def _create_single_2d_variable(self, nc_dataset: netCDF4.Dataset,
                                 data_array: np.ndarray,
                                 var_info: Dict[str, Any],
                                 var_idx: Optional[int] = None) -> None:
        """Create a single 2D data variable"""
        var_name = self._get_variable_name(var_info, var_idx)

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

        var[:] = data_array
        self._add_variable_attributes(var, var_info)

    def _create_single_3d_variable(self, nc_dataset: netCDF4.Dataset,
                                 data_array: np.ndarray,
                                 var_info: Dict[str, Any],
                                 var_idx: Optional[int] = None) -> None:
        """Create a single 3D data variable"""
        var_name = self._get_variable_name(var_info, var_idx)

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

        var[:] = data_array
        self._add_variable_attributes(var, var_info)

    def _get_variable_name(self, var_info: Dict[str, Any], var_idx: Optional[int] = None) -> str:
        """Get variable name from info dictionary"""
        if isinstance(var_info, dict) and 'name' in var_info:
            return var_info['name']
        elif var_idx is not None:
            return f"variable_{var_idx}"
        else:
            return "data"

    def _add_variable_attributes(self, var: netCDF4.Variable, var_info: Dict[str, Any]) -> None:
        """Add attributes to data variable"""
        if isinstance(var_info, dict):
            # Add standard attributes
            if 'units' in var_info:
                var.units = var_info['units']
            if 'long_name' in var_info:
                var.long_name = var_info['long_name']
            if 'standard_name' in var_info:
                var.standard_name = var_info['standard_name']
            if 'description' in var_info:
                var.description = var_info['description']

            # Add any additional attributes
            for key, value in var_info.items():
                if key not in ['name', 'units', 'long_name', 'standard_name', 'description']:
                    setattr(var, key, value)


class MetadataManager:
    """Manage global and variable metadata for CARDAMOM NetCDF files"""

    def __init__(self):
        """Initialize metadata manager"""
        self.default_global_attrs = self._setup_default_global_attributes()
        self.cardamom_version = "CARDAMOM preprocessor v1.0"

    def add_global_metadata(self, nc_dataset: netCDF4.Dataset, data_dict: Dict[str, Any]) -> None:
        """
        Add global attributes to NetCDF dataset.

        Replicates MATLAB global attribute structure.

        Args:
            nc_dataset: NetCDF dataset object
            data_dict: Data dictionary with optional custom attributes
        """
        # Add default global attributes
        for attr_name, attr_value in self.default_global_attrs.items():
            setattr(nc_dataset, attr_name, attr_value)

        # Add CARDAMOM-specific attributes
        nc_dataset.description = self._get_description(data_dict)
        nc_dataset.details = self._get_details(data_dict)
        nc_dataset.creation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        nc_dataset.version = self.cardamom_version
        nc_dataset.contact = "CARDAMOM Development Team"

        # Add custom attributes if provided
        if 'Attributes' in data_dict:
            self._add_custom_attributes(nc_dataset, data_dict['Attributes'])

    def _setup_default_global_attributes(self) -> Dict[str, Any]:
        """Setup default global attributes for CARDAMOM files"""
        return {
            'title': 'CARDAMOM Preprocessed Dataset',
            'institution': 'NASA Jet Propulsion Laboratory',
            'source': 'CARDAMOM Data Assimilation System',
            'conventions': 'CF-1.6',
            'history': f'Created by CARDAMOM preprocessor on {datetime.now().isoformat()}'
        }

    def _get_description(self, data_dict: Dict[str, Any]) -> str:
        """Generate description from data dictionary"""
        if 'Attributes' in data_dict and 'description' in data_dict['Attributes']:
            return data_dict['Attributes']['description']
        else:
            return "CARDAMOM preprocessed meteorological dataset"

    def _get_details(self, data_dict: Dict[str, Any]) -> str:
        """Generate details from data dictionary"""
        if 'Attributes' in data_dict and 'details' in data_dict['Attributes']:
            return data_dict['Attributes']['details']
        else:
            return "Generated by CARDAMOM preprocessing system"

    def _add_custom_attributes(self, nc_dataset: netCDF4.Dataset, custom_attrs: Dict[str, Any]) -> None:
        """Add custom global attributes"""
        for attr_name, attr_value in custom_attrs.items():
            setattr(nc_dataset, attr_name, attr_value)


class TemplateGenerator:
    """Generate template NetCDF files matching MATLAB templates"""

    def __init__(self, writer: CARDAMOMNetCDFWriter):
        """
        Initialize template generator.

        Args:
            writer: CARDAMOMNetCDFWriter instance to use for file creation
        """
        self.writer = writer

    def create_2d_template(self, output_path: str, grid_info: Dict[str, Any]) -> None:
        """
        Create 2D template file.

        Equivalent to MATLAB CARDAMOM_MAPS_WRITE_2D_TEMPLATE.

        Args:
            output_path: Path for output template file
            grid_info: Dictionary with grid information including resolution
        """
        grid_resolution = grid_info.get('resolution', 0.5)

        if grid_resolution == 0.5:
            lon_coords = np.arange(-179.75, 180, 0.5)  # 720 points
            lat_coords = np.arange(89.75, -90, -0.5)   # 360 points
        elif grid_resolution == 0.25:
            lon_coords = np.arange(-179.75, 180, 0.25)  # 1440 points
            lat_coords = np.arange(89.75, -90, -0.25)   # 720 points
        else:
            raise ValueError(f"Grid resolution {grid_resolution} not supported")

        template_data = np.full((len(lat_coords), len(lon_coords)), np.nan)

        data_dict = {
            'filename': output_path,
            'x': lon_coords,
            'y': lat_coords,
            'data': template_data,
            'info': {'name': 'data', 'units': 'unitless', 'long_name': 'template data'},
            'Attributes': {'variable_info': 'TEMPLATE DATASET'}
        }

        self.writer.write_2d_dataset(data_dict)

    def create_3d_template(self, output_path: str, grid_info: Dict[str, Any]) -> None:
        """
        Create 3D template file.

        Equivalent to MATLAB CARDAMOM_MAPS_WRITE_3D_TEMPLATE.

        Args:
            output_path: Path for output template file
            grid_info: Dictionary with grid information including resolution
        """
        grid_resolution = grid_info.get('resolution', 0.5)

        if grid_resolution == 0.5:
            lon_coords = np.arange(-179.75, 180, 0.5)  # 720 points
            lat_coords = np.arange(89.75, -90, -0.5)   # 360 points
        elif grid_resolution == 0.25:
            lon_coords = np.arange(-179.75, 180, 0.25)  # 1440 points
            lat_coords = np.arange(89.75, -90, -0.25)   # 720 points
        else:
            raise ValueError(f"Grid resolution {grid_resolution} not supported")

        time_coords = np.arange(1, 13)  # 12 months
        template_data = np.full((len(lat_coords), len(lon_coords), len(time_coords)), np.nan)

        data_dict = {
            'filename': output_path,
            'x': lon_coords,
            'y': lat_coords,
            't': time_coords,
            'timeunits': 'months since Dec of previous year',
            'data': template_data,
            'info': {'name': 'data', 'units': 'unitless', 'long_name': 'template data'},
            'Attributes': {'variable_info': 'TEMPLATE DATASET'}
        }

        self.writer.write_3d_dataset(data_dict)