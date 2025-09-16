"""
Coordinate System Management for CARDAMOM Preprocessing

This module provides classes for managing geographic coordinate systems and grids
used in CARDAMOM processing. It supports standard resolutions and grid definitions
that match MATLAB loadworldmesh function behavior.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from scipy.interpolate import RegularGridInterpolator


class CoordinateGrid:
    """
    Represents a standard geographic grid used in CARDAMOM processing.

    This class creates and manages geographic coordinate grids with specified
    resolution and bounds. It provides functionality for grid creation,
    regional subsetting, and data regridding operations.
    """

    def __init__(self, resolution: float, bounds: Optional[List[float]] = None):
        """
        Initialize coordinate grid with specified resolution and bounds.

        Args:
            resolution: Grid resolution in decimal degrees (e.g., 0.5, 0.25)
            bounds: Geographic bounds as [South, West, North, East] in decimal degrees.
                   If None, uses global bounds [-89.75, -179.75, 89.75, 179.75]
        """
        self.resolution = resolution
        self.bounds = bounds or [-89.75, -179.75, 89.75, 179.75]  # Global default

        # Validate bounds format
        if len(self.bounds) != 4:
            raise ValueError("Bounds must be [South, West, North, East]")

        south, west, north, east = self.bounds
        if south >= north or west >= east:
            raise ValueError("Invalid bounds: South >= North or West >= East")

        # Create coordinate arrays
        self.longitude_coordinates, self.latitude_coordinates = self._create_grid()

        # Grid dimensions
        self.num_longitude_points = len(self.longitude_coordinates)
        self.num_latitude_points = len(self.latitude_coordinates)

    def _create_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create longitude/latitude arrays matching MATLAB loadworldmesh function.

        Returns:
            Tuple of (longitude_array, latitude_array) in decimal degrees
        """
        south, west, north, east = self.bounds

        # Create longitude coordinates (West to East)
        longitude_coordinates = np.arange(west, east + self.resolution/2, self.resolution)

        # Create latitude coordinates (North to South, matching MATLAB convention)
        latitude_coordinates = np.arange(north, south - self.resolution/2, -self.resolution)

        return longitude_coordinates, latitude_coordinates

    def get_indices_for_region(self, region_bounds: List[float]) -> Tuple[slice, slice]:
        """
        Get array indices for a specific geographic region.

        Args:
            region_bounds: Region bounds as [South, West, North, East] in decimal degrees

        Returns:
            Tuple of (latitude_slice, longitude_slice) for array indexing
        """
        region_south, region_west, region_north, region_east = region_bounds

        # Find longitude indices
        lon_start_idx = np.argmin(np.abs(self.longitude_coordinates - region_west))
        lon_end_idx = np.argmin(np.abs(self.longitude_coordinates - region_east))

        # Find latitude indices (remember latitude goes North to South)
        lat_start_idx = np.argmin(np.abs(self.latitude_coordinates - region_north))
        lat_end_idx = np.argmin(np.abs(self.latitude_coordinates - region_south))

        # Create slices
        latitude_slice = slice(lat_start_idx, lat_end_idx + 1)
        longitude_slice = slice(lon_start_idx, lon_end_idx + 1)

        return latitude_slice, longitude_slice

    def get_regional_subset(self, region_bounds: List[float]) -> 'CoordinateGrid':
        """
        Create a new CoordinateGrid for a specific region.

        Args:
            region_bounds: Region bounds as [South, West, North, East] in decimal degrees

        Returns:
            New CoordinateGrid instance covering the specified region
        """
        return CoordinateGrid(resolution=self.resolution, bounds=region_bounds)

    def regrid_data(self, data: np.ndarray, target_grid: 'CoordinateGrid') -> np.ndarray:
        """
        Regrid data from this grid to target grid using interpolation.

        Args:
            data: Input data array with shape (latitude, longitude) or (latitude, longitude, time)
            target_grid: Target coordinate grid for regridding

        Returns:
            Regridded data array with target grid dimensions
        """
        # Validate input data dimensions
        if data.shape[0] != self.num_latitude_points or data.shape[1] != self.num_longitude_points:
            raise ValueError(f"Data shape {data.shape} does not match grid dimensions "
                           f"({self.num_latitude_points}, {self.num_longitude_points})")

        # Handle 2D and 3D data
        if data.ndim == 2:
            return self._regrid_2d_data(data, target_grid)
        elif data.ndim == 3:
            return self._regrid_3d_data(data, target_grid)
        else:
            raise ValueError(f"Unsupported data dimensions: {data.ndim}. Expected 2 or 3 dimensions.")

    def _regrid_2d_data(self, data: np.ndarray, target_grid: 'CoordinateGrid') -> np.ndarray:
        """Regrid 2D data (latitude, longitude)"""
        # Create interpolator
        interpolator = RegularGridInterpolator(
            (self.latitude_coordinates, self.longitude_coordinates),
            data,
            method='linear',
            bounds_error=False,
            fill_value=np.nan
        )

        # Create target coordinate meshgrid
        target_lon_mesh, target_lat_mesh = np.meshgrid(
            target_grid.longitude_coordinates,
            target_grid.latitude_coordinates,
            indexing='xy'
        )

        # Interpolate to target grid
        target_points = np.column_stack([
            target_lat_mesh.ravel(),
            target_lon_mesh.ravel()
        ])

        regridded_data = interpolator(target_points)
        regridded_data = regridded_data.reshape(
            target_grid.num_latitude_points,
            target_grid.num_longitude_points
        )

        return regridded_data

    def _regrid_3d_data(self, data: np.ndarray, target_grid: 'CoordinateGrid') -> np.ndarray:
        """Regrid 3D data (latitude, longitude, time)"""
        num_time_steps = data.shape[2]
        regridded_data = np.zeros((
            target_grid.num_latitude_points,
            target_grid.num_longitude_points,
            num_time_steps
        ))

        # Regrid each time step
        for time_idx in range(num_time_steps):
            regridded_data[:, :, time_idx] = self._regrid_2d_data(
                data[:, :, time_idx], target_grid
            )

        return regridded_data

    def create_template_array(self, fill_value: float = np.nan, num_time_steps: Optional[int] = None) -> np.ndarray:
        """
        Create template data array with grid dimensions.

        Args:
            fill_value: Value to fill the array with (default: NaN)
            num_time_steps: If specified, creates 3D array with time dimension

        Returns:
            Template array with appropriate dimensions
        """
        if num_time_steps is None:
            return np.full((self.num_latitude_points, self.num_longitude_points), fill_value)
        else:
            return np.full((self.num_latitude_points, self.num_longitude_points, num_time_steps), fill_value)

    def get_grid_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the grid.

        Returns:
            Dictionary containing grid metadata and dimensions
        """
        return {
            'resolution_degrees': self.resolution,
            'bounds': self.bounds,
            'longitude_range': [self.longitude_coordinates.min(), self.longitude_coordinates.max()],
            'latitude_range': [self.latitude_coordinates.min(), self.latitude_coordinates.max()],
            'num_longitude_points': self.num_longitude_points,
            'num_latitude_points': self.num_latitude_points,
            'total_grid_points': self.num_longitude_points * self.num_latitude_points,
            'grid_area_degrees_squared': (
                (self.longitude_coordinates.max() - self.longitude_coordinates.min()) *
                (self.latitude_coordinates.max() - self.latitude_coordinates.min())
            )
        }


def load_land_sea_mask(resolution: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load land-sea mask data for specified resolution.

    Equivalent to MATLAB loadlandseamask(0.25) function called in:
    CARDAMOM_MAPS_READ_GFED_NOV24.m line 11: [~,lsfrac025]=loadlandseamask(0.25);

    This function provides land fraction data where values represent the
    fraction of each grid cell that is land (vs ocean). Used for masking
    fire data to land regions only.

    Args:
        resolution: Grid resolution in decimal degrees (default: 0.25)

    Returns:
        tuple: (land_mask_binary, land_fraction_data)
            - land_mask_binary: Binary land mask (1=land, 0=ocean)
            - land_fraction_data: Land fraction (0.0-1.0) for each grid cell

    Notes:
        Currently returns placeholder data. In full implementation, this would
        load from a land-sea mask dataset (e.g., MODIS, Natural Earth, GSHHG).
    """
    # Create coordinate grid at specified resolution
    grid = CoordinateGrid(resolution=resolution)
    
    # Create placeholder land-sea mask
    # In full implementation, this would load from actual land-sea mask dataset
    n_lat = grid.num_latitude_points
    n_lon = grid.num_longitude_points
    
    # Create simple land mask: all points as land except extreme latitudes (ice/ocean)
    land_fraction = np.ones((n_lat, n_lon))
    
    # Set polar regions (>80°N, <-60°S) as mostly ocean
    lat_coords = grid.latitude_coordinates
    for i, lat in enumerate(lat_coords):
        if lat > 80 or lat < -60:
            land_fraction[i, :] = 0.1  # Mostly ocean with some ice
    
    # Create simple ocean regions (approximation)
    # This is a placeholder - real implementation would use actual coastline data
    lon_coords = grid.longitude_coordinates
    for j, lon in enumerate(lon_coords):
        for i, lat in enumerate(lat_coords):
            # Simple Pacific Ocean approximation
            if -180 <= lon <= -80 and -40 <= lat <= 60:
                land_fraction[i, j] = 0.2
            # Simple Atlantic Ocean approximation  
            elif -80 <= lon <= 20 and -40 <= lat <= 70:
                land_fraction[i, j] = 0.3
    
    # Create binary land mask (>0.5 threshold)
    land_mask_binary = (land_fraction > 0.5).astype(float)
    
    return land_mask_binary, land_fraction


def convert_to_geoschem_grid(data: np.ndarray) -> np.ndarray:
    """
    Convert regular grid data to GeosChem 4x5 degree grid.

    Equivalent to MATLAB GEOSChem_regular_grid_to_GC() function called in:
    CARDAMOM_MAPS_READ_GFED_NOV24.m lines 51-52:
    GFED.BA=GEOSChem_regular_grid_to_GC(GBA);
    GFED.FireC=GEOSChem_regular_grid_to_GC(GCE)*12/365.25;

    Args:
        data: Input data array with shape (lat, lon) or (lat, lon, time)
              at regular grid resolution (typically 0.25 degrees)

    Returns:
        np.ndarray: Data regridded to GeosChem 4°x5° grid

    Notes:
        Currently returns placeholder aggregation. In full implementation,
        this would use the actual GeosChem grid definition and proper
        area-weighted regridding.
    """
    if data.ndim == 2:
        # 2D data (lat, lon)
        # Simple aggregation: 4° latitude × 5° longitude
        # 0.25° grid: 16 cells per 4° lat, 20 cells per 5° lon
        lat_factor = 16  # 4° / 0.25°
        lon_factor = 20  # 5° / 0.25°
        
        n_lat_gc = data.shape[0] // lat_factor
        n_lon_gc = data.shape[1] // lon_factor
        
        # Reshape and aggregate
        data_reshaped = data[:n_lat_gc*lat_factor, :n_lon_gc*lon_factor]
        data_reshaped = data_reshaped.reshape(n_lat_gc, lat_factor, n_lon_gc, lon_factor)
        
        # Take mean over aggregated cells (with NaN handling)
        gc_data = np.nanmean(data_reshaped, axis=(1, 3))
        
    elif data.ndim == 3:
        # 3D data (lat, lon, time)
        lat_factor = 16
        lon_factor = 20
        
        n_lat_gc = data.shape[0] // lat_factor
        n_lon_gc = data.shape[1] // lon_factor
        n_time = data.shape[2]
        
        gc_data = np.zeros((n_lat_gc, n_lon_gc, n_time))
        
        # Process each time step
        for t in range(n_time):
            data_2d = data[:n_lat_gc*lat_factor, :n_lon_gc*lon_factor, t]
            data_reshaped = data_2d.reshape(n_lat_gc, lat_factor, n_lon_gc, lon_factor)
            gc_data[:, :, t] = np.nanmean(data_reshaped, axis=(1, 3))
            
    else:
        raise ValueError(f"Unsupported data dimensions: {data.ndim}")
    
    return gc_data


class StandardGrids:
    """
    Factory class for creating standard CARDAMOM coordinate grids.

    This class provides convenient methods for creating commonly used
    coordinate grids in CARDAMOM processing workflows.
    """

    @staticmethod
    def create_global_half_degree() -> CoordinateGrid:
        """
        Create global 0.5° grid (720×360).

        This is the primary CARDAMOM grid used for most global analyses.

        Returns:
            CoordinateGrid with 0.5° resolution and global coverage
        """
        return CoordinateGrid(resolution=0.5, bounds=[-89.75, -179.75, 89.75, 179.75])

    @staticmethod
    def create_global_quarter_degree() -> CoordinateGrid:
        """
        Create global 0.25° grid (1440×720).

        This high-resolution grid is used for GFED fire data and detailed analyses.

        Returns:
            CoordinateGrid with 0.25° resolution and global coverage
        """
        return CoordinateGrid(resolution=0.25, bounds=[-89.75, -179.75, 89.75, 179.75])

    @staticmethod
    def create_conus_half_degree() -> CoordinateGrid:
        """
        Create CONUS 0.5° grid for diurnal processing.

        Covers the continental United States region used for diurnal flux analysis.

        Returns:
            CoordinateGrid with 0.5° resolution covering CONUS region
        """
        # CONUS bounds: North=60°, West=-130°, South=20°, East=-50°
        return CoordinateGrid(resolution=0.5, bounds=[20, -130, 60, -50])

    @staticmethod
    def create_geoschem_4x5_degree() -> CoordinateGrid:
        """
        Create GeosChem 4×5° grid (72×46).

        This coarse-resolution grid matches the GeosChem atmospheric chemistry model.

        Returns:
            CoordinateGrid with 4×5° resolution (4° latitude, 5° longitude)
        """
        # GeosChem typically uses 4° latitude, 5° longitude resolution
        # Creating simplified version - in practice, GeosChem has irregular spacing
        return CoordinateGrid(resolution=4.0, bounds=[-89.75, -179.75, 89.75, 179.75])

    @staticmethod
    def create_custom_grid(resolution: float, region_bounds: List[float]) -> CoordinateGrid:
        """
        Create custom coordinate grid with specified resolution and bounds.

        Args:
            resolution: Grid resolution in decimal degrees
            region_bounds: Geographic bounds as [South, West, North, East]

        Returns:
            CoordinateGrid with custom specifications
        """
        return CoordinateGrid(resolution=resolution, bounds=region_bounds)

    @staticmethod
    def get_standard_grid_info() -> Dict[str, Dict[str, Any]]:
        """
        Get information about all standard grids.

        Returns:
            Dictionary containing metadata for each standard grid type
        """
        grids = {
            'global_0.5deg': StandardGrids.create_global_half_degree(),
            'global_0.25deg': StandardGrids.create_global_quarter_degree(),
            'conus_0.5deg': StandardGrids.create_conus_half_degree(),
            'geoschem_4x5deg': StandardGrids.create_geoschem_4x5_degree()
        }

        return {name: grid.get_grid_info() for name, grid in grids.items()}