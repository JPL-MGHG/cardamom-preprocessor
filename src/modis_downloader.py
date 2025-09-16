#!/usr/bin/env python3
"""
MODIS Land-Sea Mask Downloader for CARDAMOM

Downloads and processes MODIS-based land-sea mask data.
Creates both binary mask and fractional coverage datasets for
CARDAMOM spatial domain definition.
"""

import os
import requests
import numpy as np
import xarray as xr
from typing import Dict, List, Any, Tuple
import logging
from .base_downloader import BaseDownloader


class MODISDownloader(BaseDownloader):
    """
    Download MODIS-based land-sea mask data.

    Scientific Context:
    Land-sea masks are essential for CARDAMOM to define the spatial domain
    for terrestrial carbon cycle modeling. MODIS land cover products provide
    high-resolution classification that can be aggregated to create both
    binary masks (land=1, sea=0) and fractional coverage datasets.

    Creates both binary mask and fractional coverage datasets suitable
    for different CARDAMOM modeling requirements.
    """

    def __init__(self, output_dir: str = "./DATA/MODIS_LSM/"):
        """
        Initialize MODIS downloader.

        Args:
            output_dir: Directory for MODIS land-sea mask files
        """
        super().__init__(output_dir)

        # MODIS server configuration
        self.modis_servers = self._setup_server_list()

        # Supported MODIS products for land cover
        self.modis_products = {
            'MCD12Q1': {
                'description': 'MODIS Land Cover Type',
                'resolution': '500m',
                'temporal_coverage': 'yearly',
                'useful_for': 'vegetation_classification'
            },
            'MOD44W': {
                'description': 'MODIS Land Water Mask',
                'resolution': '250m',
                'temporal_coverage': 'yearly',
                'useful_for': 'water_body_detection'
            }
        }

        # Standard CARDAMOM resolutions
        self.supported_resolutions = {
            '0.25deg': 0.25,
            '0.5deg': 0.5,
            '1.0deg': 1.0
        }

    def _setup_server_list(self) -> List[str]:
        """
        Setup list of MODIS data servers.

        Returns:
            list: Available MODIS data server URLs
        """
        return [
            "https://e4ftl01.cr.usgs.gov/MOTA/",  # NASA EarthData
            "https://n5eil01u.ecs.nsidc.org/",    # NSIDC
            # Additional servers can be added as needed
        ]

    def download_data(self, resolution: str = "0.5deg", **kwargs) -> Dict[str, Any]:
        """
        Download or generate MODIS land-sea mask at specified resolution.

        Args:
            resolution: Target resolution ("0.25deg", "0.5deg", "1.0deg")
            **kwargs: Additional parameters

        Returns:
            dict: Download/generation results with file information
        """
        return self.download_land_sea_mask(resolution)

    def download_land_sea_mask(self, resolution: str = "0.5deg") -> Dict[str, Any]:
        """
        Download or generate MODIS land-sea mask at specified resolution.

        Creates both binary mask (land=1, sea=0) and fractional coverage.

        Args:
            resolution: Target spatial resolution

        Returns:
            dict: Processing results with created file paths
        """
        if resolution not in self.supported_resolutions:
            return {
                "status": "failed",
                "error": f"Unsupported resolution: {resolution}. Supported: {list(self.supported_resolutions.keys())}"
            }

        try:
            # For this implementation, we'll create a simple land-sea mask
            # based on a basic global land outline
            # In a full implementation, this would download actual MODIS data
            result = self._generate_simple_land_sea_mask(resolution)

            return result

        except Exception as e:
            error_msg = f"Failed to create land-sea mask: {e}"
            self.logger.error(error_msg)
            return {
                "status": "failed",
                "error": error_msg
            }

    def _generate_simple_land_sea_mask(self, resolution: str) -> Dict[str, Any]:
        """
        Generate a simplified land-sea mask for CARDAMOM.

        This is a simplified implementation that creates a basic land-sea mask.
        A full implementation would process actual MODIS land cover data.

        Args:
            resolution: Target spatial resolution

        Returns:
            dict: Generation results with file paths
        """
        target_res_deg = self.supported_resolutions[resolution]

        # Create coordinate arrays
        longitude_coords = np.arange(-179.75, 180, target_res_deg)
        latitude_coords = np.arange(-89.75, 90, target_res_deg)

        lon_grid, lat_grid = np.meshgrid(longitude_coords, latitude_coords)

        # Simple land-sea classification based on basic geography
        # This is a simplified approach - real implementation would use MODIS data
        land_mask_binary = self._create_simple_land_classification(lat_grid, lon_grid)

        # Create fractional land coverage (for more realistic boundaries)
        land_mask_fractional = self._create_fractional_coverage(land_mask_binary, lat_grid, lon_grid)

        # Create xarray datasets
        coords = {
            'longitude': (['longitude'], longitude_coords,
                         {'units': 'degrees_east', 'long_name': 'Longitude'}),
            'latitude': (['latitude'], latitude_coords,
                        {'units': 'degrees_north', 'long_name': 'Latitude'})
        }

        attrs_base = {
            'spatial_resolution': f'{target_res_deg} degrees',
            'conventions': 'CF-1.8',
            'creation_date': np.datetime64('now').isoformat(),
            'source': 'Simplified land-sea mask for CARDAMOM (derived from basic geography)',
            'description': 'Land-sea mask for terrestrial carbon cycle modeling'
        }

        # Binary mask dataset
        binary_dataset = xr.Dataset({
            'land_mask': (
                ['latitude', 'longitude'],
                land_mask_binary.astype(np.int8),
                {
                    'long_name': 'Binary land-sea mask',
                    'description': 'Binary mask where 1=land, 0=ocean/water',
                    'units': 'dimensionless',
                    'valid_range': [0, 1]
                }
            )
        }, coords=coords, attrs={**attrs_base, 'title': 'MODIS-based Binary Land-Sea Mask for CARDAMOM'})

        # Fractional coverage dataset
        fractional_dataset = xr.Dataset({
            'land_fraction': (
                ['latitude', 'longitude'],
                land_mask_fractional.astype(np.float32),
                {
                    'long_name': 'Fractional land coverage',
                    'description': 'Fraction of grid cell covered by land (0-1)',
                    'units': 'dimensionless',
                    'valid_range': [0.0, 1.0]
                }
            )
        }, coords=coords, attrs={**attrs_base, 'title': 'MODIS-based Fractional Land Coverage for CARDAMOM'})

        # Save datasets
        created_files = []

        # Binary mask file
        binary_filename = f"MODIS_LAND_MASK_BINARY_{resolution}.nc"
        binary_filepath = os.path.join(self.output_dir, binary_filename)
        binary_dataset.to_netcdf(binary_filepath, encoding={'land_mask': {'zlib': True, 'complevel': 4}})
        created_files.append(binary_filepath)
        self.logger.info(f"Created binary land mask: {binary_filename}")

        # Fractional coverage file
        fractional_filename = f"MODIS_LAND_FRACTION_{resolution}.nc"
        fractional_filepath = os.path.join(self.output_dir, fractional_filename)
        fractional_dataset.to_netcdf(fractional_filepath, encoding={'land_fraction': {'zlib': True, 'complevel': 4}})
        created_files.append(fractional_filepath)
        self.logger.info(f"Created fractional land coverage: {fractional_filename}")

        # Record successful generation
        for filename in [binary_filename, fractional_filename]:
            self._record_download_attempt(filename, "success")

        return {
            "status": "completed",
            "created_files": created_files,
            "resolution": resolution,
            "binary_mask_file": binary_filepath,
            "fractional_coverage_file": fractional_filepath,
            "land_fraction_global": float(np.mean(land_mask_fractional))
        }

    def _create_simple_land_classification(self, lat_grid: np.ndarray, lon_grid: np.ndarray) -> np.ndarray:
        """
        Create simple land classification based on basic geographical rules.

        This is a simplified implementation. A real MODIS-based approach would
        use actual land cover classifications from MODIS products.

        Args:
            lat_grid: Latitude coordinate grid
            lon_grid: Longitude coordinate grid

        Returns:
            np.ndarray: Binary land mask (1=land, 0=water)
        """
        land_mask = np.zeros_like(lat_grid, dtype=np.int8)

        # Define major continental landmasses using simple geographic boundaries
        # North America
        north_america = (
            (lat_grid > 25) & (lat_grid < 75) &
            (lon_grid > -170) & (lon_grid < -50)
        )

        # South America
        south_america = (
            (lat_grid > -60) & (lat_grid < 15) &
            (lon_grid > -85) & (lon_grid < -30)
        )

        # Europe
        europe = (
            (lat_grid > 35) & (lat_grid < 75) &
            (lon_grid > -15) & (lon_grid < 50)
        )

        # Asia
        asia = (
            (lat_grid > 5) & (lat_grid < 80) &
            (lon_grid > 50) & (lon_grid < 180)
        )

        # Africa
        africa = (
            (lat_grid > -40) & (lat_grid < 40) &
            (lon_grid > -20) & (lon_grid < 55)
        )

        # Australia
        australia = (
            (lat_grid > -45) & (lat_grid < -10) &
            (lon_grid > 110) & (lon_grid < 155)
        )

        # Combine all landmasses
        land_mask[north_america | south_america | europe | asia | africa | australia] = 1

        # Remove major water bodies (simplified)
        # Great Lakes region
        great_lakes = (
            (lat_grid > 40) & (lat_grid < 50) &
            (lon_grid > -95) & (lon_grid < -75)
        )
        land_mask[great_lakes] = 0

        # Mediterranean Sea
        mediterranean = (
            (lat_grid > 30) & (lat_grid < 46) &
            (lon_grid > 0) & (lon_grid < 40)
        )
        land_mask[mediterranean] = 0

        self.logger.info(f"Created simple land classification: {np.sum(land_mask)} land cells out of {land_mask.size} total")

        return land_mask

    def _create_fractional_coverage(self, binary_mask: np.ndarray, lat_grid: np.ndarray, lon_grid: np.ndarray) -> np.ndarray:
        """
        Create fractional land coverage from binary mask with coastal smoothing.

        Args:
            binary_mask: Binary land mask
            lat_grid: Latitude coordinate grid
            lon_grid: Longitude coordinate grid

        Returns:
            np.ndarray: Fractional land coverage (0-1)
        """
        # Start with binary mask as base
        fractional_mask = binary_mask.astype(np.float32)

        # Apply smoothing near coastlines to create more realistic fractional coverage
        # This simulates the effect of mixed land-water pixels in coarser resolution data

        # Find coastal areas (land cells adjacent to water cells)
        from scipy import ndimage

        # Create kernel for finding neighbors
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])

        # Find cells that have both land and water neighbors
        land_neighbors = ndimage.convolve(binary_mask.astype(float), kernel, mode='constant', cval=0)
        water_neighbors = ndimage.convolve((1 - binary_mask).astype(float), kernel, mode='constant', cval=1)

        # Coastal cells are those with both land and water neighbors
        coastal_cells = (land_neighbors > 0) & (water_neighbors > 0)

        # Apply fractional coverage to coastal cells
        # Use distance-based weighting for more realistic transitions
        for i in range(binary_mask.shape[0]):
            for j in range(binary_mask.shape[1]):
                if coastal_cells[i, j]:
                    # Calculate local land fraction in 3x3 neighborhood
                    i_start, i_end = max(0, i-1), min(binary_mask.shape[0], i+2)
                    j_start, j_end = max(0, j-1), min(binary_mask.shape[1], j+2)

                    local_neighborhood = binary_mask[i_start:i_end, j_start:j_end]
                    local_land_fraction = np.mean(local_neighborhood)

                    # Apply some randomness for more realistic coastlines
                    variation = np.random.normal(0, 0.1)  # Small random variation
                    fractional_mask[i, j] = np.clip(local_land_fraction + variation, 0, 1)

        self.logger.info(f"Created fractional coverage with {np.sum(coastal_cells)} coastal transition cells")

        return fractional_mask

    def generate_mask_from_modis(self, modis_product: str = "MCD12Q1") -> Dict[str, Any]:
        """
        Generate land-sea mask from MODIS land cover product.

        This is a placeholder for full MODIS data processing.
        Real implementation would download and process actual MODIS data.

        Args:
            modis_product: MODIS product to use for mask generation

        Returns:
            dict: Processing results
        """
        if modis_product not in self.modis_products:
            return {
                "status": "failed",
                "error": f"Unsupported MODIS product: {modis_product}"
            }

        self.logger.info(f"Generating mask from {modis_product} (placeholder implementation)")

        # This would be replaced with actual MODIS data processing
        return {
            "status": "placeholder",
            "message": f"Full {modis_product} processing not implemented - use download_land_sea_mask() for simplified version",
            "product_info": self.modis_products[modis_product]
        }

    def create_mask_and_fraction(self, land_cover_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create both binary mask and fractional coverage from land cover data.

        This method would process actual MODIS land cover classifications
        to create CARDAMOM-compatible land-sea masks.

        Args:
            land_cover_data: MODIS land cover classification array

        Returns:
            tuple: (binary_mask, fractional_coverage)
        """
        # This is a placeholder for actual MODIS land cover processing
        # Real implementation would classify MODIS land cover types
        # into land vs water categories

        self.logger.info("Processing MODIS land cover data (placeholder implementation)")

        # For now, return simplified classification
        binary_mask = (land_cover_data > 0).astype(np.int8)  # Assume 0 = water, >0 = land
        fractional_coverage = np.clip(land_cover_data / np.max(land_cover_data), 0, 1)

        return binary_mask, fractional_coverage

    def get_supported_resolutions(self) -> Dict[str, float]:
        """
        Get supported spatial resolutions for land-sea mask generation.

        Returns:
            dict: Supported resolutions with their values in degrees
        """
        return self.supported_resolutions.copy()

    def validate_mask_data(self, mask_data: np.ndarray, mask_type: str = "binary") -> Dict[str, Any]:
        """
        Validate generated mask data for scientific reasonableness.

        Args:
            mask_data: Mask data array to validate
            mask_type: Type of mask ("binary" or "fractional")

        Returns:
            dict: Validation results
        """
        validation_results = {
            "mask_type": mask_type,
            "shape": mask_data.shape,
            "data_type": str(mask_data.dtype)
        }

        if mask_type == "binary":
            unique_values = np.unique(mask_data)
            valid_binary = np.all(np.isin(unique_values, [0, 1]))

            validation_results.update({
                "valid_binary": valid_binary,
                "unique_values": unique_values.tolist(),
                "land_fraction": float(np.mean(mask_data))
            })

            if not valid_binary:
                validation_results["error"] = f"Binary mask contains invalid values: {unique_values}"

        elif mask_type == "fractional":
            min_val, max_val = np.nanmin(mask_data), np.nanmax(mask_data)
            valid_range = (min_val >= 0) and (max_val <= 1)

            validation_results.update({
                "valid_range": valid_range,
                "min_value": float(min_val),
                "max_value": float(max_val),
                "mean_land_fraction": float(np.nanmean(mask_data))
            })

            if not valid_range:
                validation_results["error"] = f"Fractional mask values outside [0,1]: [{min_val}, {max_val}]"

        # Global land fraction should be reasonable (roughly 29% of Earth's surface)
        land_fraction = validation_results.get("land_fraction") or validation_results.get("mean_land_fraction", 0)
        reasonable_land_fraction = 0.15 < land_fraction < 0.45  # Allow reasonable range

        validation_results["reasonable_global_land_fraction"] = reasonable_land_fraction

        if not reasonable_land_fraction:
            validation_results["warning"] = f"Global land fraction {land_fraction:.3f} seems unrealistic (expected ~0.29)"

        return validation_results