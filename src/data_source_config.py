#!/usr/bin/env python3
"""
Data Source Configuration Management for CARDAMOM Downloaders

Provides centralized configuration management for all data sources
used in CARDAMOM preprocessing pipeline with validation and defaults.
"""

import os
import yaml
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path


class DataSourceConfig:
    """
    Centralized configuration management for CARDAMOM data sources.

    Manages configuration for ECMWF, NOAA, GFED, and MODIS downloaders
    with validation, defaults, and environment-specific overrides.

    Scientific Context:
    Each data source has specific requirements for authentication, spatial/temporal
    coverage, and processing parameters. Centralized configuration ensures
    consistent settings across different CARDAMOM preprocessing workflows.
    """

    def __init__(self, config_file: Optional[str] = None, custom_config: Optional[Dict[str, Any]] = None):
        """
        Initialize data source configuration.

        Args:
            config_file: Path to YAML configuration file
            custom_config: Dictionary with custom configuration overrides
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        # Load default configuration
        self.config = self._get_default_config()

        # Load from file if provided
        if config_file:
            self._load_config_file(config_file)

        # Apply custom overrides
        if custom_config:
            self._merge_config(custom_config)

        # Validate configuration
        self._validate_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for all data sources.

        Returns:
            dict: Default configuration dictionary
        """
        return {
            'ecmwf': {
                'base_url': 'https://cds.climate.copernicus.eu/api/v2',
                'datasets': {
                    'hourly': 'reanalysis-era5-single-levels',
                    'monthly': 'reanalysis-era5-single-levels-monthly-means'
                },
                'rate_limit_requests_per_minute': 10,
                'default_area': [-89.75, -179.75, 89.75, 179.75],  # Global
                'default_grid': ['0.5/0.5'],
                'data_format': 'netcdf',
                'download_format': 'unarchived',
                'timeout_seconds': 3600,
                'max_retries': 3,
                'output_dir': './DATA/ECMWF/',
                'essential_variables': [
                    '2m_temperature',
                    '2m_dewpoint_temperature',
                    'total_precipitation',
                    'surface_solar_radiation_downwards'
                ]
            },
            'noaa': {
                'ftp_server': 'aftp.cmdl.noaa.gov',
                'data_path': '/products/trends/co2/co2_mm_gl.txt',
                'update_frequency': 'monthly',
                'timeout_seconds': 300,
                'max_retries': 3,
                'output_dir': './DATA/NOAA_CO2/',
                'spatial_replication': {
                    'default_resolution': '0.5deg',
                    'supported_resolutions': ['0.25deg', '0.5deg', '1.0deg']
                },
                'data_validation': {
                    'min_co2_ppm': 250,
                    'max_co2_ppm': 500,
                    'expected_trend_ppm_per_year': [1.0, 3.0]
                }
            },
            'gfed': {
                'base_url': 'https://www.globalfiredata.org/data_new/',
                'file_pattern': 'GFED4.1s_{year}{beta}.hdf5',
                'requires_registration': True,
                'available_years': {
                    'historical': [2001, 2016],
                    'beta': [2017, 2024]
                },
                'timeout_seconds': 1800,  # 30 minutes for large files
                'max_retries': 2,
                'output_dir': './DATA/GFED4/',
                'vegetation_types': ['SAVA', 'BORF', 'TEMF', 'DEFO', 'PEAT', 'AGRI'],
                'spatial_resolution': '0.25deg',
                'temporal_resolution': 'monthly'
            },
            'modis': {
                'servers': [
                    'https://e4ftl01.cr.usgs.gov/MOTA/',
                    'https://n5eil01u.ecs.nsidc.org/'
                ],
                'products': {
                    'MCD12Q1': {
                        'description': 'Land Cover Type',
                        'resolution': '500m',
                        'temporal_coverage': 'yearly'
                    },
                    'MOD44W': {
                        'description': 'Land Water Mask',
                        'resolution': '250m',
                        'temporal_coverage': 'yearly'
                    }
                },
                'supported_resolutions': ['0.25deg', '0.5deg', '1.0deg'],
                'default_resolution': '0.5deg',
                'timeout_seconds': 1200,
                'max_retries': 3,
                'output_dir': './DATA/MODIS_LSM/',
                'authentication_required': True
            },
            'common': {
                'logging': {
                    'level': 'INFO',
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    'file': 'cardamom_downloader.log'
                },
                'parallel_downloads': {
                    'enabled': False,  # Simple implementation - no parallel downloads
                    'max_workers': 1
                },
                'quality_control': {
                    'enabled': True,
                    'checksum_validation': True,
                    'file_size_checks': True,
                    'data_range_validation': True
                }
            }
        }

    def _load_config_file(self, config_file: str) -> None:
        """
        Load configuration from YAML file.

        Args:
            config_file: Path to YAML configuration file
        """
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                self.logger.warning(f"Configuration file not found: {config_file}")
                return

            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)

            if file_config:
                self._merge_config(file_config)
                self.logger.info(f"Loaded configuration from {config_file}")

        except Exception as e:
            self.logger.error(f"Failed to load configuration file {config_file}: {e}")

    def _merge_config(self, custom_config: Dict[str, Any]) -> None:
        """
        Merge custom configuration with existing configuration.

        Args:
            custom_config: Dictionary with configuration overrides
        """
        def merge_dict(base_dict: Dict, custom_dict: Dict) -> Dict:
            """Recursively merge dictionaries."""
            result = base_dict.copy()
            for key, value in custom_dict.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dict(result[key], value)
                else:
                    result[key] = value
            return result

        self.config = merge_dict(self.config, custom_config)

    def _validate_config(self) -> None:
        """Validate configuration for required fields and reasonable values."""
        validation_errors = []

        # Validate ECMWF configuration
        ecmwf_config = self.config.get('ecmwf', {})
        if not ecmwf_config.get('base_url'):
            validation_errors.append("ECMWF base_url is required")

        if ecmwf_config.get('rate_limit_requests_per_minute', 0) <= 0:
            validation_errors.append("ECMWF rate_limit_requests_per_minute must be positive")

        # Validate NOAA configuration
        noaa_config = self.config.get('noaa', {})
        if not noaa_config.get('ftp_server'):
            validation_errors.append("NOAA ftp_server is required")

        # Validate data validation ranges
        data_val = noaa_config.get('data_validation', {})
        min_co2 = data_val.get('min_co2_ppm', 0)
        max_co2 = data_val.get('max_co2_ppm', 0)
        if min_co2 >= max_co2:
            validation_errors.append("NOAA min_co2_ppm must be less than max_co2_ppm")

        # Validate GFED configuration
        gfed_config = self.config.get('gfed', {})
        if not gfed_config.get('base_url'):
            validation_errors.append("GFED base_url is required")

        # Validate MODIS configuration
        modis_config = self.config.get('modis', {})
        if not modis_config.get('servers'):
            validation_errors.append("MODIS servers list is required")

        # Validate common settings

        if validation_errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in validation_errors)
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info("Configuration validation passed")

    def get_downloader_config(self, source: str) -> Dict[str, Any]:
        """
        Get configuration for specific downloader.

        Args:
            source: Data source name ('ecmwf', 'noaa', 'gfed', 'modis')

        Returns:
            dict: Configuration for the specified downloader

        Raises:
            ValueError: If source is not supported
        """
        if source not in self.config:
            available_sources = list(self.config.keys())
            raise ValueError(f"Unknown data source: {source}. Available: {available_sources}")

        # Merge with common settings
        downloader_config = self.config[source].copy()
        common_config = self.config.get('common', {})

        # Add common settings that are relevant to downloaders

        if 'quality_control' in common_config:
            downloader_config['quality_control'] = common_config['quality_control']

        return downloader_config

    def get_credentials_config(self, source: str) -> Dict[str, Any]:
        """
        Get credentials configuration for specific data source.

        Args:
            source: Data source name

        Returns:
            dict: Credentials configuration (may be empty if not required)
        """
        credentials = {}

        if source == 'ecmwf':
            # ECMWF requires CDS API credentials
            cds_uid = os.getenv('ECMWF_CDS_UID')
            cds_key = os.getenv('ECMWF_CDS_KEY')

            if cds_uid and cds_key:
                credentials = {
                    'uid': cds_uid,
                    'key': cds_key
                }
            else:
                self.logger.warning("ECMWF CDS credentials not found in environment variables")

        elif source == 'gfed':
            # GFED may require registration but typically no authentication
            gfed_config = self.config.get('gfed', {})
            if gfed_config.get('requires_registration'):
                self.logger.info("GFED data requires registration at globalfiredata.org")

        elif source == 'modis':
            # MODIS may require EarthData credentials
            earthdata_user = os.getenv('EARTHDATA_USERNAME')
            earthdata_pass = os.getenv('EARTHDATA_PASSWORD')

            if earthdata_user and earthdata_pass:
                credentials = {
                    'username': earthdata_user,
                    'password': earthdata_pass
                }
            else:
                self.logger.warning("MODIS EarthData credentials not found in environment variables")

        return credentials

    def validate_downloader_config(self, source: str, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration for specific downloader.

        Args:
            source: Data source name
            config: Configuration to validate

        Returns:
            list: List of validation errors (empty if valid)
        """
        errors = []

        if source == 'ecmwf':
            required_fields = ['output_dir', 'data_format']
            for field in required_fields:
                if field not in config:
                    errors.append(f"ECMWF config missing required field: {field}")

            # Validate area bounds
            area = config.get('default_area', [])
            if len(area) != 4:
                errors.append("ECMWF area must have 4 values: [North, West, South, East]")
            elif area[0] <= area[2]:  # North <= South
                errors.append("ECMWF area: North latitude must be greater than South latitude")

        elif source == 'noaa':
            required_fields = ['ftp_server', 'data_path', 'output_dir']
            for field in required_fields:
                if field not in config:
                    errors.append(f"NOAA config missing required field: {field}")

        elif source == 'gfed':
            required_fields = ['base_url', 'output_dir']
            for field in required_fields:
                if field not in config:
                    errors.append(f"GFED config missing required field: {field}")

            # Validate available years
            available_years = config.get('available_years', {})
            if 'historical' not in available_years or 'beta' not in available_years:
                errors.append("GFED config must specify historical and beta year ranges")

        elif source == 'modis':
            required_fields = ['servers', 'output_dir', 'supported_resolutions']
            for field in required_fields:
                if field not in config:
                    errors.append(f"MODIS config missing required field: {field}")

            # Validate default resolution is in supported list
            default_res = config.get('default_resolution')
            supported_res = config.get('supported_resolutions', [])
            if default_res and default_res not in supported_res:
                errors.append(f"MODIS default_resolution '{default_res}' not in supported_resolutions")

        return errors

    def create_config_template(self, output_file: str) -> None:
        """
        Create a template configuration file.

        Args:
            output_file: Path for the template configuration file
        """
        template_config = self._get_default_config()

        # Add comments and examples
        template_with_comments = {
            '# CARDAMOM Data Source Configuration': None,
            '# This file configures all data downloaders for CARDAMOM preprocessing': None,
            '# Modify values as needed for your specific environment and requirements': None,
            '': None,
            **template_config
        }

        try:
            with open(output_file, 'w') as f:
                yaml.dump(template_config, f, default_flow_style=False, sort_keys=False, indent=2)

            self.logger.info(f"Created configuration template: {output_file}")

        except Exception as e:
            self.logger.error(f"Failed to create configuration template: {e}")
            raise

    def get_all_output_directories(self) -> Dict[str, str]:
        """
        Get all output directories for data sources.

        Returns:
            dict: Mapping of data source to output directory
        """
        output_dirs = {}
        for source in ['ecmwf', 'noaa', 'gfed', 'modis']:
            config = self.get_downloader_config(source)
            output_dirs[source] = config.get('output_dir', f'./DATA/{source.upper()}/')

        return output_dirs

    def ensure_output_directories(self) -> None:
        """Create all output directories if they don't exist."""
        output_dirs = self.get_all_output_directories()

        for source, output_dir in output_dirs.items():
            try:
                os.makedirs(output_dir, exist_ok=True)
                self.logger.info(f"Ensured output directory exists: {source} -> {output_dir}")
            except Exception as e:
                self.logger.error(f"Failed to create output directory for {source}: {e}")

    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get summary of current configuration.

        Returns:
            dict: Configuration summary
        """
        summary = {
            'data_sources': list(self.config.keys()),
            'output_directories': self.get_all_output_directories(),
            'quality_control_enabled': self.config.get('common', {}).get('quality_control', {}).get('enabled', False)
        }

        # Add source-specific information
        for source in ['ecmwf', 'noaa', 'gfed', 'modis']:
            if source in self.config:
                source_config = self.config[source]
                summary[f'{source}_configured'] = True
                summary[f'{source}_timeout'] = source_config.get('timeout_seconds', 0)
                summary[f'{source}_max_retries'] = source_config.get('max_retries', 0)

        return summary