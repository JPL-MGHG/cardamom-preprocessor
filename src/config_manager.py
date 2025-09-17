"""
Unified Configuration System for CARDAMOM Preprocessing

This module provides centralized configuration management with clear hierarchy:
1. Built-in defaults (lowest priority)
2. Configuration files (YAML/JSON)
3. Environment variables
4. Command-line arguments (highest priority)

This single class replaces all configuration management across phases.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union


class CardamomConfig:
    """
    Unified configuration system for CARDAMOM preprocessing.

    Provides centralized configuration management with clear hierarchy:
    1. Built-in defaults
    2. Configuration files (YAML/JSON)
    3. Environment variables
    4. Command-line arguments (highest priority)

    This single class replaces all configuration management across phases.
    """

    def __init__(self, config_file: Optional[str] = None, cli_args: Optional[Dict] = None):
        """
        Initialize configuration system with proper precedence order.

        Args:
            config_file: Path to YAML or JSON configuration file
            cli_args: Dictionary of command-line arguments (highest priority)
        """
        self.config_file = config_file
        self.cli_args = cli_args or {}
        self._config = {}
        self._load_configuration()

    def _load_configuration(self):
        """Load configuration with proper precedence order"""
        # 1. Start with built-in defaults
        self._config = self._get_default_config()

        # 2. Override with config file
        if self.config_file:
            file_config = self._load_config_file(self.config_file)
            self._merge_config(self._config, file_config)

        # 3. Override with environment variables
        env_config = self._load_environment_config()
        self._merge_config(self._config, env_config)

        # 4. Override with CLI arguments (highest priority)
        if self.cli_args:
            self._merge_config(self._config, self.cli_args)

        # 5. Validate final configuration
        self._validate_configuration()

    def _get_default_config(self) -> Dict[str, Any]:
        """Built-in default configuration"""
        return {
            'processing': {
                'output_directory': "./DATA/CARDAMOM-MAPS_05deg_MET/",
                'temp_directory': "./temp/",
                'max_workers': 4,
                'retry_attempts': 3,
                'compression': True,
                'log_level': 'INFO'
            },
            'pipeline': {
                'resume_on_failure': True,
                'cleanup_temp_files': True,
                'generate_qa_reports': True,
                'parallel_downloads': True,
                'state_file': 'pipeline_state.json'
            },
            'global_monthly': {
                'resolution': 0.5,
                'grid_bounds': [-89.75, -179.75, 89.75, 179.75],  # S, W, N, E
                'variables': {
                    'era5': ["2m_temperature", "2m_dewpoint_temperature", "total_precipitation",
                            "skin_temperature", "surface_solar_radiation_downwards", "snowfall",
                            "surface_thermal_radiation_downwards"],
                    'noaa': ["co2"],
                    'gfed': ["burned_area", "fire_carbon"],
                    'modis': ["land_sea_mask", "land_sea_fraction"]
                },
                'netcdf_template': '3D',
                'time_aggregation': 'monthly'
            },
            'conus_diurnal': {
                'resolution': 0.5,
                'region': [60, -130, 20, -50],  # N, W, S, E
                'variables': {
                    'cms': ["GPP", "NEE", "REC", "FIR", "NBE"],
                    'era5': ["skin_temperature", "surface_solar_radiation_downwards"],
                    'gfed': ["diurnal_fire_pattern"]
                },
                'netcdf_template': '3D',
                'time_aggregation': 'hourly',
                'diurnal_hours': list(range(24)),
                'years_range': [2015, 2020],  # Default processing years
                'cms_experiment_id': 'CMS_V1',  # Default CMS experiment identifier
                'output_base_dir': './DUMPFILES',  # Base directory for outputs
                'data_source_dir': './DATA',  # Base directory for input data
                'era5_diurnal_subdir': 'ERA5_CUSTOM/CONUS_DIURNAL'  # ERA5 diurnal data subdirectory
            },
            'downloaders': {
                'era5': {
                    'api_timeout': 3600,
                    'max_concurrent': 3,
                    'chunk_size': 'auto'
                },
                'noaa': {
                    'ftp_timeout': 300,
                    'retry_delay': 5
                },
                'gfed': {
                    'username': None,  # Set via environment
                    'password': None,  # Set via environment
                    'hdf5_chunks': True,
                    'default_years': [2001, 2024],  # Default year range for GFED processing
                    'historical_cutoff_year': 2016,  # Year after which beta versions are used
                    'gap_fill_reference_period': [2001, 2016]  # Reference period for gap-filling
                },
                'modis': {
                    'earthdata_username': None,  # Set via environment
                    'earthdata_password': None   # Set via environment
                }
            },
            'quality_control': {
                'enable_validation': True,
                'physical_range_checks': True,
                'spatial_continuity_checks': True,
                'temporal_continuity_checks': True,
                'missing_data_tolerance': 0.05  # 5% missing data allowed
            }
        }

    def _load_config_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                try:
                    import yaml
                    return yaml.safe_load(f) or {}
                except ImportError:
                    raise ImportError("PyYAML is required for YAML configuration files. Install with: pip install pyyaml")
            elif config_path.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    def _load_environment_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        env_config = {}

        # Map environment variables to config paths
        env_mappings = {
            'CARDAMOM_OUTPUT_DIR': 'processing.output_directory',
            'CARDAMOM_TEMP_DIR': 'processing.temp_directory',
            'CARDAMOM_LOG_LEVEL': 'processing.log_level',
            'CARDAMOM_MAX_WORKERS': 'processing.max_workers',
            'ECMWF_CDS_UID': 'downloaders.era5.username',
            'ECMWF_CDS_KEY': 'downloaders.era5.api_key',
            'NOAA_FTP_USER': 'downloaders.noaa.username',
            'NOAA_FTP_PASS': 'downloaders.noaa.password',
            'GFED_USERNAME': 'downloaders.gfed.username',
            'GFED_PASSWORD': 'downloaders.gfed.password',
            'EARTHDATA_USERNAME': 'downloaders.modis.earthdata_username',
            'EARTHDATA_PASSWORD': 'downloaders.modis.earthdata_password'
        }

        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_config(env_config, config_path, value)

        return env_config

    def _set_nested_config(self, config_dict: Dict, path: str, value: Any):
        """Set nested configuration value using dot notation path"""
        keys = path.split('.')
        current = config_dict

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Convert string values to appropriate types
        if isinstance(value, str):
            if value.lower() in ['true', 'false']:
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            elif value.replace('.', '', 1).isdigit():
                value = float(value)

        current[keys[-1]] = value

    def _merge_config(self, base_config: Dict, override_config: Dict):
        """Deep merge configuration dictionaries"""
        for key, value in override_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._merge_config(base_config[key], value)
            else:
                base_config[key] = value

    def _validate_configuration(self):
        """Validate final configuration"""
        required_sections = ['processing', 'pipeline', 'global_monthly', 'conus_diurnal', 'downloaders']

        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Required configuration section missing: {section}")

        # Validate output directory
        output_dir = Path(self._config['processing']['output_directory'])
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        # Validate temp directory
        temp_dir = Path(self._config['processing']['temp_directory'])
        if not temp_dir.exists():
            temp_dir.mkdir(parents=True, exist_ok=True)

    # Public interface methods
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation path.

        Args:
            path: Dot-separated path to configuration value (e.g., 'processing.output_directory')
            default: Default value if path not found

        Returns:
            Configuration value or default if not found
        """
        keys = path.split('.')
        current = self._config

        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default

    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing-specific configuration"""
        return self._config['processing']

    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get pipeline-specific configuration"""
        return self._config['pipeline']

    def get_workflow_config(self, workflow_type: str) -> Dict[str, Any]:
        """
        Get configuration for specific workflow.

        Args:
            workflow_type: Type of workflow ('global_monthly' or 'conus_diurnal')

        Returns:
            Workflow-specific configuration dictionary
        """
        if workflow_type not in self._config:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
        return self._config[workflow_type]

    def get_downloader_config(self, downloader_name: str) -> Dict[str, Any]:
        """
        Get configuration for specific downloader.

        Args:
            downloader_name: Name of downloader ('era5', 'noaa', 'gfed', 'modis')

        Returns:
            Downloader-specific configuration dictionary
        """
        return self._config['downloaders'].get(downloader_name, {})

    def get_quality_control_config(self) -> Dict[str, Any]:
        """Get quality control configuration"""
        return self._config['quality_control']

    def to_dict(self) -> Dict[str, Any]:
        """Return complete configuration as dictionary"""
        return self._config.copy()

    def save_config(self, output_path: str):
        """
        Save current configuration to file.

        Args:
            output_path: Path where to save configuration file
        """
        output_path = Path(output_path)

        if output_path.suffix.lower() in ['.yaml', '.yml']:
            try:
                import yaml
                with open(output_path, 'w') as f:
                    yaml.dump(self._config, f, default_flow_style=False, indent=2)
            except ImportError:
                raise ImportError("PyYAML is required for YAML output. Install with: pip install pyyaml")
        elif output_path.suffix.lower() == '.json':
            with open(output_path, 'w') as f:
                json.dump(self._config, f, indent=2)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")