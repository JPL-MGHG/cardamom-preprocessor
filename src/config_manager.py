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
        """Validate final configuration with comprehensive checks for all workflow types"""
        required_sections = ['processing', 'pipeline', 'global_monthly', 'conus_diurnal', 'downloaders']

        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Required configuration section missing: {section}")

        # Validate processing configuration
        self._validate_processing_config()

        # Validate pipeline configuration
        self._validate_pipeline_config()

        # Validate workflow configurations
        self._validate_global_monthly_config()
        self._validate_conus_diurnal_config()

        # Validate downloader configurations
        self._validate_downloader_configs()

        # Validate quality control configuration
        self._validate_quality_control_config()

    def _validate_processing_config(self):
        """Validate processing section configuration"""
        processing = self._config['processing']

        # Validate and create output directory
        output_dir = Path(processing['output_directory'])
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        # Validate and create temp directory
        temp_dir = Path(processing['temp_directory'])
        if not temp_dir.exists():
            temp_dir.mkdir(parents=True, exist_ok=True)

        # Validate numeric parameters
        if processing.get('max_workers', 1) < 1:
            raise ValueError("max_workers must be at least 1")

        if processing.get('retry_attempts', 0) < 0:
            raise ValueError("retry_attempts must be non-negative")

        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if processing.get('log_level', 'INFO') not in valid_log_levels:
            raise ValueError(f"log_level must be one of: {valid_log_levels}")

    def _validate_pipeline_config(self):
        """Validate pipeline section configuration"""
        pipeline = self._config['pipeline']

        # Validate boolean parameters
        boolean_params = ['resume_on_failure', 'cleanup_temp_files', 'generate_qa_reports', 'parallel_downloads']
        for param in boolean_params:
            if param in pipeline and not isinstance(pipeline[param], bool):
                raise ValueError(f"Pipeline parameter '{param}' must be boolean")

    def _validate_global_monthly_config(self):
        """Validate global_monthly workflow configuration"""
        global_config = self._config['global_monthly']

        # Validate resolution
        valid_resolutions = [0.25, 0.5, 1.0]
        if global_config.get('resolution') not in valid_resolutions:
            raise ValueError(f"Resolution must be one of: {valid_resolutions}")

        # Validate grid bounds [S, W, N, E]
        bounds = global_config.get('grid_bounds')
        if bounds and len(bounds) != 4:
            raise ValueError("Grid bounds must be [South, West, North, East]")

        if bounds:
            south, west, north, east = bounds
            if not (-90 <= south <= north <= 90):
                raise ValueError("Invalid latitude bounds. Must be -90 <= South <= North <= 90")
            if not (-180 <= west <= east <= 180):
                raise ValueError("Invalid longitude bounds. Must be -180 <= West <= East <= 180")

        # Validate variables structure
        variables = global_config.get('variables', {})
        expected_sources = ['era5', 'noaa', 'gfed', 'modis']
        for source in expected_sources:
            if source in variables and not isinstance(variables[source], list):
                raise ValueError(f"Variables for '{source}' must be a list")

    def _validate_conus_diurnal_config(self):
        """Validate conus_diurnal workflow configuration"""
        diurnal_config = self._config['conus_diurnal']

        # Validate resolution
        valid_resolutions = [0.25, 0.5, 1.0]
        if diurnal_config.get('resolution') not in valid_resolutions:
            raise ValueError(f"CONUS diurnal resolution must be one of: {valid_resolutions}")

        # Validate CONUS region bounds [N, W, S, E]
        region = diurnal_config.get('region')
        if region and len(region) != 4:
            raise ValueError("CONUS region must be [North, West, South, East]")

        if region:
            north, west, south, east = region
            if not (south <= north):
                raise ValueError("Invalid CONUS region: South must be <= North")
            if not (west <= east):
                raise ValueError("Invalid CONUS region: West must be <= East")

            # Check if region is reasonable for CONUS
            if not (10 <= south <= 70 and 10 <= north <= 70):
                print("Warning: CONUS region latitude bounds seem outside typical range (10-70°N)")
            if not (-180 <= west <= -50 and -180 <= east <= -50):
                print("Warning: CONUS region longitude bounds seem outside typical range (-180 to -50°E)")

        # Validate years range
        years_range = diurnal_config.get('years_range')
        if years_range and len(years_range) != 2:
            raise ValueError("years_range must be [start_year, end_year]")

        if years_range:
            start_year, end_year = years_range
            if start_year > end_year:
                raise ValueError("Start year must be <= end year")
            if start_year < 1979:
                print("Warning: Start year before 1979 may not have ERA5 data available")

        # Validate diurnal hours
        diurnal_hours = diurnal_config.get('diurnal_hours', list(range(24)))
        if not all(0 <= hour <= 23 for hour in diurnal_hours):
            raise ValueError("diurnal_hours must be integers between 0-23")

    def _validate_downloader_configs(self):
        """Validate downloader configurations"""
        downloaders = self._config['downloaders']

        # Validate ERA5 config
        if 'era5' in downloaders:
            era5_config = downloaders['era5']
            if era5_config.get('api_timeout', 0) <= 0:
                raise ValueError("ERA5 api_timeout must be positive")
            if era5_config.get('max_concurrent', 0) <= 0:
                raise ValueError("ERA5 max_concurrent must be positive")

        # Validate NOAA config
        if 'noaa' in downloaders:
            noaa_config = downloaders['noaa']
            if noaa_config.get('ftp_timeout', 0) <= 0:
                raise ValueError("NOAA ftp_timeout must be positive")
            if noaa_config.get('retry_delay', 0) < 0:
                raise ValueError("NOAA retry_delay must be non-negative")

        # Validate GFED config
        if 'gfed' in downloaders:
            gfed_config = downloaders['gfed']
            default_years = gfed_config.get('default_years')
            if default_years and len(default_years) != 2:
                raise ValueError("GFED default_years must be [start_year, end_year]")

            gap_fill_period = gfed_config.get('gap_fill_reference_period')
            if gap_fill_period and len(gap_fill_period) != 2:
                raise ValueError("GFED gap_fill_reference_period must be [start_year, end_year]")

    def _validate_quality_control_config(self):
        """Validate quality control configuration"""
        qc_config = self._config['quality_control']

        # Validate tolerance
        tolerance = qc_config.get('missing_data_tolerance', 0.05)
        if not (0 <= tolerance <= 1):
            raise ValueError("missing_data_tolerance must be between 0 and 1")

        # Validate boolean flags
        boolean_flags = ['enable_validation', 'physical_range_checks',
                        'spatial_continuity_checks', 'temporal_continuity_checks']
        for flag in boolean_flags:
            if flag in qc_config and not isinstance(qc_config[flag], bool):
                raise ValueError(f"Quality control flag '{flag}' must be boolean")

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

    @classmethod
    def create_template_config(cls, template_type: str = "complete", output_path: Optional[str] = None) -> str:
        """
        Create configuration file templates for different use cases.

        Args:
            template_type: Type of template ('complete', 'minimal', 'global_only', 'diurnal_only')
            output_path: Optional path to save template (default: generates appropriate filename)

        Returns:
            String containing the generated template configuration
        """
        if template_type == "complete":
            template = cls._create_complete_template()
            default_filename = "cardamom_config_complete.yaml"
        elif template_type == "minimal":
            template = cls._create_minimal_template()
            default_filename = "cardamom_config_minimal.yaml"
        elif template_type == "global_only":
            template = cls._create_global_monthly_template()
            default_filename = "cardamom_config_global_monthly.yaml"
        elif template_type == "diurnal_only":
            template = cls._create_diurnal_template()
            default_filename = "cardamom_config_conus_diurnal.yaml"
        else:
            raise ValueError(f"Unknown template type: {template_type}")

        # Generate template content
        template_content = cls._format_template_with_comments(template, template_type)

        # Save to file if path specified
        if output_path:
            output_file = Path(output_path)
        else:
            output_file = Path(default_filename)

        try:
            import yaml
            with open(output_file, 'w') as f:
                f.write(template_content)
        except ImportError:
            raise ImportError("PyYAML is required for template generation. Install with: pip install pyyaml")

        return str(output_file)

    @classmethod
    def _create_complete_template(cls) -> Dict[str, Any]:
        """Create complete configuration template with all options"""
        instance = cls()  # Create temporary instance to get defaults
        return instance._config

    @classmethod
    def _create_minimal_template(cls) -> Dict[str, Any]:
        """Create minimal configuration template with essential settings only"""
        return {
            'processing': {
                'output_directory': "./DATA/CARDAMOM-MAPS_05deg_MET/",
                'log_level': 'INFO'
            },
            'global_monthly': {
                'resolution': 0.5,
                'variables': {
                    'era5': ["2m_temperature", "total_precipitation"]
                }
            },
            'downloaders': {
                'era5': {
                    'api_timeout': 3600
                }
            }
        }

    @classmethod
    def _create_global_monthly_template(cls) -> Dict[str, Any]:
        """Create template focused on global monthly processing"""
        return {
            'processing': {
                'output_directory': "./DATA/CARDAMOM-MAPS_GLOBAL/",
                'temp_directory': "./temp/",
                'compression': True,
                'log_level': 'INFO'
            },
            'global_monthly': {
                'resolution': 0.5,
                'grid_bounds': [-89.75, -179.75, 89.75, 179.75],
                'variables': {
                    'era5': ["2m_temperature", "2m_dewpoint_temperature", "total_precipitation",
                            "skin_temperature", "surface_solar_radiation_downwards"],
                    'gfed': ["burned_area", "fire_carbon"]
                },
                'time_aggregation': 'monthly'
            },
            'downloaders': {
                'era5': {
                    'api_timeout': 3600,
                    'max_concurrent': 3
                },
                'gfed': {
                    'default_years': [2001, 2024]
                }
            },
            'quality_control': {
                'enable_validation': True,
                'missing_data_tolerance': 0.05
            }
        }

    @classmethod
    def _create_diurnal_template(cls) -> Dict[str, Any]:
        """Create template focused on CONUS diurnal processing"""
        return {
            'processing': {
                'output_directory': "./DATA/CARDAMOM-MAPS_CONUS/",
                'temp_directory': "./temp/",
                'compression': True,
                'log_level': 'INFO'
            },
            'conus_diurnal': {
                'resolution': 0.5,
                'region': [60, -130, 20, -50],  # CONUS bounds
                'variables': {
                    'cms': ["GPP", "NEE", "REC", "FIR"],
                    'era5': ["skin_temperature", "surface_solar_radiation_downwards"],
                    'gfed': ["diurnal_fire_pattern"]
                },
                'years_range': [2015, 2020],
                'time_aggregation': 'hourly',
                'diurnal_hours': list(range(24))
            },
            'downloaders': {
                'era5': {
                    'api_timeout': 3600,
                    'max_concurrent': 2
                }
            },
            'quality_control': {
                'enable_validation': True,
                'missing_data_tolerance': 0.10
            }
        }

    @classmethod
    def _format_template_with_comments(cls, template_dict: Dict[str, Any], template_type: str) -> str:
        """Format template dictionary as YAML with helpful comments"""

        header_comments = {
            'complete': """# CARDAMOM Preprocessor - Complete Configuration Template
# This template includes all available configuration options with defaults.
# Customize the values below for your specific processing needs.
#
# Environment variables can override any setting:
#   CARDAMOM_OUTPUT_DIR, CARDAMOM_LOG_LEVEL, ECMWF_CDS_UID, ECMWF_CDS_KEY, etc.
#
# For more information, see: docs/configuration.md

""",
            'minimal': """# CARDAMOM Preprocessor - Minimal Configuration Template
# This template includes only essential settings for basic processing.
# Many options will use built-in defaults.

""",
            'global_only': """# CARDAMOM Preprocessor - Global Monthly Processing Template
# This template is optimized for global monthly meteorological data processing.
# Use this for creating CARDAMOM global drivers from ERA5 and other sources.

""",
            'diurnal_only': """# CARDAMOM Preprocessor - CONUS Diurnal Processing Template
# This template is optimized for CONUS diurnal flux processing.
# Use this for creating hourly carbon flux data from monthly CMS fluxes.

"""
        }

        # Convert to YAML with comments
        import yaml
        yaml_content = yaml.dump(template_dict, default_flow_style=False, indent=2, sort_keys=False)

        # Add section comments based on template type
        if template_type == 'complete':
            yaml_content = cls._add_complete_template_comments(yaml_content)
        elif template_type in ['global_only', 'diurnal_only']:
            yaml_content = cls._add_workflow_specific_comments(yaml_content, template_type)

        return header_comments.get(template_type, '') + yaml_content

    @classmethod
    def _add_complete_template_comments(cls, yaml_content: str) -> str:
        """Add detailed comments to complete template"""
        lines = yaml_content.split('\n')
        commented_lines = []

        for line in lines:
            if line.strip().startswith('processing:'):
                commented_lines.append('# Processing configuration - controls output directories, logging, and performance')
            elif line.strip().startswith('global_monthly:'):
                commented_lines.append('\n# Global monthly workflow - for processing ERA5, GFED, and other global datasets')
            elif line.strip().startswith('conus_diurnal:'):
                commented_lines.append('\n# CONUS diurnal workflow - for processing hourly carbon fluxes over continental US')
            elif line.strip().startswith('downloaders:'):
                commented_lines.append('\n# Downloader configurations - API settings and authentication')
            elif line.strip().startswith('quality_control:'):
                commented_lines.append('\n# Quality control settings - data validation and QA reporting')

            commented_lines.append(line)

        return '\n'.join(commented_lines)

    @classmethod
    def _add_workflow_specific_comments(cls, yaml_content: str, template_type: str) -> str:
        """Add workflow-specific comments"""
        lines = yaml_content.split('\n')
        commented_lines = []

        workflow_comments = {
            'global_only': {
                'resolution:': '  # Grid resolution in degrees (0.25, 0.5, or 1.0)',
                'grid_bounds:': '  # Global bounds: [South, West, North, East] in degrees',
                'era5:': '    # ERA5 meteorological variables to download',
                'gfed:': '    # GFED fire emissions variables to process'
            },
            'diurnal_only': {
                'region:': '  # CONUS bounds: [North, West, South, East] in degrees',
                'years_range:': '  # Years to process for diurnal analysis',
                'cms:': '    # CMS carbon flux variables for diurnal processing',
                'diurnal_hours:': '  # Hours of day to process (0-23)'
            }
        }

        comments = workflow_comments.get(template_type, {})

        for line in lines:
            commented_lines.append(line)
            for key, comment in comments.items():
                if line.strip().endswith(key):
                    commented_lines.append(comment)

        return '\n'.join(commented_lines)