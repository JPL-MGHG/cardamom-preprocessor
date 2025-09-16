# CARDAMOM Configuration Hierarchy Documentation

## Overview

This document describes the unified configuration system implemented in Phase 1 that addresses the configuration overlap identified in the architectural review. The `CardamomConfig` class provides centralized configuration management with a clear hierarchy and eliminates redundant configuration classes across phases.

## Configuration Hierarchy

The configuration system follows a strict precedence order where later sources override earlier ones:

```
1. Built-in Defaults (lowest priority)
   ↓
2. Configuration Files (YAML/JSON)
   ↓
3. Environment Variables
   ↓
4. Command-line Arguments (highest priority)
```

### 1. Built-in Defaults

The system starts with comprehensive defaults defined in `CardamomConfig._get_default_config()`. These ensure the system can run without any external configuration.

**Key Default Sections:**
- `processing`: Core processing settings (output dirs, workers, compression, logging)
- `pipeline`: Pipeline behavior (resumability, cleanup, parallel execution)
- `global_monthly`: Global monthly workflow configuration
- `conus_diurnal`: CONUS diurnal workflow configuration
- `downloaders`: Data source-specific settings (timeouts, authentication placeholders)
- `quality_control`: Validation and QA settings

### 2. Configuration Files

YAML or JSON files override defaults. The system automatically detects file format by extension.

**Standard Locations Searched:**
- `./config/cardamom_processing.yaml` (primary)
- `./cardamom_config.yaml`
- `~/.cardamom/config.yaml`
- `/etc/cardamom/config.yaml`

**File Format:**
```yaml
# Example configuration file
processing:
  output_directory: "./custom/output/"
  max_workers: 8
  log_level: "DEBUG"

global_monthly:
  resolution: 0.25  # Override default 0.5

downloaders:
  era5:
    max_concurrent: 5
```

### 3. Environment Variables

Environment variables provide runtime configuration, particularly useful for secrets and deployment-specific settings.

**Supported Environment Variables:**
```bash
# Processing settings
CARDAMOM_OUTPUT_DIR       → processing.output_directory
CARDAMOM_TEMP_DIR         → processing.temp_directory
CARDAMOM_LOG_LEVEL        → processing.log_level
CARDAMOM_MAX_WORKERS      → processing.max_workers

# Authentication credentials
ECMWF_CDS_UID            → downloaders.era5.username
ECMWF_CDS_KEY            → downloaders.era5.api_key
NOAA_FTP_USER            → downloaders.noaa.username
NOAA_FTP_PASS            → downloaders.noaa.password
GFED_USERNAME            → downloaders.gfed.username
GFED_PASSWORD            → downloaders.gfed.password
EARTHDATA_USERNAME       → downloaders.modis.earthdata_username
EARTHDATA_PASSWORD       → downloaders.modis.earthdata_password
```

### 4. Command-line Arguments

CLI arguments have the highest priority and immediately override any other setting.

**CLI Override Examples:**
```bash
# These CLI args override config file and defaults
python cardamom_cli.py pipeline global-monthly \
    --output-dir /custom/output \
    --workers 12 \
    --verbose

# Translates to configuration overrides:
# processing.output_directory = "/custom/output"
# processing.max_workers = 12
# processing.log_level = "DEBUG"
```

## Configuration Access Patterns

### Dot Notation Access

The unified system supports dot notation for accessing nested configuration values:

```python
config = CardamomConfig()

# Access nested values
output_dir = config.get('processing.output_directory')
era5_timeout = config.get('downloaders.era5.api_timeout')
missing_tolerance = config.get('quality_control.missing_data_tolerance', 0.05)
```

### Section-based Access

Components can retrieve their relevant configuration sections:

```python
# Pipeline manager gets its configuration
pipeline_config = config.get_pipeline_config()
resume_enabled = pipeline_config.get('resume_on_failure', True)

# Downloaders get their specific configs
era5_config = config.get_downloader_config('era5')
max_concurrent = era5_config.get('max_concurrent', 3)

# Workflows get their configurations
global_monthly_config = config.get_workflow_config('global_monthly')
resolution = global_monthly_config.get('resolution')
```

## Integration Across Phases

### Phase 1: Core Framework
- **Role**: Defines and implements the unified configuration system
- **Usage**: All other phases import and use `CardamomConfig`
- **File**: `config_manager.py`

### Phase 2: Downloaders
```python
from config_manager import CardamomConfig

class ECMWFDownloader:
    def __init__(self, config_system):
        era5_config = config_system.get_downloader_config('era5')
        self.api_timeout = era5_config.get('api_timeout', 3600)
        self.max_concurrent = era5_config.get('max_concurrent', 3)
```

### Phase 3 & 4: Processors
```python
class GFEDProcessor:
    def __init__(self, config_system):
        processing_config = config_system.get_processing_config()
        self.output_dir = processing_config.get('output_directory')
        self.compression = processing_config.get('compression', True)
```

### Phase 6: Pipeline Manager
```python
class CARDAMOMPipelineManager:
    def __init__(self, config_file=None, cli_args=None):
        # Central configuration system
        self.config = CardamomConfig(config_file, cli_args)

        # Pass relevant sections to components
        self.downloader_manager = DownloaderManager(self.config.get('downloaders'))
        self.processor_manager = ProcessorManager(self.config.get_processing_config())
```

### Phase 7: CLI
```python
class CARDAMOMCLIProcessor:
    def process_command(self, args):
        # Convert CLI args and initialize unified config
        cli_config = self._convert_args_to_config(args)
        self.config_system = CardamomConfig(
            config_file=getattr(args, 'config_file', None),
            cli_args=cli_config
        )
```

## Configuration Validation

### Automatic Validation

The configuration system performs automatic validation during initialization:

1. **Required Sections**: Ensures all required configuration sections exist
2. **Directory Creation**: Creates output and temp directories if they don't exist
3. **Type Conversion**: Converts string environment variables to appropriate types
4. **Path Expansion**: Expands `~` and environment variables in paths

### Custom Validation

Components can add their own validation:

```python
def validate_downloader_config(config):
    """Custom validation for downloader configuration"""
    era5_config = config.get_downloader_config('era5')

    if era5_config.get('max_concurrent', 0) > 10:
        raise ValueError("ERA5 max_concurrent should not exceed 10")

    if not era5_config.get('api_timeout'):
        raise ValueError("ERA5 api_timeout must be specified")
```

## Migration Benefits

### Before (Multiple Configuration Systems)
- **Phase 1**: `ConfigManager`
- **Phase 6**: `PipelineConfig`
- **Phase 7**: `ConfigCLIManager`
- **Issues**: Duplication, inconsistent hierarchy, complex integration

### After (Unified System)
- **Single Class**: `CardamomConfig` handles all configuration needs
- **Clear Hierarchy**: Defaults → File → Environment → CLI
- **Consistent Access**: Dot notation and section-based access patterns
- **Automatic Validation**: Built-in validation with extensibility
- **Simplified Integration**: All phases use the same configuration interface

## Example Usage Scenarios

### Development with Config File
```bash
# Create custom config
echo "processing:
  log_level: DEBUG
  max_workers: 2
global_monthly:
  resolution: 0.25" > dev_config.yaml

# Run with config file
python cardamom_cli.py --config dev_config.yaml pipeline global-monthly
```

### Production with Environment Variables
```bash
# Set production environment
export CARDAMOM_OUTPUT_DIR="/data/cardamom/output"
export CARDAMOM_LOG_LEVEL="INFO"
export ECMWF_CDS_UID="your_uid"
export ECMWF_CDS_KEY="your_key"

# Run without config file (uses env vars)
python cardamom_cli.py pipeline global-monthly --years 2020-2022
```

### Testing with CLI Overrides
```bash
# Override everything for testing
python cardamom_cli.py \
    --config production.yaml \
    --output-dir ./test_output \
    --workers 1 \
    --verbose \
    pipeline global-monthly --years 2020 --months 1-3
```

## Configuration Schema Reference

The complete configuration schema with all available options is documented in the `CardamomConfig._get_default_config()` method in Phase 1. Key sections include:

- **processing**: Core processing settings
- **pipeline**: Pipeline management behavior
- **global_monthly**: Global monthly workflow settings
- **conus_diurnal**: CONUS diurnal workflow settings
- **downloaders**: Data source configurations
- **quality_control**: Validation settings

For the most up-to-date schema, refer to the default configuration in `config_manager.py`.