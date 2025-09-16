# Phase 7: Single-Task CLI Operations

## Overview
Extend the existing CLI infrastructure to support individual CARDAMOM component operations designed for MAAP job execution. Each CLI command performs a single, specific task without complex orchestration. Maintain backward compatibility while adding component-specific operations.

## 7.1 Enhanced Main CLI (`cardamom_cli.py`)

### Single-Task CLI Structure
```python
class CARDAMOMCLIManager:
    """
    CLI manager for individual CARDAMOM component operations.
    Each command performs a single task suitable for MAAP job execution.
    """

    def __init__(self):
        self.components = CARDAMOMComponents()
        self.existing_ecmwf_cli = self._import_existing_cli()

    def create_master_parser(self):
        """
        Create argument parser for individual component operations.
        """
        parser = argparse.ArgumentParser(
            description="CARDAMOM Data Preprocessing Components",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._get_usage_examples()
        )

        # Add global options
        parser.add_argument('--config', '-c',
                          help='Configuration file path (default: config/cardamom_config.yaml)')
        parser.add_argument('--output-dir', '-o', default='./output/',
                          help='Output directory for processed data (default: ./output/)')
        parser.add_argument('--verbose', '-v', action='count', default=0,
                          help='Increase verbosity (use -v, -vv, or -vvv)')
        parser.add_argument('--log-file',
                          help='Log file path (default: logs/cardamom.log)')
        parser.add_argument('--dry-run', action='store_true',
                          help='Show what would be done without executing')

        # Create subcommands for individual operations
        subparsers = parser.add_subparsers(dest='command', help='Available operations')

        # Legacy ECMWF commands (maintain backward compatibility)
        self._add_legacy_ecmwf_commands(subparsers)

        # Individual component operations
        self._add_download_commands(subparsers)
        self._add_processing_commands(subparsers)
        self._add_output_commands(subparsers)
        self._add_utility_commands(subparsers)

        return parser

    def _get_usage_examples(self):
        """Generate usage examples for individual component operations"""
        return """
Examples:

Legacy ECMWF Downloads (backward compatibility):
  %(prog)s ecmwf hourly -v skin_temperature -y 2020 -m 6-8 --area 60,-130,20,-50
  %(prog)s ecmwf monthly -v 2m_temperature -y 2020-2022 -m 1-12
  %(prog)s ecmwf cardamom-monthly -y 2020 -o ./monthly_data/

Individual Download Operations (for MAAP jobs):
  %(prog)s download-era5 -y 2020 -m 1-3 -v 2m_temperature,2m_dewpoint_temperature
  %(prog)s download-noaa-co2 -y 2020-2022
  %(prog)s download-gfed -y 2020
  %(prog)s download-modis-mask --resolution 0.5deg

Individual Processing Operations (for MAAP jobs):
  %(prog)s process-gfed -y 2020
  %(prog)s process-diurnal-conus -y 2020 -m 1-3 --experiment 1
  %(prog)s calculate-vpd -y 2020 -m 1-3

Individual Output Operations (for MAAP jobs):
  %(prog)s generate-netcdf temperature -y 2020 -m 1-3
  %(prog)s generate-netcdf monthly-global -y 2020

Utility Operations:
  %(prog)s validate-outputs ./output/
  %(prog)s check-component era5
        """
```

### Individual Download Commands
```python
def _add_download_commands(self, subparsers):
    """Add individual download operation commands"""

    # ERA5 download
    era5_parser = subparsers.add_parser('download-era5',
                                       help='Download ERA5 data for specific variables and time range')
    era5_parser.add_argument('-y', '--years', required=True,
                           help='Years to download (e.g., 2020, 2020-2022)')
    era5_parser.add_argument('-m', '--months', default='1-12',
                           help='Months to download (e.g., 1-3, 6,7,8)')
    era5_parser.add_argument('-v', '--variables', required=True,
                           help='Variables to download (comma-separated)')
    era5_parser.add_argument('--area',
                           help='Area bounds as N,W,S,E (default: global)')

    # NOAA CO2 download
    noaa_parser = subparsers.add_parser('download-noaa-co2',
                                       help='Download NOAA CO2 concentration data')
    noaa_parser.add_argument('-y', '--years', required=True,
                           help='Years to download')
    noaa_parser.add_argument('--force-update', action='store_true',
                           help='Force download even if cached data exists')

    # GFED download
    gfed_parser = subparsers.add_parser('download-gfed',
                                       help='Download GFED fire data for specific year')
    gfed_parser.add_argument('-y', '--year', type=int, required=True,
                           help='Year to download')

    # MODIS land-sea mask download
    modis_parser = subparsers.add_parser('download-modis-mask',
                                        help='Download MODIS land-sea mask')
    modis_parser.add_argument('--resolution', default='0.5deg',
                            help='Spatial resolution (default: 0.5deg)')

def _add_processing_commands(self, subparsers):
    """Add individual processing operation commands"""

    # GFED processing
    gfed_proc_parser = subparsers.add_parser('process-gfed',
                                           help='Process GFED data for specific year')
    gfed_proc_parser.add_argument('-y', '--year', type=int, required=True,
                                help='Year to process')
    gfed_proc_parser.add_argument('--gap-fill', action='store_true',
                                help='Apply gap-filling for missing data')

    # Diurnal processing
    diurnal_parser = subparsers.add_parser('process-diurnal-conus',
                                         help='Process CONUS diurnal fluxes')
    diurnal_parser.add_argument('-y', '--year', type=int, required=True,
                              help='Year to process')
    diurnal_parser.add_argument('-m', '--months', default='1-12',
                              help='Months to process')
    diurnal_parser.add_argument('--experiment', type=int, default=1,
                              help='CMS experiment number')

    # VPD calculation
    vpd_parser = subparsers.add_parser('calculate-vpd',
                                     help='Calculate VPD from ERA5 temperature and dewpoint')
    vpd_parser.add_argument('-y', '--year', type=int, required=True,
                          help='Year to process')
    vpd_parser.add_argument('-m', '--months', default='1-12',
                          help='Months to process')

def _add_output_commands(self, subparsers):
    """Add individual output generation commands"""

    # NetCDF generation
    netcdf_parser = subparsers.add_parser('generate-netcdf',
                                        help='Generate NetCDF files for specific variable group')
    netcdf_parser.add_argument('variable_group',
                             choices=['temperature', 'radiation', 'precipitation', 'atmospheric', 'fire'],
                             help='Variable group to generate')
    netcdf_parser.add_argument('-y', '--year', type=int, required=True,
                             help='Year to generate')
    netcdf_parser.add_argument('-m', '--months', default='1-12',
                             help='Months to generate')

### Legacy Command Integration
```python
def _add_legacy_ecmwf_commands(self, subparsers):
    """
    Add existing ECMWF commands for backward compatibility.
    Wraps existing functionality from ecmwf_downloader.py
    """

    # Main ECMWF command group
    ecmwf_parser = subparsers.add_parser('ecmwf',
                                         help='ECMWF ERA5 data downloads (legacy commands)')
    ecmwf_subparsers = ecmwf_parser.add_subparsers(dest='ecmwf_command')

    # Import existing parser structure from ecmwf_downloader.py
    existing_parser = self.existing_ecmwf_cli.create_parser()

    # Extract and adapt existing subcommands
    for action in existing_parser._subparsers._actions:
        if isinstance(action, argparse._SubParsersAction):
            for choice, subparser in action.choices.items():
                # Clone existing subparser structure
                new_subparser = ecmwf_subparsers.add_parser(
                    choice,
                    help=subparser.description,
                    parents=[subparser],
                    conflict_handler='resolve'
                )

def _execute_legacy_ecmwf_command(self, args):
    """Execute legacy ECMWF command using existing implementation"""

    # Reconstruct arguments for existing CLI
    legacy_args = [args.ecmwf_command]

    # Map arguments back to legacy format
    if hasattr(args, 'variables'):
        legacy_args.extend(['-v'] + args.variables)
    if hasattr(args, 'years'):
        legacy_args.extend(['-y', self._format_years_for_legacy(args.years)])
    if hasattr(args, 'months'):
        legacy_args.extend(['-m', self._format_months_for_legacy(args.months)])

    # Execute using existing main function
    return self.existing_ecmwf_cli.main_with_args(legacy_args)
```

## 7.2 Utility Commands (`utility_commands.py`)

### Simple Utility Commands
```python
def _add_utility_commands(self, subparsers):
    """Add simple utility commands for individual operations"""

    # Validate outputs
    validate_parser = subparsers.add_parser('validate-outputs',
                                          help='Validate generated output files')
    validate_parser.add_argument('output_dir',
                               help='Directory containing output files to validate')
    validate_parser.add_argument('--variable-group',
                               help='Specific variable group to validate')

    # Check component status
    check_parser = subparsers.add_parser('check-component',
                                       help='Check if component dependencies are available')
    check_parser.add_argument('component',
                            choices=['era5', 'noaa', 'gfed', 'modis'],
                            help='Component to check')

    # List available operations
    list_parser = subparsers.add_parser('list-operations',
                                      help='List all available operations')
    list_parser.add_argument('--component',
                           help='Filter by specific component type')

def execute_utility_command(self, args):
    """Execute utility commands"""

    if args.command == 'validate-outputs':
        return self._validate_outputs(args)
    elif args.command == 'check-component':
        return self._check_component(args)
    elif args.command == 'list-operations':
        return self._list_operations(args)

def _validate_outputs(self, args):
    """Validate output files"""
    # Simple validation of output file existence and basic structure
    import os
    import netCDF4

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        print(f"Error: Output directory {output_dir} does not exist")
        return False

    nc_files = [f for f in os.listdir(output_dir) if f.endswith('.nc')]
    print(f"Found {len(nc_files)} NetCDF files in {output_dir}")

    for file_path in nc_files:
        full_path = os.path.join(output_dir, file_path)
        try:
            with netCDF4.Dataset(full_path, 'r') as nc:
                print(f"✓ {file_path}: valid NetCDF file")
        except Exception as e:
            print(f"✗ {file_path}: {e}")

    return True

def _check_component(self, args):
    """Check component dependencies"""
    component = args.component

    if component == 'era5':
        # Check ECMWF CDS API credentials
        try:
            import cdsapi
            print("✓ cdsapi library available")
            # Check for credentials file
            import os
            if os.path.exists(os.path.expanduser('~/.cdsapirc')):
                print("✓ CDS API credentials found")
            else:
                print("✗ CDS API credentials not found (~/.cdsapirc)")
        except ImportError:
            print("✗ cdsapi library not available")

    elif component == 'gfed':
        # Check HDF5 support
        try:
            import h5py
            print("✓ h5py library available")
        except ImportError:
            print("✗ h5py library not available")

    # Add other component checks for other components
    return True
```

## 7.3 Command Execution (`command_execution.py`)

### Individual Operation Execution
```python
def execute_component_operation(self, args):
    """Execute individual component operations"""

    if args.command.startswith('download-'):
        return self._execute_download_operation(args)
    elif args.command.startswith('process-'):
        return self._execute_processing_operation(args)
    elif args.command.startswith('generate-'):
        return self._execute_output_operation(args)
    elif args.command.startswith('calculate-'):
        return self._execute_calculation_operation(args)
    else:
        return self.execute_utility_command(args)

def _execute_download_operation(self, args):
    """Execute download operations"""

    if args.command == 'download-era5':
        return self.components.download_era5_data(
            variables=args.variables.split(','),
            years=self._parse_years(args.years),
            months=self._parse_months(args.months)
        )
    elif args.command == 'download-noaa-co2':
        return self.components.download_noaa_co2(
            years=self._parse_years(args.years)
        )
    elif args.command == 'download-gfed':
        return self.components.download_gfed_data(args.year)
    elif args.command == 'download-modis-mask':
        return self.components.download_modis_land_sea_mask(args.resolution)

def _execute_processing_operation(self, args):
    """Execute processing operations"""

    if args.command == 'process-gfed':
        return self.components.process_gfed_year(args.year)
    elif args.command == 'process-diurnal-conus':
        return self.components.process_diurnal_conus(
            args.year,
            self._parse_months(args.months),
            [args.experiment]
        )

def _execute_calculation_operation(self, args):
    """Execute calculation operations"""

    if args.command == 'calculate-vpd':
        return self.components.calculate_vpd_from_era5(
            args.year,
            self._parse_months(args.months)
        )

def _execute_output_operation(self, args):
    """Execute output generation operations"""

    if args.command == 'generate-netcdf':
        return self.components.generate_monthly_netcdf(
            args.variable_group,
            args.year,
            self._parse_months(args.months)
        )
```

## 7.4 Testing Framework

### Individual Operation Tests
```
tests/cli/
├── test_individual_operations.py
├── test_download_commands.py
├── test_processing_commands.py
├── test_output_commands.py
├── test_utility_commands.py
└── fixtures/
    ├── sample_configs/
    ├── mock_responses/
    └── test_outputs/
```

### Individual Operation Tests
```python
def test_download_era5_command():
    """Test ERA5 download command execution"""

def test_process_gfed_command():
    """Test GFED processing command execution"""

def test_calculate_vpd_command():
    """Test VPD calculation command execution"""

def test_utility_commands():
    """Test utility command execution"""
```

## 7.5 Success Criteria

### Functional Requirements
- [ ] Maintain complete backward compatibility with existing ECMWF CLI
- [ ] Support individual component operations without orchestration
- [ ] Provide clear single-task commands suitable for MAAP jobs
- [ ] Include basic validation and utility capabilities

### Simplicity Requirements
- [ ] Remove complex pipeline management commands
- [ ] Simple, single-purpose command structure
- [ ] No internal coordination or state management in CLI
- [ ] Clear separation between individual operations

### Integration Requirements
- [ ] Seamless integration with existing ECMWF downloader
- [ ] Consistent configuration management with Phase 1
- [ ] Compatible with MAAP platform job execution
- [ ] Support for external job submission and management

### Quality Requirements
- [ ] Clear error handling and user feedback for individual operations
- [ ] Simple help documentation and usage examples
- [ ] Testing framework for individual component operations
