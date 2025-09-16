# Phase 6: Unified Processing Pipeline

## Overview
Create a comprehensive pipeline management system that orchestrates all downloaders, processors, and writers into unified workflows. Supports both global monthly processing and CONUS diurnal flux processing with robust error handling and progress tracking.

## 6.1 Core Pipeline Manager (`pipeline_manager.py`)

### Main Pipeline Orchestrator
```python
class CARDAMOMPipelineManager:
    """
    Main orchestrator for CARDAMOM preprocessing workflows.
    Coordinates multiple data sources, processors, and output formats.
    """

    def __init__(self, config_file=None, workspace_dir="./workspace/"):
        self.config = self._load_pipeline_config(config_file)
        self.workspace_dir = workspace_dir
        self.logger = self._setup_logging()

        # Initialize component managers
        self.downloader_manager = DownloaderManager(self.config)
        self.processor_manager = ProcessorManager(self.config)
        self.output_manager = OutputManager(self.config)

        # Pipeline state tracking
        self.pipeline_state = PipelineState()
        self.execution_history = []

    def execute_global_monthly_pipeline(self, years, months=None, resume=True):
        """
        Execute complete global monthly preprocessing pipeline.

        Pipeline stages:
        1. Download ERA5, NOAA CO2, GFED, MODIS data
        2. Process individual datasets
        3. Create CARDAMOM-compliant NetCDF outputs
        4. Generate quality reports and summaries
        """

        pipeline_id = f"global_monthly_{'-'.join(map(str, years))}"
        self.logger.info(f"Starting global monthly pipeline: {pipeline_id}")

        try:
            # Stage 1: Data Download
            download_status = self._execute_download_stage(years, months, 'global_monthly')

            # Stage 2: Data Processing
            processing_status = self._execute_processing_stage(years, months, 'global_monthly')

            # Stage 3: Output Generation
            output_status = self._execute_output_stage(years, months, 'global_monthly')

            # Stage 4: Quality Control and Reporting
            qa_status = self._execute_qa_stage(pipeline_id)

            self.logger.info(f"Global monthly pipeline completed: {pipeline_id}")
            return self._create_pipeline_summary(pipeline_id, [download_status, processing_status, output_status, qa_status])

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            self._handle_pipeline_failure(pipeline_id, e)
            raise

    def execute_conus_diurnal_pipeline(self, years, months=None, experiment_numbers=[1, 2]):
        """
        Execute CONUS diurnal flux processing pipeline.

        Pipeline stages:
        1. Load CMS monthly flux data
        2. Download ERA5 hourly and GFED diurnal data
        3. Perform diurnal downscaling
        4. Generate hourly and monthly output files
        """

        pipeline_id = f"conus_diurnal_{'-'.join(map(str, years))}"
        self.logger.info(f"Starting CONUS diurnal pipeline: {pipeline_id}")

        try:
            # Stage 1: CMS Data Loading
            cms_status = self._load_cms_data(years, experiment_numbers)

            # Stage 2: Meteorological and Fire Data
            met_fire_status = self._download_met_fire_data(years, months)

            # Stage 3: Diurnal Processing
            diurnal_status = self._execute_diurnal_processing(years, months, experiment_numbers)

            # Stage 4: Output and Validation
            output_status = self._execute_diurnal_outputs(years, months, experiment_numbers)

            self.logger.info(f"CONUS diurnal pipeline completed: {pipeline_id}")
            return self._create_pipeline_summary(pipeline_id, [cms_status, met_fire_status, diurnal_status, output_status])

        except Exception as e:
            self.logger.error(f"Diurnal pipeline failed: {e}")
            self._handle_pipeline_failure(pipeline_id, e)
            raise
```

### Pipeline Stage Execution
```python
def _execute_download_stage(self, years, months, workflow_type):
    """Execute data download stage for specified workflow"""

    self.logger.info("Starting download stage")
    download_tasks = self._create_download_tasks(years, months, workflow_type)

    download_results = {}

    for source, tasks in download_tasks.items():
        self.logger.info(f"Downloading from {source}")

        try:
            downloader = self.downloader_manager.get_downloader(source)
            download_results[source] = downloader.execute_tasks(tasks)
            self.logger.info(f"Successfully downloaded {source} data")

        except Exception as e:
            self.logger.error(f"Download failed for {source}: {e}")
            download_results[source] = {'status': 'failed', 'error': str(e)}

            # Check if this is a critical failure
            if self._is_critical_source(source, workflow_type):
                raise RuntimeError(f"Critical download failure: {source}")

    return download_results

def _execute_processing_stage(self, years, months, workflow_type):
    """Execute data processing stage"""

    self.logger.info("Starting processing stage")
    processing_results = {}

    # Process each data source
    for source in self._get_required_sources(workflow_type):
        self.logger.info(f"Processing {source} data")

        try:
            processor = self.processor_manager.get_processor(source)
            processing_results[source] = processor.process_data(years, months)
            self.logger.info(f"Successfully processed {source} data")

        except Exception as e:
            self.logger.error(f"Processing failed for {source}: {e}")
            processing_results[source] = {'status': 'failed', 'error': str(e)}

    return processing_results

def _execute_output_stage(self, years, months, workflow_type):
    """Execute output generation stage"""

    self.logger.info("Starting output generation stage")
    output_results = {}

    # Generate outputs for each variable type
    for variable_group in self._get_output_variables(workflow_type):
        self.logger.info(f"Generating outputs for {variable_group}")

        try:
            output_results[variable_group] = self.output_manager.create_outputs(
                variable_group, years, months, workflow_type
            )
            self.logger.info(f"Successfully created {variable_group} outputs")

        except Exception as e:
            self.logger.error(f"Output generation failed for {variable_group}: {e}")
            output_results[variable_group] = {'status': 'failed', 'error': str(e)}

    return output_results
```

## 6.2 Component Managers (`component_managers.py`)

### Downloader Manager
```python
class DownloaderManager:
    """Manage multiple data source downloaders"""

    def __init__(self, config):
        self.config = config
        self.downloaders = self._initialize_downloaders()
        self.executor = ThreadPoolExecutor(max_workers=4)

    def _initialize_downloaders(self):
        """Initialize all downloader instances"""
        from ..downloaders import ECMWFDownloader, NOAADownloader, GFEDDownloader, MODISDownloader

        return {
            'ecmwf': ECMWFDownloader(
                output_dir=self.config['paths']['ecmwf_data']
            ),
            'noaa': NOAADownloader(
                output_dir=self.config['paths']['noaa_data']
            ),
            'gfed': GFEDDownloader(
                output_dir=self.config['paths']['gfed_data']
            ),
            'modis': MODISDownloader(
                output_dir=self.config['paths']['modis_data']
            )
        }

    def get_downloader(self, source):
        """Get downloader instance for specified source"""
        if source not in self.downloaders:
            raise ValueError(f"Unknown data source: {source}")
        return self.downloaders[source]

    def execute_parallel_downloads(self, download_plan):
        """Execute multiple downloads in parallel"""

        futures = {}

        for source, tasks in download_plan.items():
            if tasks:  # Only submit if there are tasks
                future = self.executor.submit(
                    self._execute_source_downloads, source, tasks
                )
                futures[source] = future

        # Collect results
        results = {}
        for source, future in futures.items():
            try:
                results[source] = future.result(timeout=3600)  # 1 hour timeout
            except Exception as e:
                results[source] = {'status': 'failed', 'error': str(e)}

        return results

    def _execute_source_downloads(self, source, tasks):
        """Execute downloads for a single source"""
        downloader = self.get_downloader(source)

        results = []
        for task in tasks:
            try:
                result = downloader.download_data(**task)
                results.append({'task': task, 'status': 'success', 'result': result})
            except Exception as e:
                results.append({'task': task, 'status': 'failed', 'error': str(e)})

        return results
```

### Processor Manager
```python
class ProcessorManager:
    """Manage data processing components"""

    def __init__(self, config):
        self.config = config
        self.processors = self._initialize_processors()

    def _initialize_processors(self):
        """Initialize all processor instances"""
        from ..processors import GFEDProcessor, DiurnalProcessor
        from ..core import CARDAMOMProcessor

        return {
            'gfed': GFEDProcessor(
                data_dir=self.config['paths']['gfed_data'],
                output_dir=self.config['paths']['processed_gfed']
            ),
            'diurnal': DiurnalProcessor(
                config_file=self.config.get('diurnal_config')
            ),
            'cardamom': CARDAMOMProcessor(
                output_dir=self.config['paths']['cardamom_output']
            )
        }

    def get_processor(self, processor_type):
        """Get processor instance for specified type"""
        if processor_type not in self.processors:
            raise ValueError(f"Unknown processor type: {processor_type}")
        return self.processors[processor_type]

    def process_data_source(self, source, years, months=None):
        """Process data from specific source"""

        if source == 'gfed':
            return self.processors['gfed'].process_multi_year_data(years)
        elif source == 'era5':
            return self._process_era5_data(years, months)
        elif source == 'noaa':
            return self._process_noaa_co2(years)
        else:
            raise ValueError(f"No processor available for source: {source}")

    def _process_era5_data(self, years, months):
        """Process ERA5 data including VPD calculation"""

        # This would coordinate ERA5 processing including:
        # - Temperature min/max extraction
        # - VPD calculation
        # - Unit conversions
        # - Quality control

        results = {
            'temperature': self._process_era5_temperature(years, months),
            'radiation': self._process_era5_radiation(years, months),
            'precipitation': self._process_era5_precipitation(years, months),
            'vpd': self._calculate_vpd_from_era5(years, months)
        }

        return results
```

### Output Manager
```python
class OutputManager:
    """Manage output file generation and organization"""

    def __init__(self, config):
        self.config = config
        self.netcdf_writer = CARDAMOMNetCDFWriter()
        self.output_structure = self._setup_output_structure()

    def _setup_output_structure(self):
        """Setup standardized output directory structure"""

        base_dir = self.config['paths']['output_base']

        structure = {
            'global_monthly': os.path.join(base_dir, 'CARDAMOM-MAPS_05deg_MET'),
            'conus_diurnal': os.path.join(base_dir, 'CARDAMOM_CONUS_DIURNAL_FLUXES'),
            'quality_reports': os.path.join(base_dir, 'quality_reports'),
            'processing_logs': os.path.join(base_dir, 'logs')
        }

        # Create directories
        for dir_path in structure.values():
            os.makedirs(dir_path, exist_ok=True)

        return structure

    def create_outputs(self, variable_group, years, months, workflow_type):
        """Create output files for variable group"""

        output_results = {}

        for year in years:
            year_results = {}

            if workflow_type == 'global_monthly':
                year_results = self._create_global_monthly_outputs(variable_group, year, months)
            elif workflow_type == 'conus_diurnal':
                year_results = self._create_conus_diurnal_outputs(variable_group, year, months)

            output_results[year] = year_results

        return output_results

    def _create_global_monthly_outputs(self, variable_group, year, months):
        """Create global monthly output files"""

        results = {}

        if variable_group == 'temperature':
            results.update(self._create_temperature_files(year, months))
        elif variable_group == 'radiation':
            results.update(self._create_radiation_files(year, months))
        elif variable_group == 'precipitation':
            results.update(self._create_precipitation_files(year, months))
        elif variable_group == 'atmospheric':
            results.update(self._create_atmospheric_files(year, months))
        elif variable_group == 'fire':
            results.update(self._create_fire_files(year, months))
        elif variable_group == 'land_sea':
            results.update(self._create_land_sea_files())

        return results
```

## 6.3 Pipeline State Management (`pipeline_state.py`)

### State Tracking
```python
class PipelineState:
    """Track pipeline execution state for resumability"""

    def __init__(self, state_file=None):
        self.state_file = state_file or "pipeline_state.json"
        self.state = self._load_state()

    def _load_state(self):
        """Load pipeline state from file"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'pipelines': {},
                'completed_tasks': {},
                'failed_tasks': {},
                'last_update': None
            }

    def save_state(self):
        """Save current state to file"""
        self.state['last_update'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def mark_task_completed(self, pipeline_id, stage, task_id, result):
        """Mark a task as completed"""
        if pipeline_id not in self.state['completed_tasks']:
            self.state['completed_tasks'][pipeline_id] = {}
        if stage not in self.state['completed_tasks'][pipeline_id]:
            self.state['completed_tasks'][pipeline_id][stage] = {}

        self.state['completed_tasks'][pipeline_id][stage][task_id] = {
            'timestamp': datetime.now().isoformat(),
            'result': result
        }
        self.save_state()

    def mark_task_failed(self, pipeline_id, stage, task_id, error):
        """Mark a task as failed"""
        if pipeline_id not in self.state['failed_tasks']:
            self.state['failed_tasks'][pipeline_id] = {}
        if stage not in self.state['failed_tasks'][pipeline_id]:
            self.state['failed_tasks'][pipeline_id][stage] = {}

        self.state['failed_tasks'][pipeline_id][stage][task_id] = {
            'timestamp': datetime.now().isoformat(),
            'error': str(error)
        }
        self.save_state()

    def is_task_completed(self, pipeline_id, stage, task_id):
        """Check if task is already completed"""
        return (pipeline_id in self.state['completed_tasks'] and
                stage in self.state['completed_tasks'][pipeline_id] and
                task_id in self.state['completed_tasks'][pipeline_id][stage])

    def get_resumable_tasks(self, pipeline_id):
        """Get list of tasks that can be resumed"""
        completed = self.state['completed_tasks'].get(pipeline_id, {})
        failed = self.state['failed_tasks'].get(pipeline_id, {})

        return {
            'completed': completed,
            'failed': failed,
            'can_resume': len(failed) == 0  # Can only resume if no failures
        }
```

### Progress Tracking
```python
class ProgressTracker:
    """Track and report pipeline progress"""

    def __init__(self, pipeline_id, total_tasks):
        self.pipeline_id = pipeline_id
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.start_time = datetime.now()
        self.current_stage = None

    def update_progress(self, stage=None, completed=0, failed=0):
        """Update progress counters"""
        if stage:
            self.current_stage = stage
        self.completed_tasks += completed
        self.failed_tasks += failed

    def get_progress_report(self):
        """Generate progress report"""
        elapsed = datetime.now() - self.start_time
        completion_rate = self.completed_tasks / self.total_tasks if self.total_tasks > 0 else 0

        return {
            'pipeline_id': self.pipeline_id,
            'current_stage': self.current_stage,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'total_tasks': self.total_tasks,
            'completion_rate': completion_rate,
            'elapsed_time': str(elapsed),
            'estimated_remaining': self._estimate_remaining_time(elapsed, completion_rate)
        }

    def _estimate_remaining_time(self, elapsed, completion_rate):
        """Estimate remaining processing time"""
        if completion_rate > 0:
            total_estimated = elapsed / completion_rate
            remaining = total_estimated - elapsed
            return str(remaining)
        else:
            return "Unknown"

    def print_progress(self):
        """Print formatted progress report"""
        report = self.get_progress_report()

        print(f"\n=== Pipeline Progress: {report['pipeline_id']} ===")
        print(f"Current Stage: {report['current_stage']}")
        print(f"Progress: {report['completed_tasks']}/{report['total_tasks']} ({report['completion_rate']:.1%})")
        print(f"Failed Tasks: {report['failed_tasks']}")
        print(f"Elapsed Time: {report['elapsed_time']}")
        print(f"Estimated Remaining: {report['estimated_remaining']}")
        print("=" * 50)
```

## 6.4 Error Handling and Recovery (`error_handling.py`)

### Comprehensive Error Handler
```python
class PipelineErrorHandler:
    """Handle errors and implement recovery strategies"""

    def __init__(self, config):
        self.config = config
        self.retry_limits = config.get('retry_limits', {})
        self.recovery_strategies = self._setup_recovery_strategies()

    def _setup_recovery_strategies(self):
        """Setup error recovery strategies for different error types"""
        return {
            'network_error': self._handle_network_error,
            'file_not_found': self._handle_file_not_found,
            'data_corruption': self._handle_data_corruption,
            'memory_error': self._handle_memory_error,
            'processing_error': self._handle_processing_error
        }

    def handle_error(self, error, context):
        """Main error handling dispatcher"""
        error_type = self._classify_error(error)

        if error_type in self.recovery_strategies:
            return self.recovery_strategies[error_type](error, context)
        else:
            return self._handle_unknown_error(error, context)

    def _classify_error(self, error):
        """Classify error type for appropriate handling"""
        if isinstance(error, (requests.exceptions.RequestException, urllib.error.URLError)):
            return 'network_error'
        elif isinstance(error, FileNotFoundError):
            return 'file_not_found'
        elif isinstance(error, MemoryError):
            return 'memory_error'
        elif 'corrupt' in str(error).lower() or 'invalid' in str(error).lower():
            return 'data_corruption'
        else:
            return 'processing_error'

    def _handle_network_error(self, error, context):
        """Handle network-related errors with retry logic"""
        max_retries = self.retry_limits.get('network', 3)
        retry_count = context.get('retry_count', 0)

        if retry_count < max_retries:
            wait_time = 2 ** retry_count  # Exponential backoff
            time.sleep(wait_time)

            return {
                'action': 'retry',
                'retry_count': retry_count + 1,
                'message': f"Retrying after network error (attempt {retry_count + 1}/{max_retries})"
            }
        else:
            return {
                'action': 'fail',
                'message': f"Network error persists after {max_retries} attempts"
            }

    def _handle_file_not_found(self, error, context):
        """Handle missing file errors"""
        missing_file = str(error).split("'")[1] if "'" in str(error) else "unknown"

        # Check if file can be regenerated
        if self._can_regenerate_file(missing_file, context):
            return {
                'action': 'regenerate',
                'file': missing_file,
                'message': f"Attempting to regenerate missing file: {missing_file}"
            }
        else:
            return {
                'action': 'fail',
                'message': f"Required file not found and cannot be regenerated: {missing_file}"
            }

    def _handle_memory_error(self, error, context):
        """Handle memory-related errors"""
        return {
            'action': 'reduce_batch_size',
            'message': "Memory error detected, reducing batch size and retrying"
        }

    def _can_regenerate_file(self, filename, context):
        """Check if a missing file can be regenerated"""
        regenerable_patterns = [
            'intermediate_',
            'temp_',
            'cache_',
            '_processed.nc'
        ]

        return any(pattern in filename for pattern in regenerable_patterns)
```

### Recovery Actions
```python
class RecoveryManager:
    """Implement recovery actions for pipeline failures"""

    def __init__(self, pipeline_manager):
        self.pipeline_manager = pipeline_manager

    def execute_recovery_action(self, action_spec, context):
        """Execute specified recovery action"""

        action = action_spec['action']

        if action == 'retry':
            return self._retry_task(context, action_spec.get('retry_count', 1))
        elif action == 'regenerate':
            return self._regenerate_file(action_spec['file'], context)
        elif action == 'reduce_batch_size':
            return self._reduce_batch_size(context)
        elif action == 'skip_task':
            return self._skip_task(context)
        else:
            raise ValueError(f"Unknown recovery action: {action}")

    def _retry_task(self, context, retry_count):
        """Retry a failed task"""
        context['retry_count'] = retry_count

        # Re-execute the original task with updated context
        return self.pipeline_manager._execute_task(context['task'], context)

    def _regenerate_file(self, filename, context):
        """Attempt to regenerate a missing file"""
        # Implementation would depend on file type and context
        # This is a placeholder for the regeneration logic
        pass

    def _reduce_batch_size(self, context):
        """Reduce batch size and retry processing"""
        current_batch_size = context.get('batch_size', 12)  # Default 12 months
        new_batch_size = max(1, current_batch_size // 2)

        context['batch_size'] = new_batch_size
        return self._retry_task(context, context.get('retry_count', 0))
```

## 6.5 Configuration Management (`pipeline_config.py`)

### Pipeline Configuration
```python
class PipelineConfig:
    """Manage pipeline configuration and settings"""

    def __init__(self, config_file=None):
        self.config_file = config_file or "config/pipeline_config.yaml"
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self):
        """Load configuration from YAML file"""
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Apply environment variable substitutions
        config = self._substitute_env_vars(config)

        return config

    def _substitute_env_vars(self, config):
        """Substitute environment variables in config values"""
        def substitute_recursive(obj):
            if isinstance(obj, dict):
                return {k: substitute_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute_recursive(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith('$'):
                env_var = obj[1:]
                return os.environ.get(env_var, obj)
            else:
                return obj

        return substitute_recursive(config)

    def _validate_config(self):
        """Validate configuration completeness and correctness"""
        required_sections = ['paths', 'processing', 'workflows']

        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Required configuration section missing: {section}")

        # Validate paths exist or can be created
        for path_key, path_value in self.config['paths'].items():
            if not os.path.exists(path_value):
                os.makedirs(path_value, exist_ok=True)

    def get_workflow_config(self, workflow_name):
        """Get configuration for specific workflow"""
        workflows = self.config.get('workflows', {})
        if workflow_name not in workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}")

        return workflows[workflow_name]

    def get_processing_config(self, processor_name):
        """Get configuration for specific processor"""
        processing = self.config.get('processing', {})
        if processor_name not in processing:
            raise ValueError(f"Unknown processor: {processor_name}")

        return processing[processor_name]
```

### Configuration Schema
```yaml
# config/pipeline_config.yaml
paths:
  workspace: "./workspace/"
  output_base: "./DATA/"
  ecmwf_data: "./DATA/ERA5/"
  noaa_data: "./DATA/NOAA_CO2/"
  gfed_data: "./DATA/GFED4/"
  modis_data: "./DATA/MODIS_LSM/"
  processed_gfed: "./DATA/PROCESSED_GFED/"
  cardamom_output: "./DATA/CARDAMOM-MAPS_05deg_MET/"
  quality_reports: "./DATA/quality_reports/"
  logs: "./logs/"

processing:
  global_monthly:
    resolution: 0.5
    years_default: [2001, 2024]
    variables:
      era5: ["2m_temperature", "2m_dewpoint_temperature", "total_precipitation",
             "skin_temperature", "surface_solar_radiation_downwards",
             "surface_thermal_radiation_downwards", "snowfall"]
      noaa: ["co2"]
      gfed: ["burned_area", "fire_carbon"]
      modis: ["land_sea_mask", "land_sea_fraction"]

  conus_diurnal:
    resolution: 0.5
    region: [60, -130, 20, -50]  # N, W, S, E
    years_default: [2015, 2020]
    experiments: [1, 2]
    variables:
      cms: ["GPP", "NEE", "REC", "FIR", "NBE"]
      era5: ["skin_temperature", "surface_solar_radiation_downwards"]
      gfed: ["diurnal_fire_pattern"]

workflows:
  global_monthly:
    stages: ["download", "process", "output", "qa"]
    parallel_downloads: true
    resumable: true

  conus_diurnal:
    stages: ["cms_load", "met_fire_download", "diurnal_process", "output"]
    parallel_processing: true
    resumable: true

error_handling:
  retry_limits:
    network: 3
    processing: 2
    file_io: 2
  recovery_strategies:
    enable_auto_recovery: true
    backup_sources: true

logging:
  level: "INFO"
  file: "./logs/pipeline.log"
  max_size_mb: 100
  backup_count: 5
```

## 6.6 Quality Assurance and Reporting (`qa_reporting.py`)

### Quality Assurance Manager
```python
class QualityAssuranceManager:
    """Manage quality assurance checks and reporting"""

    def __init__(self, config):
        self.config = config
        self.qa_checks = self._setup_qa_checks()
        self.report_generator = ReportGenerator(config)

    def _setup_qa_checks(self):
        """Setup quality assurance check registry"""
        return {
            'file_completeness': self._check_file_completeness,
            'data_ranges': self._check_data_ranges,
            'temporal_continuity': self._check_temporal_continuity,
            'spatial_coverage': self._check_spatial_coverage,
            'metadata_compliance': self._check_metadata_compliance,
            'format_validation': self._check_format_validation
        }

    def run_qa_suite(self, pipeline_id, output_files):
        """Run complete QA suite on pipeline outputs"""

        qa_results = {}

        for check_name, check_function in self.qa_checks.items():
            try:
                qa_results[check_name] = check_function(output_files)
            except Exception as e:
                qa_results[check_name] = {
                    'status': 'error',
                    'message': f"QA check failed: {e}"
                }

        # Generate QA report
        report = self.report_generator.generate_qa_report(pipeline_id, qa_results)

        return {
            'qa_results': qa_results,
            'report_path': report,
            'overall_status': self._determine_overall_status(qa_results)
        }

    def _check_file_completeness(self, output_files):
        """Check if all expected files were created"""
        expected_files = self._get_expected_files()
        missing_files = []

        for expected_file in expected_files:
            if not os.path.exists(expected_file):
                missing_files.append(expected_file)

        return {
            'status': 'pass' if not missing_files else 'fail',
            'missing_files': missing_files,
            'total_expected': len(expected_files),
            'total_found': len(expected_files) - len(missing_files)
        }

    def _check_data_ranges(self, output_files):
        """Check if data values are within expected physical ranges"""

        range_checks = {
            'temperature': {'min': -100, 'max': 60},  # Celsius
            'precipitation': {'min': 0, 'max': 1000},  # mm/day
            'radiation': {'min': 0, 'max': 50000},  # J/m²
            'co2': {'min': 300, 'max': 500},  # ppm
            'vpd': {'min': 0, 'max': 100}  # hPa
        }

        range_results = {}

        for variable, ranges in range_checks.items():
            variable_files = [f for f in output_files if variable.upper() in f]

            for file_path in variable_files:
                try:
                    with netCDF4.Dataset(file_path, 'r') as nc:
                        data = nc.variables['data'][:]

                        # Check ranges (ignoring NaN values)
                        valid_data = data[~np.isnan(data)]

                        if len(valid_data) > 0:
                            data_min = np.min(valid_data)
                            data_max = np.max(valid_data)

                            range_results[file_path] = {
                                'min_value': float(data_min),
                                'max_value': float(data_max),
                                'expected_min': ranges['min'],
                                'expected_max': ranges['max'],
                                'within_range': (data_min >= ranges['min'] and
                                               data_max <= ranges['max'])
                            }
                        else:
                            range_results[file_path] = {
                                'status': 'no_valid_data'
                            }

                except Exception as e:
                    range_results[file_path] = {
                        'status': 'error',
                        'message': str(e)
                    }

        return range_results
```

### Report Generator
```python
class ReportGenerator:
    """Generate comprehensive pipeline reports"""

    def __init__(self, config):
        self.config = config
        self.report_template_dir = "templates/reports/"

    def generate_pipeline_summary(self, pipeline_id, stage_results):
        """Generate overall pipeline summary report"""

        report_data = {
            'pipeline_id': pipeline_id,
            'execution_time': datetime.now().isoformat(),
            'stage_results': stage_results,
            'overall_status': self._determine_pipeline_status(stage_results),
            'statistics': self._calculate_pipeline_statistics(stage_results)
        }

        # Generate HTML report
        html_report = self._generate_html_report(report_data, 'pipeline_summary.html')

        # Generate JSON summary
        json_report = self._generate_json_report(report_data, 'pipeline_summary.json')

        return {
            'html_report': html_report,
            'json_report': json_report,
            'status': report_data['overall_status']
        }

    def generate_qa_report(self, pipeline_id, qa_results):
        """Generate quality assurance report"""

        report_data = {
            'pipeline_id': pipeline_id,
            'qa_execution_time': datetime.now().isoformat(),
            'qa_results': qa_results,
            'overall_qa_status': self._determine_qa_status(qa_results),
            'recommendations': self._generate_qa_recommendations(qa_results)
        }

        return self._generate_html_report(report_data, 'qa_report.html')

    def _generate_html_report(self, data, template_name):
        """Generate HTML report from template"""

        # This would use a templating engine like Jinja2
        # For now, we'll create a simple HTML structure

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CARDAMOM Pipeline Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .status-pass {{ color: green; }}
                .status-fail {{ color: red; }}
                .status-warning {{ color: orange; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>CARDAMOM Pipeline Report</h1>
            <h2>Pipeline ID: {data['pipeline_id']}</h2>
            <p>Execution Time: {data.get('execution_time', 'Unknown')}</p>

            <!-- Report content would be generated here -->

        </body>
        </html>
        """

        # Save report to file
        report_dir = self.config['paths']['quality_reports']
        report_filename = f"{data['pipeline_id']}_report.html"
        report_path = os.path.join(report_dir, report_filename)

        with open(report_path, 'w') as f:
            f.write(html_content)

        return report_path
```

## 6.7 Testing Framework

### Integration Tests
```
tests/pipeline/
├── test_pipeline_manager.py
├── test_component_managers.py
├── test_pipeline_state.py
├── test_error_handling.py
├── test_qa_reporting.py
└── fixtures/
    ├── sample_configs/
    ├── test_datasets/
    └── expected_outputs/
```

### End-to-End Tests
```python
def test_complete_global_monthly_pipeline():
    """Test complete global monthly processing pipeline"""

def test_complete_conus_diurnal_pipeline():
    """Test complete CONUS diurnal processing pipeline"""

def test_pipeline_resumability():
    """Test pipeline can be resumed after interruption"""

def test_error_recovery():
    """Test error handling and recovery mechanisms"""
```

## 6.8 Success Criteria

### Functional Requirements
- [ ] Successfully orchestrate all workflow components
- [ ] Support both global monthly and CONUS diurnal workflows
- [ ] Implement robust error handling and recovery
- [ ] Provide comprehensive progress tracking

### Performance Requirements
- [ ] Efficiently coordinate parallel processing
- [ ] Support resumable execution for long workflows
- [ ] Optimize resource usage across components
- [ ] Scale to multi-year processing jobs

### Quality Requirements
- [ ] Comprehensive quality assurance checks
- [ ] Detailed reporting and logging
- [ ] Consistent output format validation
- [ ] Integration with existing CARDAMOM infrastructure

### Reliability Requirements
- [ ] Graceful handling of component failures
- [ ] Automatic recovery where possible
- [ ] State persistence for resumability
- [ ] Comprehensive error logging and reporting