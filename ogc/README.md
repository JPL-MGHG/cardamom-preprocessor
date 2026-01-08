# OGC Application Packages for CARDAMOM Preprocessor

This directory contains OGC-compliant Application Packages for deploying CARDAMOM data downloaders and processing tools on NASA MAAP (Multi-Mission Algorithm and Analysis Platform).

## Overview

### What are OGC Application Packages?

OGC Application Packages standardize the deployment of scientific algorithms on distributed cloud platforms. They combine:

1. **CWL (Common Workflow Language)** - Workflow definitions describing inputs, outputs, and execution
2. **Docker Container** - Reproducible execution environment with all dependencies
3. **Wrapper Scripts** - Platform-specific integration (e.g., MAAP secrets, credential management)

This ensures algorithms run consistently across different OGC-compliant platforms (MAAP, EMS, ADES, etc.).

### Why OGC for CARDAMOM?

- **Portability**: Algorithm runs on any OGC-compliant platform
- **Discoverability**: Metadata enables automated algorithm discovery and chaining
- **Reproducibility**: Docker ensures consistent execution across environments
- **Decoupling**: Wrapper scripts keep core code platform-independent

## Directory Structure

```
ogc/
├── README.md                          # This file
├── Dockerfile                         # Unified Docker image for all packages
├── ecmwf/                             # ECMWF ERA5 meteorology downloader
│   ├── process.cwl                    # CWL workflow definition
│   ├── run_ecmwf.sh                   # MAAP secrets wrapper script
│   └── examples/
│       ├── basic.yml                  # Single-month example
│       └── multi_variable.yml         # Multi-variable example
├── noaa/                              # NOAA CO₂ concentration downloader
│   ├── process.cwl                    # CWL workflow definition
│   ├── run_noaa.sh                    # Wrapper script (public data, no credentials)
│   └── examples/
│       ├── basic.yml                  # Single-month example
│       └── full_timeseries.yml        # Complete timeseries example
├── gfed/                              # GFED fire emissions downloader
│   ├── process.cwl                    # CWL workflow definition
│   ├── run_gfed.sh                    # SFTP credentials wrapper script
│   └── examples/
│       ├── single_year.yml            # Single year example
│       └── multi_year.yml             # Multi-year batch example
└── cbf/                               # CBF file generator (STAC-based)
    ├── process.cwl                    # CWL workflow definition
    ├── run_cbf.sh                     # Wrapper script (no credentials needed)
    └── examples/
        ├── basic.yml                  # CONUS region example
        └── with_files.yml             # Example with optional input files
```

## Package Details: ECMWF ERA5 Downloader

### Purpose

Downloads ERA5 reanalysis meteorological variables from ECMWF Climate Data Store for CARDAMOM carbon cycle modeling. Produces NetCDF files with STAC metadata catalogs.

### Supported Variables

- **t2m_min, t2m_max**: 2-meter temperature extrema (Kelvin)
- **vpd**: Vapor Pressure Deficit (hectopascals)
- **total_prec**: Total precipitation (millimeters)
- **ssrd**: Surface solar radiation downwards (W/m²)
- **strd**: Surface thermal radiation downwards (W/m²)
- **skt**: Skin temperature (Kelvin)
- **snowfall**: Snowfall (millimeters)

### Output Structure

```
outputs/
├── catalog.json                          # Root STAC catalog
├── data/
│   ├── t2m_min_2020_01.nc               # NetCDF data files
│   ├── t2m_max_2020_01.nc
│   └── vpd_2020_01.nc
└── cardamom-meteorology/                 # Collections by variable type
    ├── collection.json
    └── items/
        ├── t2m_min_2020_01.json
        ├── t2m_max_2020_01.json
        └── vpd_2020_01.json
```

## Usage

### 1. Local Testing with Docker

#### Build the container image:

```bash
cd /path/to/cardamom-preprocessor
docker build -t cardamom-ecmwf:test -f ogc/ecmwf/Dockerfile .
```

#### Run with test credentials:

```bash
# Create test output directory
mkdir -p test_outputs

# Run with explicit credentials (for testing, NOT for production)
docker run --rm -it \
  -v $(pwd)/test_outputs:/app/outputs \
  cardamom-ecmwf:test \
  /app/ogc/ecmwf/run_ecmwf.sh \
    --ecmwf_cds_key YOUR_CDS_API_KEY \
    --variables t2m_min,t2m_max \
    --year 2020 \
    --month 1 \
    --verbose
```

**Note**: For testing, you need ECMWF CDS API credentials:
1. Register at https://cds.climate.copernicus.eu/
2. Log in and go to your profile
3. Copy your API Key

#### Verify outputs:

```bash
ls -lh test_outputs/
ls -lh test_outputs/data/
cat test_outputs/catalog.json | python -m json.tool
```

### 2. CWL Validation and Local Execution

#### Install cwltool:

```bash
pip install cwltool
```

#### Validate CWL workflow:

```bash
cwltool --validate ogc/ecmwf/process.cwl
```

#### Execute with example inputs:

```bash
cd ogc/ecmwf

# Using basic example
cwltool process.cwl examples/basic.yml

# Using multi-variable example
cwltool process.cwl examples/multi_variable.yml
```

### 3. MAAP Platform Deployment

#### Step 1: Configure MAAP Secrets (One-time Setup)

Store your ECMWF CDS credentials securely in MAAP:

```python
from maap.maap import MAAP

maap = MAAP()

# Create secrets (one-time setup)
maap.secrets.create_secret("ECMWF_CDS_KEY", "your-cds-api-key")

# Verify secret created
key = maap.secrets.get_secret("ECMWF_CDS_KEY")
print(f"Secret configured: {key is not None}")
```

#### Step 2: Build and Push Docker Image

```bash
# Build image
docker build -t ghcr.io/jpl-mghg/cardamom-preprocessor-ecmwf:latest \
  -f ogc/ecmwf/Dockerfile .

# Push to GitHub Container Registry
docker push ghcr.io/jpl-mghg/cardamom-preprocessor-ecmwf:latest
```

Alternatively, use GitHub Actions for automated builds on commits.

#### Step 3: Register Algorithm with MAAP

Upload the CWL to MAAP's algorithm registry:

```python
from maap.maap import MAAP

maap = MAAP()

maap.register_algorithm(
    name="cardamom-ecmwf-downloader",
    version="1.0.0",
    cwl_path="path/to/ogc/ecmwf/process.cwl",
    repository_name="cardamom-preprocessor",
    docker_image="ghcr.io/jpl-mghg/cardamom-preprocessor-ecmwf:latest"
)
```

#### Step 4: Submit Download Job

```python
from maap.maap import MAAP

maap = MAAP()

# Submit a single-month download
job = maap.submit_algorithm(
    algorithm_id="cardamom-ecmwf-downloader",
    version="1.0.0",
    inputs={
        "variables": "t2m_min,t2m_max,vpd",
        "year": 2020,
        "month": 1,
        "verbose": True
    }
)

print(f"Job submitted: {job.id}")
print(f"Status: {job.status}")
```

#### Step 5: Monitor and Retrieve Results

```python
from maap.maap import MAAP

maap = MAAP()

# Get job status
job = maap.get_job(job_id)
print(f"Status: {job.status}")

# Retrieve outputs once complete
if job.status == "SUCCESSFUL":
    outputs = maap.get_job_output(job_id)
    print(f"Output directory: {outputs['outputs_result']['location']}")
```

### 4. Batch Processing (Multi-Month Downloads)

For downloading multiple months, submit jobs programmatically:

```python
from maap.maap import MAAP

maap = MAAP()

# Download entire year (12 jobs, one per month)
jobs = []
for month in range(1, 13):
    job = maap.submit_algorithm(
        algorithm_id="cardamom-ecmwf-downloader",
        version="1.0.0",
        inputs={
            "variables": "t2m_min,t2m_max,vpd,total_prec",
            "year": 2020,
            "month": month
        }
    )
    jobs.append(job.id)
    print(f"Submitted job {month}: {job.id}")

# Monitor all jobs
for job_id in jobs:
    job = maap.get_job(job_id)
    print(f"{job_id}: {job.status}")
```

Alternatively, use a CWL Workflow with scatter pattern for multi-month downloads in a single job.

## CLI Arguments and Examples

### Basic Single-Month Download

```bash
python -m src.stac_cli ecmwf \
    --variables t2m_min,t2m_max \
    --year 2020 --month 1 \
    --output ./era5_output
```

### Multi-Variable Download

```bash
python -m src.stac_cli ecmwf \
    --variables t2m_min,t2m_max,vpd,total_prec,ssrd,strd,skt,snowfall \
    --year 2020 --month 1 \
    --output ./era5_output
```

### With Optional Parameters

```bash
python -m src.stac_cli ecmwf \
    --variables t2m_min,vpd \
    --year 2020 --month 6 \
    --output ./era5_output \
    --keep-raw \
    --verbose \
    --stac-duplicate-policy update
```

## Package Details: NOAA CO₂ Downloader

### Purpose

Downloads global CO₂ concentration data from NOAA Global Monitoring Laboratory for CARDAMOM carbon cycle modeling. Produces NetCDF files with STAC metadata catalogs. Data spans from 1974 to present (monthly resolution).

### Key Features

- **No credentials required** - Public NOAA data via HTTPS
- **Optional time parameters** - Can download single month or entire timeseries
- **Simple wrapper** - No secrets retrieval needed (unlike ECMWF)
- **STAC output** - Organized by measurement type with full metadata

### Usage Example

```bash
# Download single month
cwltool ogc/noaa/process.cwl ogc/noaa/examples/basic.yml

# Download entire timeseries (all available data)
cwltool ogc/noaa/process.cwl ogc/noaa/examples/full_timeseries.yml
```

### CLI Command

```bash
python -m src.stac_cli noaa \
  [--year YEAR] \
  [--month MONTH] \
  --output OUTPUT \
  [--verbose] \
  [--no-stac-incremental] \
  [--stac-duplicate-policy {update,skip,error}]
```

---

## Package Details: GFED Fire Emissions Downloader

### Purpose

Downloads Global Fire Emissions Database (GFED4) fire emissions data for CARDAMOM carbon cycle modeling. Processes yearly HDF5 files from SFTP server and produces analysis-ready NetCDF outputs. Data spans 2001-present at 0.25° resolution, regridded to 0.5° for CARDAMOM analysis.

### Key Features

- **SFTP credentials required** - Username/password from MAAP secrets
- **Batch yearly processing** - Downloads year ranges (e.g., 2001-2024)
- **Large data volumes** - ~100GB per year range
- **Land-sea masking** - Optional spatial filtering via mask file
- **STAC output** - Organized by variable type with comprehensive metadata

### MAAP Secrets Configuration

Before deploying, configure GFED SFTP credentials in MAAP:

```python
from maap.maap import MAAP

maap = MAAP()
maap.secrets.create_secret("GFED_SFTP_USERNAME", "sftp-username")
maap.secrets.create_secret("GFED_SFTP_PASSWORD", "sftp-password")
```

### Usage Example

```bash
# Download single year
cwltool ogc/gfed/process.cwl ogc/gfed/examples/single_year.yml

# Download multi-year batch
cwltool ogc/gfed/process.cwl ogc/gfed/examples/multi_year.yml
```

### CLI Command

```bash
python -m src.stac_cli gfed \
  --start-year START_YEAR \
  --end-year END_YEAR \
  --output OUTPUT \
  [--keep-raw] \
  [--verbose] \
  [--no-stac-incremental] \
  [--stac-duplicate-policy {update,skip,error}] \
  [--land-sea-mask-file FILE]
```

---

## Package Details: CBF File Generator

### Purpose

Generates CARDAMOM CBF (CARDAMOM Binary Format) input files for carbon cycle data assimilation. Consumes STAC catalogs from preprocessor downloaders (ECMWF, NOAA, GFED) and optional observational constraint files. Produces pixel-specific CBF NetCDF files ready for CARDAMOM model runs.

### Key Features

- **STAC-based input discovery** - Reads meteorological data from STAC catalogs
- **Multi-source support** - Combines meteorology + optional observations
- **Graceful degradation** - Missing observational data NaN-filled for forward-mode processing
- **No credentials required** - Reads from files and catalogs only
- **Pixel-level processing** - Generates one CBF file per valid land pixel
- **MCMC ready** - Includes configuration for data assimilation

### Workflow

1. Discover meteorological variables from STAC catalog (required)
2. Load optional observational constraint files
3. Extract pixel-level data for spatial domain
4. Generate CARDAMOM-ready CBF NetCDF files

### Usage Example

```bash
# Basic CONUS example (meteorology only)
cwltool ogc/cbf/process.cwl ogc/cbf/examples/basic.yml

# With observational constraints
cwltool ogc/cbf/process.cwl ogc/cbf/examples/with_files.yml
```

### CLI Command

```bash
python -m src.stac_cli cbf-generate \
  --stac-api STAC_API \
  --start START_DATE \
  --end END_DATE \
  --output OUTPUT \
  [--region {global,conus}] \
  [--land-fraction-file FILE] \
  [--obs-driver-file FILE] \
  [--som-file FILE] \
  [--fir-file FILE] \
  [--scaffold-file FILE] \
  [--verbose]
```

---

## Unified Docker Image

All packages (ECMWF, NOAA, GFED, CBF) share a single Docker image: `ghcr.io/jpl-mghg/cardamom-preprocessor:latest`

### Build Instructions

```bash
# Build unified image
docker build -t ghcr.io/jpl-mghg/cardamom-preprocessor:latest -f ogc/Dockerfile .

# Test ECMWF package
docker run --rm -it \
  -v $(pwd)/test_outputs:/app/outputs \
  ghcr.io/jpl-mghg/cardamom-preprocessor:latest \
  /app/ogc/ecmwf/run_ecmwf.sh \
    --ecmwf_cds_key YOUR_KEY \
    --variables t2m_min,t2m_max \
    --year 2020 --month 1

# Test NOAA package (no credentials)
docker run --rm -it \
  -v $(pwd)/test_outputs:/app/outputs \
  ghcr.io/jpl-mghg/cardamom-preprocessor:latest \
  /app/ogc/noaa/run_noaa.sh \
    --year 2020 --month 1
```

### Resource Requirements by Package

| Package | RAM | Cores | Temp | Output | Network | Time |
|---------|-----|-------|------|--------|---------|------|
| ECMWF | 8GB | 2 | 10GB | 50GB | Yes | 15-60min |
| NOAA | 4GB | 1 | 5GB | 10GB | Yes | 5-15min |
| GFED | 16GB | 2 | 20GB | 100GB | Yes (SFTP) | 1-4hrs |
| CBF | 16GB | 4 | 10GB | 20GB | No | 30min-2hr |

## File Descriptions

### process.cwl

OGC entry point defining:
- Input parameters (mapped to CLI arguments)
- Output specifications and STAC structure
- Docker image and resource requirements
- Schema.org metadata for discoverability

Validated with: `cwltool --validate process.cwl`

### Dockerfile (Unified)

Multi-stage Docker build (`ogc/Dockerfile`) containing all packages:
- Stage 1 (Builder): Create conda environment, install maap-py, dependencies
- Stage 2 (Production): Copy environment, all source code, all wrapper scripts

Key feature: Single image serves ECMWF, NOAA, GFED, and CBF packages via different entrypoints

Build with: `docker build -t ghcr.io/jpl-mghg/cardamom-preprocessor:latest -f ogc/Dockerfile .`

Historical note: Package-specific Dockerfiles (e.g., `ogc/ecmwf/Dockerfile`) are deprecated. Use unified image instead.

### Wrapper Scripts

Each package has a platform-specific wrapper script handling integration with MAAP:

#### run_ecmwf.sh (ECMWF ERA5)
1. Parse CLI arguments
2. Retrieve ECMWF CDS API credentials from MAAP secrets (or CLI inputs)
3. Configure CDS API authentication (~/.cdsapirc file)
4. Prepare output directory
5. Invoke core CLI: `python -m src.stac_cli ecmwf ...`
6. Report success/failure with output statistics

#### run_noaa.sh (NOAA CO₂)
1. Validate arguments
2. Prepare output directory (no credentials needed)
3. Invoke core CLI: `python -m src.stac_cli noaa ...`
4. Report success/failure with output statistics

**Key difference:** Simpler than ECMWF (public data, no API key needed)

#### run_gfed.sh (GFED Fire)
1. Parse CLI arguments
2. Retrieve GFED SFTP credentials from MAAP secrets (or CLI inputs)
3. Export credentials as environment variables
4. Prepare output directory
5. Invoke core CLI: `python -m src.stac_cli gfed ...`
6. Report success/failure with output statistics

**Key difference:** Retrieves SFTP username/password for GFED data access

#### run_cbf.sh (CBF Generator)
1. Validate STAC API source (file:// or https://)
2. Prepare output directory
3. Invoke core CLI: `python -m src.stac_cli cbf-generate ...`
4. Report success/failure with CBF file counts

**Key difference:** No credentials (reads from files/catalogs only)

### examples/basic.yml

Simple YAML input file for testing:
```yaml
variables: "t2m_min,t2m_max"
year: 2020
month: 1
verbose: true
```

### examples/multi_variable.yml

Comprehensive example with all available variables:
```yaml
variables: "t2m_min,t2m_max,vpd,total_prec,ssrd,strd,skt,snowfall"
year: 2020
month: 1
keep_raw: false
stac_duplicate_policy: "update"
```

## Troubleshooting

### CDS API Authentication Failures

**Error**: `Invalid credentials` or `401 Unauthorized`

**Solutions:**
1. Verify CDS API key at https://cds.climate.copernicus.eu/user
2. Check MAAP secrets are configured correctly:
   ```python
   maap = MAAP()
   key = maap.secrets.get_secret("ECMWF_CDS_KEY")
   print(f"API Key exists: {key is not None}")
   ```
3. For local testing, pass credentials explicitly to wrapper script

### Download Timeouts

**Error**: `Request timed out` or `Connection refused`

**Solutions:**
1. CDS API can be slow; increase timeout in cdsapi configuration
2. Check CDS API status: https://cds.climate.copernicus.eu/
3. Reduce number of variables per request (though batch optimization should help)
4. Retry job after CDS API recovers

### Docker Build Failures

**Error**: `Dependency installation failed`

**Solutions:**
1. Ensure `environment.yml` exists and is in build context
2. Check internet connectivity for conda package downloads
3. Use `--no-cache` flag to force fresh build:
   ```bash
   docker build --no-cache -t cardamom-ecmwf:test -f ogc/ecmwf/Dockerfile .
   ```

### Output Directory Permission Errors

**Error**: `Permission denied: /app/outputs`

**Solutions:**
1. Check MAAP job has write permissions to output directory
2. Verify output directory exists and is writable
3. Check file permissions in wrapper script

## References

- **OGC Application Packages**: https://docs.ogc.org/bp/20-089r1.html
- **Common Workflow Language**: https://www.commonwl.org/
- **NASA MAAP Documentation**: https://docs.maap-project.org/
- **STAC Specification**: https://stacspec.org/
- **ECMWF CDS API**: https://cds.climate.copernicus.eu/cdsapp#!/home

## Contact and Support

For issues or questions:
1. Check existing GitHub issues: https://github.com/JPL-MGHG/cardamom-preprocessor/issues
2. Contact CARDAMOM team: support@maap-project.org
3. MAAP platform issues: https://github.com/MAAP-Project/issues

## Future Enhancements

- [ ] Multi-month CWL Workflow with scatter pattern
- [ ] Output validation step (verify STAC catalog correctness)
- [ ] GitHub Actions for automated Docker image builds
- [ ] STAC catalog auto-publishing to MAAP STAC endpoint
- [ ] Job progress reporting to MAAP API
- [ ] Cost estimation for CDS API requests

## License

Apache License 2.0 - See LICENSE file in repository root.
