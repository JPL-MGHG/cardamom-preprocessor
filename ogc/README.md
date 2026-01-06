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
├── ecmwf/                             # ECMWF ERA5 downloader package
│   ├── process.cwl                    # CWL workflow definition
│   ├── Dockerfile                     # Container image specification
│   ├── run_ecmwf.sh                   # MAAP secrets + CLI wrapper script
│   └── examples/
│       ├── basic.yml                  # Simple single-month example
│       └── multi_variable.yml         # Multi-variable example
├── noaa/                              # NOAA CO₂ downloader (future)
│   └── .gitkeep
└── gfed/                              # GFED fire emissions downloader (future)
    └── .gitkeep
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

## Extending to Other Downloaders

### NOAA CO₂ Downloader (No Credentials Required)

Create `ogc/noaa/` with similar structure:

**Key differences from ECMWF:**
- No credentials needed (public data)
- Simpler wrapper script (skip MAAP secrets retrieval)
- CLI: `python -m src.stac_cli noaa --year 2020 --month 1`
- Optional parameters: `--year`, `--month` (if omitted, downloads all available data)

**Template approach:**
1. Copy `ogc/ecmwf/` to `ogc/noaa/`
2. Modify `process.cwl` to remove credential inputs
3. Modify `run_ecmwf.sh` to `run_noaa.sh` (skip secrets retrieval)
4. Update CLI command in wrapper script
5. Update documentation

### GFED Fire Emissions Downloader (SFTP Credentials)

Create `ogc/gfed/` with similar structure:

**Key differences from ECMWF:**
- SFTP credentials (username/password)
- MAAP secrets: `GFED_SFTP_USERNAME`, `GFED_SFTP_PASSWORD`
- Batch yearly processing: `--start-year 2001 --end-year 2024`
- CLI: `python -m src.stac_cli gfed --start-year 2001 --end-year 2024`
- Larger data volumes (~100GB per year range)

**Template approach:**
1. Copy `ogc/ecmwf/` to `ogc/gfed/`
2. Modify `process.cwl` for GFED-specific inputs/outputs
3. Modify wrapper script to retrieve SFTP credentials
4. Update CLI command and parameters
5. Adjust resource requirements (larger output storage)
6. Update documentation

## File Descriptions

### process.cwl

OGC entry point defining:
- Input parameters (mapped to CLI arguments)
- Output specifications and STAC structure
- Docker image and resource requirements
- Schema.org metadata for discoverability

Validated with: `cwltool --validate process.cwl`

### Dockerfile

Multi-stage Docker build containing:
- Stage 1 (Builder): Create conda environment, install maap-py
- Stage 2 (Production): Copy environment, source code, configure entrypoint

Build with: `docker build -t cardamom-ecmwf:test -f ogc/ecmwf/Dockerfile .`

### run_ecmwf.sh

MAAP wrapper script handling:
1. Credential retrieval from MAAP secrets or CWL inputs
2. CDS API configuration (~/.cdsapirc creation)
3. Output directory preparation
4. Core CLI invocation
5. Status reporting and error handling

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
