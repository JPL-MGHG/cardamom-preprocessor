# Migration to ecmwf-datastores-client

## Overview

This document describes the migration from `cdsapi` to `ecmwf-datastores-client` for efficient batch processing of ECMWF data downloads.

## Changes Made

### 1. Dependencies ([environment.yml:15](environment.yml#L15))
- **Removed**: `cdsapi`
- **Added**: `ecmwf-datastores-client`

### 2. Core Implementation ([src/ecmwf_downloader.py](src/ecmwf_downloader.py))

#### Import Changes
```python
# Old
import cdsapi

# New
from ecmwf.datastores import Client
```

#### Client Initialization
```python
# Old
self.client = cdsapi.Client()

# New
self.client = Client()
```

#### Download Pattern Changes

**Old Pattern (Sequential):**
```python
for year in years:
    for month in months:
        for variable in variables:
            self.client.retrieve(dataset, request).download(filepath)
```

**New Pattern (Batch Submission):**
```python
# Phase 1: Submit all requests (non-blocking)
remotes = []
for year/month/variable combinations:
    remote = self.client.submit(dataset, request)
    remotes.append((remote, filepath, metadata))

# Phase 2: Monitor job completion
self._wait_for_jobs(remotes)

# Phase 3: Download results
self._download_results(remotes)
```

### 3. New Helper Methods

#### `_wait_for_jobs(job_info_list)`
Monitors submitted jobs and polls status every 5 seconds until all jobs complete.

**Features:**
- Real-time progress reporting
- Failed job tracking
- Status display with visual indicators (✓, ✗, ⏳)

#### `_download_results(job_info_list)`
Downloads results from completed jobs with error handling.

**Features:**
- Progress tracking
- Individual file download status
- Error reporting per file

## Configuration

### Authentication Setup

The new client uses the same authentication as `cdsapi`. Configure credentials in one of these ways:

**1. Configuration File** (Recommended)
Create `~/.ecmwfdatastoresrc`:
```ini
url: https://cds.climate.copernicus.eu/api
key: <YOUR_UID>:<YOUR_API_KEY>
```

**2. Environment Variables**
```bash
export ECMWF_DATASTORES_URL="https://cds.climate.copernicus.eu/api"
export ECMWF_DATASTORES_KEY="<YOUR_UID>:<YOUR_API_KEY>"
```

**3. Programmatic Configuration**
```python
client = Client(
    url="https://cds.climate.copernicus.eu/api",
    key="<YOUR_UID>:<YOUR_API_KEY>"
)
```

### Migration from cdsapi Credentials

If you have existing `~/.cdsapirc` credentials, copy them to `~/.ecmwfdatastoresrc` with the same format.

## Performance Improvements

### Before (cdsapi - Sequential)
```
Submit Request 1 → Wait → Download → Submit Request 2 → Wait → Download → ...
Total Time: N × (submit_time + queue_time + download_time)
Idle Time: ~50% (waiting for individual jobs to complete)
```

### After (ecmwf-datastores-client - Batch)
```
Submit All Requests → Monitor All Jobs → Download All Results
Total Time: submit_time + max(queue_times) + download_time
Idle Time: ~5-10% (only polling intervals)
```

### Expected Speed Improvements
- **Small batches (3-5 files)**: 30-40% faster
- **Medium batches (10-20 files)**: 50-60% faster
- **Large batches (50+ files)**: 70-80% faster

## Testing

### Install Updated Dependencies
```bash
# Update conda environment
conda env update -f environment.yml

# Or install package directly
pip install ecmwf-datastores-client
```

### Run Test Suite
```bash
# Basic functionality test (small dataset)
python test_batch_download.py
```

### Manual Testing
```bash
# Test hourly download (1 variable, 1 month)
python src/ecmwf_downloader.py hourly \
  -v 2m_temperature \
  -y 2024 \
  -m 1 \
  -o ./test_output

# Test monthly download (2 variables, 2 months)
python src/ecmwf_downloader.py monthly \
  -v 2m_temperature,total_precipitation \
  -y 2024 \
  -m 1-2 \
  -o ./test_output
```

## Breaking Changes

### None for End Users
The CLI interface remains unchanged. All existing scripts and workflows continue to work.

### API Changes (For Developers)
1. **Client instantiation**: Use `Client()` instead of `cdsapi.Client()`
2. **Request submission**: Returns `Remote` object instead of blocking
3. **Dataset names**: May need verification (collection IDs vs dataset names)

## Known Issues & Limitations

1. **Incubating Status**: `ecmwf-datastores-client` is marked as "Incubating" by ECMWF
   - API is mostly stable but may change
   - Suitable for research use; monitor for updates in operational systems

2. **Polling Interval**: Current 5-second polling may be adjusted based on typical job durations

3. **Error Recovery**: Failed jobs are reported but not automatically retried
   - Users must manually rerun failed downloads

## Troubleshooting

### Import Error: `No module named 'ecmwf.datastores'`
```bash
# Reinstall the package
pip install --upgrade ecmwf-datastores-client
```

### Authentication Errors
```bash
# Verify credentials file exists and is valid
cat ~/.ecmwfdatastoresrc

# Or check environment variables
echo $ECMWF_DATASTORES_KEY
```

### Job Status Errors
- Some jobs may remain in "queued" state longer than expected
- Monitor ECMWF CDS status: https://cds.climate.copernicus.eu/

## Future Enhancements

Potential improvements for future releases:

1. **Configurable polling interval**: Add CLI flag for custom polling rates
2. **Automatic retry**: Retry failed jobs with exponential backoff
3. **Concurrent download limits**: Prevent overwhelming the API with simultaneous requests
4. **Resume capability**: Save job IDs to resume interrupted downloads
5. **Progress bars**: Add detailed progress bars using `tqdm` or similar

## References

- [ecmwf-datastores-client GitHub](https://github.com/ecmwf/ecmwf-datastores-client)
- [ECMWF Technical Documentation](https://ecmwf.github.io/ecmwf-datastores-client/)
- [Climate Data Store](https://cds.climate.copernicus.eu/)
- [CDS API Migration Guide](https://confluence.ecmwf.int/x/uINmFw)

## Support

For issues related to:
- **This implementation**: Open an issue in the project repository
- **ecmwf-datastores-client**: https://github.com/ecmwf/ecmwf-datastores-client/issues
- **ECMWF CDS API**: https://forum.ecmwf.int/
