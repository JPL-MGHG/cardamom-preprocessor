# STAC Loader Update: Local and Remote Catalog Support

## Summary of Changes

Updated the STAC meteorology loader to accept both local STAC catalog.json files and remote STAC API URLs, and refactored the CBF generation architecture.

## Modified Files

### 1. `src/stac_met_loader.py`
**Changes:**
- Added `pystac` and `Path` imports for local catalog support
- Renamed parameter `stac_api_url` → `stac_source` in:
  - `discover_stac_items()`
  - `load_met_data_from_stac()`
- Added helper functions:
  - `_is_local_catalog()`: Determines if source is local file or remote URL
  - `_item_in_date_range()`: Filters STAC items by date range for local catalogs
- Updated `discover_stac_items()` to handle both:
  - **Local catalogs**: Uses `pystac.Catalog.from_file()` and manual filtering
  - **Remote APIs**: Uses `pystac_client.Client.open()` with search API

**Usage Examples:**
```python
# Local STAC catalog
met_data = load_met_data_from_stac(
    stac_source='file:///path/to/catalog.json',  # or just '/path/to/catalog.json'
    start_date='2020-01',
    end_date='2020-12'
)

# Remote STAC API
met_data = load_met_data_from_stac(
    stac_source='https://stac-api.example.com',
    start_date='2020-01',
    end_date='2020-12'
)
```

### 2. `src/cbf_main.py`
**Changes:**
- Refactored `main()` into parameterized `generate_cbf_files()` function
- Added parameters for flexible configuration:
  - `stac_source`: Local or remote STAC catalog
  - `start_date`, `end_date`: Date range for meteorology
  - `output_directory`: Where to save CBF files
  - `land_frac_file`, `obs_driver_file`, etc.: Optional input file paths
  - `experiment_id`, `lat_range`, `lon_range`, `land_threshold`: Processing parameters
- Returns result dictionary with:
  - `successful_pixels`: Number of successfully generated CBF files
  - `failed_pixels`: Number of failed pixels
  - `output_directory`: Path to output directory
- Kept backward-compatible `main()` for script execution

**Usage:**
```python
from cbf_main import generate_cbf_files

results = generate_cbf_files(
    stac_source='/path/to/catalog.json',
    start_date='2020-01',
    end_date='2020-12',
    output_directory='./output',
    experiment_id='001'
)

print(f"Generated {results['successful_pixels']} CBF files")
```

### 3. `src/stac_cli.py`
**Changes:**
- Updated import: `from cbf_generator import CBFGenerator` → `from cbf_main import generate_cbf_files`
- Refactored `handle_cbf_generate()` to use `generate_cbf_files()` directly
- Removed dependency on obsolete `CBFGenerator` class
- CLI now works with both local and remote STAC sources

**CLI Usage:**
```bash
# Using local STAC catalog
python -m src.stac_cli cbf-generate \
    --stac-api /path/to/catalog.json \
    --start 2020-01 --end 2020-12 \
    --output ./cbf_output

# Using remote STAC API
python -m src.stac_cli cbf-generate \
    --stac-api https://stac.maap-project.org \
    --start 2020-01 --end 2020-12 \
    --output ./cbf_output
```

### 4. `src/cbf_generator.py`
**Status:** REMOVED (obsolete)

This file has been removed as its functionality was:
- Redundant with `cbf_main.py`
- Less flexible (hardcoded discovery logic)
- Not aligned with the new STAC-based architecture

## Architecture Improvements

### Before
- Separate `cbf_generator.py` with `CBFGenerator` class (unused)
- Hardcoded configuration in `cbf_main.py`
- Only supported remote STAC APIs

### After
- Single `cbf_main.py` with parameterized `generate_cbf_files()` function
- Flexible configuration via function parameters
- Supports both local and remote STAC catalogs
- CLI and programmatic usage unified

## Migration Guide

If you were using `CBFGenerator` (unlikely as it wasn't used anywhere):

**Old:**
```python
from cbf_generator import CBFGenerator

generator = CBFGenerator(
    stac_api_url='https://stac.example.com',
    output_directory='./output',
    verbose=True
)

results = generator.generate(
    start_date='2020-01',
    end_date='2020-12',
    region='conus'
)
```

**New:**
```python
from cbf_main import generate_cbf_files

results = generate_cbf_files(
    stac_source='https://stac.example.com',  # or local path
    start_date='2020-01',
    end_date='2020-12',
    output_directory='./output',
)
```

## Testing

Verified all imports work:
```bash
.venv/bin/python -c "from src.stac_met_loader import discover_stac_items, load_met_data_from_stac; from src.cbf_main import generate_cbf_files; print('✓ All imports successful')"
```

Result: ✓ All imports successful

## Benefits

1. **Flexibility**: Users can now test with local STAC catalogs without setting up a remote API
2. **Offline Workflows**: Support for air-gapped environments
3. **Simplified Architecture**: Removed redundant code and unified CBF generation
4. **Better Testing**: Easier to test with local fixture catalogs
5. **Backward Compatibility**: Existing `cbf_main.py` script usage unchanged
