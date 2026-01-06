# Time Coordinate Alignment - Mermaid Diagram

## Architecture Shift: OLD vs NEW

```mermaid
graph TD
    subgraph OLD["OLD ARCHITECTURE: Scaffold as Time Authority"]
        SCAFFOLD["Scaffold Template<br/>(Static File)<br/>Time: 12 months of 2001<br/>[2001-01-01, ..., 2001-12-01]"]

        MET_OLD["Meteorology<br/>Loaded from static file<br/>Must have 12 time steps<br/>Must match scaffold months"]

        OBS_OLD["Observations<br/>Loaded from static file<br/>Must match scaffold time exactly"]

        ISSUES["Issues:<br/>✗ Tightly coupled to scaffold<br/>✗ Can't process different years<br/>✗ Inflexible for scenarios<br/>✗ Hard to update<br/>✗ Mismatch causes errors"]

        SCAFFOLD -.->|Time authority| MET_OLD
        SCAFFOLD -.->|Time authority| OBS_OLD
        MET_OLD --> ISSUES
        OBS_OLD --> ISSUES
    end

    subgraph NEW["NEW ARCHITECTURE: Meteorology as Time Authority"]
        METLOAD["Step 1: Load Meteorology from STAC<br/>stac_met_loader.load_variable_from_stac_items()<br/><br/>Source: STAC catalog (variable TS)<br/>Output: xr.Dataset with time coordinate<br/>Time steps: 12 months of 2020<br/>[2020-01-01, ..., 2020-12-01]<br/>Dimensions: [time=12, lat=360, lon=720]"]

        METAUTH["Meteorology Time Authority<br/>Source: STAC item metadata<br/>Extracted from: NetCDF CF encoding<br/><br/>This DEFINES time authority:<br/>✓ All other datasets align to this<br/>✓ CBF files use this time<br/>✓ Immutable for source"]

        OBSLOAD["Step 2: Load Observations (Optional)<br/>cbf_obs_handler.load_obs()<br/><br/>Source: User-provided files (optional)<br/>Output: xr.Dataset with (different?) time<br/><br/>Scenarios:<br/>• Obs same period: Align directly<br/>• Obs partial period: NaN-fill missing<br/>• Obs missing entirely: All NaN"]

        OBSALIGN["Step 3: Align Obs to Met Time<br/>align_observations_to_meteorology_time()<br/><br/>Algorithm:<br/>1. Get time from meteorology<br/>2. For each met time step:<br/>   • Find matching obs (if exists)<br/>   • Copy obs value<br/>   • If no match: NaN<br/>3. Result: Obs aligned to met time<br/><br/>Graceful Degradation:<br/>• Missing months → NaN (allowed)<br/>• Missing vars → NaN (allowed)<br/>• Missing pixels → NaN (allowed)"]

        SCAFFOLD_NEW["Scaffold Role Changed<br/>OLD: Provides time coordinate<br/>NEW: Template only<br/><br/>Provides Now:<br/>✓ Variable metadata (attributes)<br/>✓ Encoding specs<br/>✓ MCMC defaults<br/><br/>NO LONGER Provides:<br/>✗ Time coordinate<br/>✗ Forcing data<br/>✗ Observation data"]

        CBF["Step 5: Generate CBF Files<br/>For each pixel (lat, lon):<br/>1. Extract meteorology<br/>   → Time series with met time<br/>2. Extract observations<br/>   → Time series with met time (NaN missing)<br/>3. Create CBF:<br/>   Time: From meteorology<br/>   Forcing: From meteorology<br/>   Constraints: From obs (with NaN)<br/>   Metadata: From scaffold"]

        METLOAD --> METAUTH
        METAUTH -->|Authority| OBSALIGN
        OBSLOAD --> OBSALIGN
        METAUTH --> SCAFFOLD_NEW
        OBSALIGN --> CBF
        SCAFFOLD_NEW --> CBF
    end

    subgraph SUMMARY["Key Changes"]
        CHANGE1["OLD: Scaffold Time → Met Time → Obs Time<br/>NEW: Met Time (AUTHORITY) ← Obs align"]

        BENEFITS["Benefits of NEW Approach<br/>✓ Flexible: Different time periods<br/>✓ Robust: Obs can have gaps<br/>✓ Scientific: Met completeness validated<br/>✓ Modular: Each component independent"]
    end

    OLD -.->|Replaced by| NEW
    NEW --> SUMMARY

    style OLD fill:#ffe6cc
    style NEW fill:#d5e8d4
    style METAUTH fill:#add1fe
    style OBSALIGN fill:#f8cecc
    style SCAFFOLD_NEW fill:#fff2cc
    style SUMMARY fill:#fff2cc
    style ISSUES fill:#ff6666
