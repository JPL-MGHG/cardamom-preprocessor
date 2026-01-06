# CBF Generator Flow - Mermaid Diagram

## Complete Workflow: STAC to CBF Files

```mermaid
graph TD
    CLI["CLI Command<br/>python -m src.stac_cli cbf-generate<br/>--stac-api-url ./catalog.json<br/>--start 2020-01 --end 2020-12<br/>--output ./cbf_output"]

    INIT["Initialize CBF Generator"]
    PARSE["Parse Date Range<br/>2020-01 to 2020-12<br/>→ 12 months"]

    subgraph MET["METEOROLOGY PATH (REQUIRED)"]
        METDISCOVER["Discover Meteorology from STAC<br/>stac_met_loader.discover_stac_items()"]
        METLOAD["Load Meteorology from STAC Items<br/>Handle 3 temporal structures<br/>Single month, yearly, full TS"]
        METVALIDATE["Validate Completeness<br/>CRITICAL: FAIL if missing<br/>Variable or month missing"]
        METASSEMBLY["Assemble Unified Dataset<br/>Regrid, normalize coords<br/>Apply land-sea mask"]
    end

    subgraph OBS["OBSERVATION PATH (OPTIONAL)"]
        OBSLOAD["Load Observational Data<br/>cbf_obs_handler.load_obs()<br/>Try each file, NaN-fill if missing"]
        OBSALIGN["Align to Meteorology Time<br/>Get time from meteorology<br/>NaN-fill missing steps<br/>Allow temporal mismatches"]
    end

    subgraph PIXELS["PIXEL PROCESSING (For Each Land Pixel)"]
        LANDPIXELS["Find Land Pixels<br/>land_frac > 0.5"]
        PIXELLOOP["For Each Pixel: lat, lon"]
        FORCING["Extract Forcing Variables<br/>Meteorology time series<br/>All 10 variables"]
        OBSCONST["Extract Obs Constraints<br/>Try: obs_data[var].isel()<br/>Except: Use NaN"]
        SINGLE["Set Single-Value Constraints<br/>PEQ_iniSOM, PEQ_CUE<br/>Mean_LAI, Mean_FIR"]
        MCMC["Set MCMC Attributes<br/>nITERATIONS: 500,000<br/>nSAMPLES: 20"]
        FINALIZE["Finalize and Save CBF File<br/>Rename ABGB → ABGB_val<br/>Set encoding, write NetCDF"]
    end

    OUTPUT["Output CBF Files<br/>Directory: ./cbf_output/<br/>Files: site*.cbf.nc<br/>Total: ~12,800 for CONUS"]

    RESULTS["Return Results<br/>cbf_files: List of paths<br/>success: True<br/>metadata: date_range, region, num_files"]

    CLI --> INIT
    INIT --> PARSE

    PARSE --> METDISCOVER
    PARSE --> OBSLOAD

    METDISCOVER --> METLOAD
    METLOAD --> METVALIDATE
    METVALIDATE -->|Pass| METASSEMBLY
    METVALIDATE -->|FAIL| FAIL1["❌ FAIL: Cannot proceed<br/>(meteorology incomplete)"]

    OBSLOAD --> OBSALIGN

    METASSEMBLY --> LANDPIXELS
    OBSALIGN --> OBSALIGN

    LANDPIXELS --> PIXELLOOP
    PIXELLOOP --> FORCING
    PIXELLOOP --> OBSCONST
    FORCING --> SINGLE
    OBSCONST --> SINGLE
    SINGLE --> MCMC
    MCMC --> FINALIZE
    FINALIZE --> PIXELLOOP
    PIXELLOOP -->|All pixels done| OUTPUT

    OUTPUT --> RESULTS

    style CLI fill:#d5e8d4
    style MET fill:#add1fe
    style OBS fill:#add1fe
    style METVALIDATE fill:#ffe6cc
    style METASSEMBLY fill:#f8cecc
    style OBSALIGN fill:#f8cecc
    style PIXELS fill:#d5e8d4
    style OUTPUT fill:#d5e8d4
    style FAIL1 fill:#ff6666
