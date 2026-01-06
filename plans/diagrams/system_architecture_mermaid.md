# CARDAMOM System Architecture - Mermaid Diagram

## 5-Layer Architecture Overview

```mermaid
graph TD
    subgraph Layer1["Layer 1: Data Acquisition"]
        ECMWF["ECMWFDownloader<br/>(ERA5 Meteorology)<br/>T2M_MIN, T2M_MAX, VPD<br/>TOTAL_PREC, SSRD, STRD<br/>SKT, SNOWFALL"]
        NOAA["NOAADownloader<br/>(CO₂ Concentrations)<br/>CO2"]
        GFED["GFEDDownloader<br/>(Fire Emissions)<br/>BURNED_AREA, FIRE_C"]
    end

    subgraph Layer2["Layer 2: STAC Discovery"]
        STAC["STAC Catalog<br/>(Root)<br/>cardamom-meteorological-variables/<br/>cardamom-fire-emissions-variables/"]
        DISCOVERY["STAC Discovery Engine<br/>discover_stac_items()<br/>Metadata filtering<br/>Variable & temporal queries"]
    end

    subgraph Layer3["Layer 3: Data Loading"]
        METLOADER["Meteorology Loader<br/>stac_met_loader<br/>Handle 3 temporal structures<br/>(monthly, yearly, full TS)"]
        METVALIDATE["Validate Meteorology<br/>CRITICAL: FAIL if incomplete<br/>(scientific validity)"]
        OBSLOADER["Observational Loader<br/>cbf_obs_handler<br/>NaN-fill for missing<br/>(graceful degradation)"]
    end

    subgraph Layer4["Layer 4: Data Assembly"]
        METASSEMBLY["Meteorology Assembly<br/>Regrid to coarsest res<br/>Normalize coordinates<br/>Apply land-sea mask"]
        OBSALIGN["Observation Alignment<br/>Align to meteorology time<br/>NaN-fill missing steps<br/>Share time authority"]
    end

    subgraph Layer5["Layer 5: CBF Generation"]
        LANDPIXELS["Find Land Pixels<br/>land_frac > 0.5"]
        PIXELLOOP["For Each Land Pixel"]
        FORCING["Extract Forcing Variables<br/>(Meteorology)"]
        OBSCONST["Extract Observation Constraints<br/>(with NaN fallback)"]
        SINGLE["Set Single-Value Constraints<br/>SOM, CUE, Mean LAI, Mean FIR"]
        FINALIZE["Finalize & Save CBF File"]
        OUTPUT["Output CBF Files<br/>~12,800 files for CONUS"]
    end

    ECMWF -->|NetCDF + STAC items| STAC
    NOAA -->|NetCDF + STAC items| STAC
    GFED -->|NetCDF + STAC items| STAC

    STAC --> DISCOVERY
    DISCOVERY --> METLOADER
    DISCOVERY --> OBSLOADER

    METLOADER --> METVALIDATE
    METVALIDATE -->|FAIL if incomplete| METASSEMBLY
    OBSLOADER --> OBSALIGN

    METASSEMBLY --> LANDPIXELS
    OBSALIGN --> OBSALIGN
    LANDPIXELS --> PIXELLOOP

    PIXELLOOP --> FORCING
    PIXELLOOP --> OBSCONST
    OBSCONST --> SINGLE
    FORCING --> FINALIZE
    SINGLE --> FINALIZE
    FINALIZE --> OUTPUT

    style Layer1 fill:#d5e8d4
    style Layer2 fill:#add1fe
    style Layer3 fill:#e1d5e7
    style Layer4 fill:#f8cecc
    style Layer5 fill:#d5e8d4
    style METVALIDATE fill:#ffe6cc
    style OBSALIGN fill:#f8cecc
```

## Key Principles

- **Layer 1**: Data acquisition from third-party sources
- **Layer 2**: STAC discovery with pure metadata filtering
- **Layer 3**: Data loading with validation and graceful degradation
- **Layer 4**: Assembly and alignment to common time/space grid
- **Layer 5**: Pixel-specific CBF file generation

## Time Coordinate Authority

**Meteorology** (from STAC) → **Authority** ← **Observations** (user-provided)

- Meteorology MUST be complete (FAIL if missing)
- Observations optional (NaN-fill if missing)
