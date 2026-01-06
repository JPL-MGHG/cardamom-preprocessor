# STAC Catalog Structure - Mermaid Diagram

## Type-Based Organization

```mermaid
graph TD
    subgraph CATALOG["output/<br/>STAC Root Catalog"]
        ROOT["catalog.json<br/>(Root Catalog)"]
    end

    subgraph MET_COL["cardamom-meteorological-variables/<br/>Collection: All ERA5 Meteorology"]
        MET_COLLECT["collection.json"]
        MET_ITEMS["items/<br/>├── t2m_min_2020_01.json<br/>├── t2m_max_2020_01.json<br/>├── vpd_2020_01.json<br/>├── total_prec_2020_01.json<br/>├── ssrd_2020_01.json<br/>├── strd_2020_01.json<br/>├── skt_2020_01.json<br/>├── snowfall_2020_01.json<br/>└── ... (more months)"]
    end

    subgraph FIRE_COL["cardamom-fire-emissions-variables/<br/>Collection: GFED + NOAA CO2"]
        FIRE_COLLECT["collection.json"]
        FIRE_ITEMS["items/<br/>├── burned_area_2020.json<br/>├── fire_c_2020.json<br/>├── co2_1980_2025.json<br/>└── ... (more years)"]
    end

    subgraph OBS_COL["cardamom-observational-variables/<br/>Collection: User Obs Data"]
        OBS_COLLECT["collection.json"]
        OBS_ITEMS["items/<br/>├── lai_1km.json<br/>├── gpp_flux.json<br/>├── biomass.json<br/>└── ... (more obs)"]
    end

    subgraph DATA["data/<br/>NetCDF Files"]
        NETCDF["├── t2m_min_2020_01.nc<br/>├── t2m_max_2020_01.nc<br/>├── vpd_2020_01.nc<br/>├── burned_area_2020.nc<br/>├── co2_1980_2025.nc<br/>└── ... (all NetCDF files)"]
    end

    subgraph ITEMPROPS["STAC Item Properties<br/>(Metadata in JSON)"]
        STD["Standard STAC:<br/>• id: unique identifier<br/>• datetime/start_datetime<br/>• geometry: spatial extent<br/>• assets: link to NetCDF"]

        CUSTOM["Custom CARDAMOM:<br/>• cardamom:variable<br/>  (e.g., 't2m_min')<br/>• cardamom:variable_type<br/>  (e.g., 'meteorological')<br/>• cardamom:time_steps<br/>  (# of time steps)<br/>• cardamom:units<br/>  (variable units)"]
    end

    subgraph DISCOVERY["STAC Discovery Process"]
        D1["1. Query Catalog Recursively<br/>Start at root catalog.json<br/>Traverse all child catalogs<br/>Collect STAC items"]

        D2["2. Filter by Metadata<br/>Match cardamom:variable<br/>Filter by temporal range<br/>Select matching items"]

        D3["3. Load Data<br/>Extract asset href<br/>Load NetCDF from URL<br/>Concatenate along time"]
    end

    subgraph UPDATE["Collection Update Strategy"]
        INCR["Incremental Merging:<br/>• New downloader adds items<br/>• Collection extent updated<br/>• Merge policy: update/skip/error<br/>• No rebuild needed<br/>• Historical data preserved"]
    end

    ROOT --> MET_COLLECT
    ROOT --> FIRE_COLLECT
    ROOT --> OBS_COLLECT
    ROOT --> DATA

    MET_COLLECT --> MET_ITEMS
    FIRE_COLLECT --> FIRE_ITEMS
    OBS_COLLECT --> OBS_ITEMS

    MET_ITEMS -.->|Reference| DATA
    FIRE_ITEMS -.->|Reference| DATA
    OBS_ITEMS -.->|Reference| DATA

    MET_COLLECT -.->|Contains| STD
    MET_COLLECT -.->|Contains| CUSTOM

    D1 --> D2
    D2 --> D3

    INCR -.->|Applies to| MET_COLLECT
    INCR -.->|Applies to| FIRE_COLLECT
    INCR -.->|Applies to| OBS_COLLECT

    style CATALOG fill:#add1fe
    style MET_COL fill:#fff2cc
    style FIRE_COL fill:#fff2cc
    style OBS_COL fill:#fff2cc
    style DATA fill:#e1d5e7
    style ITEMPROPS fill:#f8cecc
    style DISCOVERY fill:#d5e8d4
    style UPDATE fill:#d5e8d4
