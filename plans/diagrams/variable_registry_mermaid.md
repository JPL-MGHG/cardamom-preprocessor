# Variable Registry System - Mermaid Diagram

## Central Metadata Hub

```mermaid
graph TD
    subgraph REGISTRY["CARDAMOM_VARIABLE_REGISTRY<br/>(Single Source of Truth)<br/>cardamom_variables.py"]
        VAR1["Variable Entry Example<br/>'2m_temperature': {<br/>  'source': 'era5',<br/>  'alternative_names': ['t2m'],<br/>  'cbf_names': ['TMIN', 'TMAX'],<br/>  'units': {'source': 'K', 'cbf': 'K'},<br/>  'interpolation': 'linear',<br/>  'spatial_nature': 'continuous',<br/>  'physical_range': (173, 333),<br/>  'essential': True,<br/>  'variable_type': 'meteorological'<br/>}"]
    end

    subgraph TYPES["Variable Type Classification"]
        MET["METEOROLOGICAL<br/>T2M_MIN, T2M_MAX, VPD<br/>TOTAL_PREC, SSRD, STRD<br/>SKT, SNOWFALL, CO2"]

        FIRE["FIRE_EMISSIONS<br/>BURNED_AREA<br/>FIRE_C"]

        OBS["OBSERVATIONAL<br/>LAI, GPP, ABGB, EWT<br/>SOM, Mean_FIR"]
    end

    subgraph FIELDS["Metadata Fields per Variable"]
        F1["source: Data source<br/>(era5, noaa, gfed)"]
        F2["alternative_names<br/>Flexible naming"]
        F3["cbf_names: Target names<br/>in output files"]
        F4["units: Source & target<br/>units for conversion"]
        F5["interpolation_method<br/>linear or nearest"]
        F6["spatial_nature<br/>continuous/patchy"]
        F7["physical_range<br/>Valid value range"]
        F8["essential<br/>Required or optional"]
        F9["variable_type<br/>Classification"]
    end

    subgraph HELPERS["Helper Functions"]
        H1["get_variable_config()<br/>Retrieve full metadata"]
        H2["get_interpolation_method()<br/>Determine interpolation"]
        H3["get_cbf_name()<br/>Map source → target"]
        H4["get_variables_by_source()<br/>Filter by data source"]
        H5["get_variables_by_type()<br/>Filter by classification"]
        H6["validate_variable()<br/>Check if exists"]
    end

    subgraph CONSUMERS["Consumers of Registry"]
        DL["Downloaders<br/>• Lookup CBF names<br/>• Get units for conversion<br/>• Validate inputs"]

        STAC["STAC Utils<br/>• Get variable type<br/>• Populate metadata<br/>• Filter by type"]

        CBF["CBF Generator<br/>• Retrieve metadata<br/>• Get CBF names<br/>• Validate variables"]

        MET["Met Loader<br/>• Query by name<br/>• Filter by type<br/>• Validate exist"]
    end

    REGISTRY --> TYPES
    REGISTRY --> FIELDS
    REGISTRY --> HELPERS

    REGISTRY -.->|Feeds| DL
    REGISTRY -.->|Feeds| STAC
    REGISTRY -.->|Feeds| CBF
    REGISTRY -.->|Feeds| MET

    MET --> MET
    DL --> DL
    STAC --> STAC
    CBF --> CBF

    style REGISTRY fill:#d5e8d4
    style TYPES fill:#fff2cc
    style FIELDS fill:#e1d5e7
    style HELPERS fill:#dae8fc
    style CONSUMERS fill:#f8cecc
    style DL fill:#f8cecc
    style STAC fill:#f8cecc
    style CBF fill:#f8cecc
    style MET fill:#f8cecc
