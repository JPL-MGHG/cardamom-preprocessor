# Observational Data Handling - Mermaid Diagram

## Graceful Degradation Strategy

```mermaid
graph TD
    subgraph INPUT["Input Files (ALL Optional)"]
        OBSFILE["Main Obs File (Optional)<br/>Contains:<br/>• LAI (Leaf Area Index)<br/>• GPP (Gross Primary Prod)<br/>• ABGB (Above-Ground Biomass)<br/>• EWT (Equiv Water Thickness)<br/><br/>If missing: All → NaN"]

        SOMFILE["SOM File (Optional)<br/>Contains:<br/>• SOM (soil carbon)<br/>From HWSD database<br/><br/>If missing: → NaN"]

        FIRFILE["FIR File (Optional)<br/>Contains:<br/>• Mean_FIR (fire emissions)<br/>From GFED4.1s<br/><br/>If missing: → NaN"]
    end

    subgraph LOAD["Loading Strategy: NaN-Fill"]
        LOGIC["for each optional file:<br/>    try:<br/>        load_file(path)<br/>    except FileNotFoundError:<br/>        Fill with NaN array<br/><br/>Result: Unified obs dataset<br/>with NaN for all missing"]

        RATIONALE["Scientific Rationale<br/>Observations are OPTIONAL:<br/>• Forward-only allowed<br/>• Data assimilation optional<br/>• Graceful degradation OK<br/><br/>Contrasts with Meteorology:<br/>• Meteorology is REQUIRED<br/>(FAIL if missing)<br/>• Observations are OPTIONAL<br/>(NaN-fill if missing)"]
    end

    subgraph PROCESS["Variable Processing & Naming"]
        VARPROC["For each variable:<br/><br/>1. Check if loaded successfully<br/>   ✓ Keep: Rename to CBF<br/>   ✗ NaN: Fill array with NaN<br/><br/>2. CBF Variable Renaming<br/>   ModLAI → LAI<br/>   GPPFluxSat → GPP<br/>   (See CARDAMOM_VARIABLE_REGISTRY)<br/><br/>3. Preserve Attributes<br/>   Copy from scaffold template<br/>   Override with data attrs if present"]
    end

    subgraph PIXEL["Pixel-Level Extraction"]
        EXTRACT["Extract with NaN Fallback<br/><br/>For each pixel (lat_idx, lon_idx):<br/>  for each obs variable:<br/>      try:<br/>          data = obs_dataset[var].isel(...)<br/>      except (KeyError, IndexError):<br/>          data = NaN<br/><br/>Result: Time series or NaN<br/>for each variable"]
    end

    subgraph SCENARIOS["Graceful Degradation Scenarios"]
        SC1["Scenario 1: All Obs Missing<br/>→ All NaN<br/>→ Forward-only mode"]

        SC2["Scenario 2: Partial Coverage<br/>→ Mix of data and NaN<br/>→ Incomplete data assimilation"]

        SC3["Scenario 3: Complete Coverage<br/>→ Full data assimilation<br/>capability"]

        SC4["Scenario 4: Temporal Mismatch<br/>→ NaN-fill unobserved steps<br/>→ Obs subset aligns to met"]

        VALID["✓ All scenarios scientifically valid<br/>✓ Carbon modeling proceeds<br/>with available data<br/>✓ Transparent about uncertainty"]
    end

    OBSFILE --> LOGIC
    SOMFILE --> LOGIC
    FIRFILE --> LOGIC

    LOGIC --> RATIONALE
    RATIONALE --> VARPROC

    VARPROC --> EXTRACT

    EXTRACT --> SC1
    EXTRACT --> SC2
    EXTRACT --> SC3
    EXTRACT --> SC4

    SC1 --> VALID
    SC2 --> VALID
    SC3 --> VALID
    SC4 --> VALID

    style INPUT fill:#add1fe
    style LOAD fill:#d5e8d4
    style PROCESS fill:#e1d5e7
    style PIXEL fill:#f8cecc
    style SCENARIOS fill:#fff2cc
    style VALID fill:#d5e8d4
    style RATIONALE fill:#fff2cc
