cwlVersion: v1.2

# ==============================================================================
# CARDAMOM ECMWF ERA5 Downloader - OGC Application Package for NASA MAAP
# ==============================================================================

$namespaces:
  s: https://schema.org/

$schemas:
  - http://schema.org/version/latest/schemaorg-current-https.rdf

$graph:

  # ============================================================================
  # WORKFLOW (Entry Point) - OGC Application Package Interface
  # ============================================================================

  - class: Workflow
    id: cardamom-ecmwf-downloader
    label: CARDAMOM ECMWF ERA5 Meteorological Data Downloader

    doc: |
      Downloads ERA5 reanalysis meteorological variables from ECMWF Climate Data Store
      for CARDAMOM carbon cycle modeling. Produces analysis-ready NetCDF files with
      STAC metadata catalogs organized by variable type.

      Supported meteorological variables:
      - t2m_min, t2m_max: 2-meter temperature extrema (Kelvin)
      - vpd: Vapor Pressure Deficit (hectopascals)
      - total_prec: Total precipitation (millimeters)
      - ssrd: Surface solar radiation downwards (W/m²)
      - strd: Surface thermal radiation downwards (W/m²)
      - skt: Skin temperature (Kelvin)
      - snowfall: Snowfall (millimeters)

      Output format: NetCDF with CF-1.8 conventions, monthly temporal resolution,
      0.5° spatial resolution (global domain).

      STAC metadata includes:
      - Root catalog at outputs/catalog.json
      - Collections organized by variable type (e.g., cardamom-meteorology)
      - Items with comprehensive variable-specific metadata

    # Schema.org Metadata for Discoverability
    s:softwareVersion: "1.0.0"
    s:datePublished: "2026-01-04"
    s:author:
      - class: s:Person
        s:name: CARDAMOM Development Team
        s:email: support@maap-project.org

    s:contributor:
      - class: s:Person
        s:name: MAAP Platform Team

    s:codeRepository: https://github.com/JPL-MGHG/cardamom-preprocessor
    s:license: https://opensource.org/licenses/Apache-2.0

    # ========================================================================
    # Workflow Inputs (OGC Interface)
    # ========================================================================

    inputs:

      # ====== REQUIRED PARAMETERS ======

      variables:
        type: string
        doc: |
          Comma-separated list of CARDAMOM meteorological variables to download.

          Available variables:
            - t2m_min: 2-meter minimum temperature (Kelvin)
            - t2m_max: 2-meter maximum temperature (Kelvin)
            - vpd: Vapor Pressure Deficit (hectopascals) [derived from T_max, T_dewpoint]
            - total_prec: Total precipitation (millimeters)
            - ssrd: Surface solar radiation downwards (W/m²)
            - strd: Surface thermal radiation downwards (W/m²)
            - skt: Skin temperature (Kelvin)
            - snowfall: Snowfall (millimeters)

          Example: "t2m_min,t2m_max,vpd" or "t2m_min,t2m_max,vpd,total_prec,ssrd,strd,skt,snowfall"

          Note: Multiple variables are downloaded in a single CDS API request for efficiency.

      year:
        type: int
        doc: |
          Year to download meteorological data.

          ERA5 data is available from 1940 to present (updated monthly).
          Common years for CARDAMOM: 2000-2024.

          Example: 2020

      month:
        type: int
        doc: |
          Month to download (1-12).

          Downloads a single month of meteorological data. For multi-month datasets,
          submit multiple jobs with different month values, or orchestrate via
          CWL Workflow with scatter.

          Example: 1 (for January)

      # ====== MAAP CREDENTIAL PARAMETERS (Optional) ======

      ecmwf_cds_key:
        type: string?
        doc: |
          ECMWF CDS API Key for authentication.

          OPTIONAL - If not provided, retrieved from MAAP platform secrets.
          See ecmwf_cds_uid for credential management details.

      # ====== OPTIONAL PARAMETERS ======

      keep_raw:
        type: boolean?
        default: false
        doc: |
          If true, retains raw ERA5 files downloaded from CDS after processing.

          Raw ERA5 files are deleted by default (save ~5GB per variable-month).
          Set to true if you need raw NetCDF files for custom processing.

          Note: Keep raw files only if needed, as they significantly increase output size.

      verbose:
        type: boolean?
        default: false
        doc: |
          Enable verbose debug logging.

          Prints detailed progress messages for troubleshooting.
          Useful for diagnosing CDS API issues or download failures.

      no_stac_incremental:
        type: boolean?
        default: false
        doc: |
          Disable incremental STAC catalog updates.

          By default (false), new data is merged into existing STAC catalogs.
          Set to true to overwrite the entire catalog (useful for rebuilding).

      stac_duplicate_policy:
        type: string?
        default: "update"
        doc: |
          How to handle duplicate STAC items when incremental mode is enabled.

          Choices:
            - update: Replace existing item with new data (default, recommended)
            - skip: Keep existing item, ignore new download
            - error: Raise error and require user decision

          Set to "update" for automated re-processing, "skip" for incremental builds.

    # ========================================================================
    # Workflow Outputs (OGC Interface)
    # ========================================================================

    outputs:

      outputs_result:
        type: Directory
        doc: |
          Complete output directory containing:

          Directory structure:
            outputs/
            ├── catalog.json                      # Root STAC catalog
            ├── data/                             # Processed NetCDF files
            │   ├── t2m_min_YYYY_MM.nc
            │   ├── t2m_max_YYYY_MM.nc
            │   └── vpd_YYYY_MM.nc
            └── cardamom-meteorology/             # STAC collection (by variable type)
                ├── collection.json               # Collection metadata
                └── items/                        # STAC items
                    ├── t2m_min_YYYY_MM.json
                    ├── t2m_max_YYYY_MM.json
                    └── vpd_YYYY_MM.json

          File descriptions:

            catalog.json: Root STAC catalog linking all collections and items.
              - Provides entry point for STAC catalog discovery
              - Links to variable-type collections

            data/*.nc: Processed NetCDF files
              - CF-1.8 conventions compliant
              - Dimensions: latitude (360), longitude (720), time (1 month)
              - Global 0.5° resolution
              - Compressed with zlib (complevel=4)
              - Fill value: -9999.0

            cardamom-meteorology/collection.json: Variable-type collection metadata
              - Describes collection extent (spatial, temporal)
              - Lists variable types included
              - Provides collection-level licensing/attribution

            cardamom-meteorology/items/*.json: Individual STAC items
              - One item per variable per time step
              - Includes variable-specific metadata:
                - cardamom:variable, cardamom:units, cardamom:source
                - cardamom:processing steps, validation results
              - Links to corresponding NetCDF file

          STAC catalog organization follows OGC Best Practices and enables:
            - Automated data discovery and harvesting
            - Integration with STAC-aware platforms
            - Machine-readable metadata for CARDAMOM processing pipelines
            - Incremental catalog updates for multi-run workflows

        outputSource: download_step/outputs_result

      stac_catalog:
        type: File
        doc: |
          Root STAC catalog JSON file at outputs/catalog.json.
          Entry point for catalog discovery and validation.
        outputSource: download_step/stac_catalog

    # ========================================================================
    # Workflow Steps
    # ========================================================================

    steps:
      download_step:
        run: "#download-ecmwf"
        in:
          variables: variables
          year: year
          month: month
          ecmwf_cds_key: ecmwf_cds_key
          keep_raw: keep_raw
          verbose: verbose
          no_stac_incremental: no_stac_incremental
          stac_duplicate_policy: stac_duplicate_policy
        out: [outputs_result, stac_catalog]

  # ============================================================================
  # COMMANDLINETOOL (Execution Step)
  # ============================================================================

  - class: CommandLineTool
    id: download-ecmwf
    label: ECMWF ERA5 Download Tool

    doc: |
      Executes the CARDAMOM ECMWF downloader within a Docker container.
      Handles CDS API authentication, meteorological data download, and
      STAC metadata generation.

    # ======================================================================
    # Runtime Requirements
    # ======================================================================

    requirements:
      DockerRequirement:
        dockerPull: ghcr.io/jpl-mghg/cardamom-preprocessor-ecmwf:latest

      ResourceRequirement:
        coresMin: 2
        ramMin: 8192  # 8GB RAM
        tmpdirMin: 10240  # 10GB temporary storage
        outdirMin: 51200  # 50GB output storage

      NetworkAccess:
        networkAccess: true

      EnvVarRequirement:
        envDef:
          PYTHONUNBUFFERED: "1"

    # ======================================================================
    # Tool Inputs (matched from Workflow)
    # ======================================================================

    inputs:

      variables:
        type: string
        inputBinding:
          prefix: --variables

      year:
        type: int
        inputBinding:
          prefix: --year

      month:
        type: int
        inputBinding:
          prefix: --month

      ecmwf_cds_key:
        type: string?
        inputBinding:
          prefix: --ecmwf_cds_key

      keep_raw:
        type: boolean?
        default: false
        inputBinding:
          prefix: --keep-raw

      verbose:
        type: boolean?
        default: false
        inputBinding:
          prefix: --verbose

      no_stac_incremental:
        type: boolean?
        default: false
        inputBinding:
          prefix: --no-stac-incremental

      stac_duplicate_policy:
        type: string?
        default: "update"
        inputBinding:
          prefix: --stac-duplicate-policy

    # ======================================================================
    # Tool Outputs
    # ======================================================================

    outputs:

      outputs_result:
        type: Directory
        outputBinding:
          glob: outputs

      stac_catalog:
        type: File
        outputBinding:
          glob: outputs/catalog.json

    # ======================================================================
    # Command Execution
    # ======================================================================

    baseCommand: ["/app/ogc/ecmwf/run_ecmwf.sh"]

    successCodes: [0]
