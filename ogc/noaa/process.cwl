cwlVersion: v1.2

# ==============================================================================
# CARDAMOM NOAA CO2 Downloader - OGC Application Package for NASA MAAP
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
    id: cardamom-noaa-downloader
    label: CARDAMOM NOAA CO₂ Concentration Downloader

    doc: |
      Downloads global CO₂ concentration data from NOAA Global Monitoring Laboratory
      for CARDAMOM carbon cycle modeling. Produces analysis-ready NetCDF files with
      STAC metadata catalogs.

      CO₂ data is derived from the Mauna Loa Observatory and global sampling network
      (1974-present, monthly resolution).

      Output format: NetCDF with CF-1.8 conventions, monthly temporal resolution,
      global domain with 0.5° spatial resolution.

      STAC metadata includes:
      - Root catalog at outputs/catalog.json
      - Collections organized by measurement type
      - Items with comprehensive metadata

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

      # ====== OPTIONAL PARAMETERS (PUBLIC DATA - NO CREDENTIALS) ======

      year:
        type: int?
        doc: |
          Year to download CO₂ data (optional).

          CO₂ data is available from 1974 to present.
          If omitted, downloads entire available time series.

          Example: 2020

      month:
        type: int?
        doc: |
          Month to download (1-12, optional).

          Only meaningful if year is specified.
          Downloads single month of CO₂ data.
          If both year and month omitted, downloads all available data.

          Example: 1 (for January)

      # ====== STANDARD PROCESSING OPTIONS ======

      verbose:
        type: boolean?
        default: false
        doc: |
          Enable verbose debug logging.

          Prints detailed progress messages for troubleshooting.

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

    # ========================================================================
    # Workflow Outputs (OGC Interface)
    # ========================================================================

    outputs:

      outputs_result:
        type: Directory
        doc: |
          Complete output directory containing NOAA CO₂ data and STAC metadata.

          Directory structure:
            outputs/
            ├── catalog.json                      # Root STAC catalog
            ├── data/                             # Processed NetCDF files
            │   └── co2_concentration_YYYY_MM.nc
            └── cardamom-noaa-co2/                # STAC collection
                ├── collection.json               # Collection metadata
                └── items/
                    └── co2_YYYY_MM.json          # STAC items

          File descriptions:
            catalog.json: Root STAC catalog linking all collections
            data/*.nc: CO₂ concentration NetCDF files
              - CF-1.8 conventions compliant
              - Global 0.5° resolution
              - Monthly temporal resolution
              - Compressed with zlib
            cardamom-noaa-co2/collection.json: Collection metadata
            cardamom-noaa-co2/items/*.json: Individual STAC items with variable metadata

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
        run: "#main"
        in:
          year: year
          month: month
          verbose: verbose
          no_stac_incremental: no_stac_incremental
          stac_duplicate_policy: stac_duplicate_policy
        out: [outputs_result, stac_catalog]

  # ============================================================================
  # COMMANDLINETOOL (Execution Step)
  # ============================================================================

  - class: CommandLineTool
    id: main
    label: NOAA CO₂ Downloader Tool

    doc: |
      Executes the CARDAMOM NOAA CO₂ downloader within a Docker container.
      Downloads public CO₂ data and generates STAC metadata catalogs.

      Note: NOAA data is public - no authentication required.

    # ======================================================================
    # Runtime Requirements
    # ======================================================================

    requirements:
      DockerRequirement:
        dockerPull: ghcr.io/jpl-mghg/cardamom-preprocessor:latest

      ResourceRequirement:
        coresMin: 1
        ramMin: 4096   # 4GB RAM
        tmpdirMin: 5120    # 5GB temporary storage
        outdirMin: 10240   # 10GB output storage

      NetworkAccess:
        networkAccess: true

      EnvVarRequirement:
        envDef:
          PYTHONUNBUFFERED: "1"

    # ======================================================================
    # Tool Inputs (matched from Workflow)
    # ======================================================================

    inputs:

      year:
        type: int?
        inputBinding:
          prefix: --year

      month:
        type: int?
        inputBinding:
          prefix: --month

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

    baseCommand: ["/app/ogc/noaa/run_noaa.sh"]

    successCodes: [0]
