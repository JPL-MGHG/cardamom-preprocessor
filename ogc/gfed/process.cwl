cwlVersion: v1.2

# ==============================================================================
# CARDAMOM GFED Fire Emissions Downloader - OGC Application Package for NASA MAAP
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
    id: cardamom-gfed-downloader
    label: CARDAMOM GFED Fire Emissions Downloader

    doc: |
      Downloads Global Fire Emissions Database (GFED4) fire emissions data
      for CARDAMOM carbon cycle modeling. Processes yearly HDF5 files and
      produces analysis-ready NetCDF outputs with STAC metadata catalogs.

      GFED4 provides burned area and carbon emissions estimates (2001-present)
      at 0.25° spatial resolution, regridded to 0.5° for CARDAMOM analysis.

      Output format: NetCDF with CF-1.8 conventions, monthly temporal resolution,
      0.5° spatial resolution (global domain).

      STAC metadata includes:
      - Root catalog at outputs/catalog.json
      - Collections organized by variable type (burned area, emissions)
      - Items with comprehensive fire-specific metadata

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

      start_year:
        type: int
        doc: |
          First year to download (2001 or later).

          GFED4 data is available from 2001 onwards.
          Typical range: 2001-2024.

          Example: 2001

      end_year:
        type: int
        doc: |
          Last year to download (2024 or later with provisioning).

          Downloads all months in years from start_year to end_year inclusive.
          Maximum available year depends on current data provisioning.

          Example: 2024

      # ====== MAAP CREDENTIAL PARAMETERS (Optional) ======

      gfed_sftp_username:
        type: string?
        doc: |
          GFED SFTP username for authentication.

          OPTIONAL - If not provided, retrieved from MAAP platform secrets.
          Requires 'GFED_SFTP_USERNAME' secret configured in MAAP.

      gfed_sftp_password:
        type: string?
        doc: |
          GFED SFTP password for authentication.

          OPTIONAL - If not provided, retrieved from MAAP platform secrets.
          Requires 'GFED_SFTP_PASSWORD' secret configured in MAAP.

      # ====== OPTIONAL INPUT FILES ======

      land_sea_mask_file:
        type: File?
        doc: |
          Land-sea mask NetCDF file for spatial filtering.

          Optional file containing land fraction (0-1) to mask
          non-land pixels. If provided, only pixels above
          land_fraction_threshold are processed.

      # ====== STANDARD PROCESSING OPTIONS ======

      keep_raw:
        type: boolean?
        default: false
        doc: |
          If true, retains raw HDF5 files downloaded via SFTP.

          Raw GFED4 HDF5 files are deleted by default (save ~20GB per year).
          Set to true if you need raw data for custom processing.

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
          Complete output directory containing GFED fire emissions data and STAC metadata.

          Directory structure:
            outputs/
            ├── catalog.json                      # Root STAC catalog
            ├── data/                             # Processed NetCDF files
            │   ├── burned_area_YYYY_MM.nc
            │   └── carbon_emissions_YYYY_MM.nc
            └── cardamom-gfed-fire/               # STAC collection
                ├── collection.json               # Collection metadata
                └── items/                        # STAC items by variable
                    ├── burned_area_YYYY_MM.json
                    └── carbon_emissions_YYYY_MM.json

          File descriptions:
            catalog.json: Root STAC catalog linking all collections
            data/*.nc: GFED fire emissions NetCDF files
              - CF-1.8 conventions compliant
              - Global 0.5° resolution
              - Monthly temporal resolution
              - Burned area in fraction/m²
              - Carbon emissions in gC/m²/month
            cardamom-gfed-fire/collection.json: Collection metadata
            cardamom-gfed-fire/items/*.json: Individual STAC items with fire metadata

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
          start_year: start_year
          end_year: end_year
          gfed_sftp_username: gfed_sftp_username
          gfed_sftp_password: gfed_sftp_password
          land_sea_mask_file: land_sea_mask_file
          keep_raw: keep_raw
          verbose: verbose
          no_stac_incremental: no_stac_incremental
          stac_duplicate_policy: stac_duplicate_policy
        out: [outputs_result, stac_catalog]

  # ============================================================================
  # COMMANDLINETOOL (Execution Step)
  # ============================================================================

  - class: CommandLineTool
    id: main
    label: GFED Fire Emissions Downloader Tool

    doc: |
      Executes the CARDAMOM GFED downloader within a Docker container.
      Handles SFTP authentication, yearly HDF5 processing, and
      STAC metadata generation for fire emissions data.

    # ======================================================================
    # Runtime Requirements
    # ======================================================================

    requirements:
      DockerRequirement:
        dockerPull: ghcr.io/jpl-mghg/cardamom-preprocessor:latest

      ResourceRequirement:
        coresMin: 2
        ramMin: 16384   # 16GB RAM (for regridding operations)
        tmpdirMin: 20480    # 20GB temporary storage
        outdirMin: 102400   # 100GB output storage

      NetworkAccess:
        networkAccess: true

      EnvVarRequirement:
        envDef:
          PYTHONUNBUFFERED: "1"

    # ======================================================================
    # Tool Inputs (matched from Workflow)
    # ======================================================================

    inputs:

      start_year:
        type: int
        inputBinding:
          prefix: --start-year

      end_year:
        type: int
        inputBinding:
          prefix: --end-year

      gfed_sftp_username:
        type: string?
        inputBinding:
          prefix: --gfed_sftp_username

      gfed_sftp_password:
        type: string?
        inputBinding:
          prefix: --gfed_sftp_password

      land_sea_mask_file:
        type: File?
        inputBinding:
          prefix: --land-sea-mask-file

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

    baseCommand: ["/app/ogc/gfed/run_gfed.sh"]

    successCodes: [0]
