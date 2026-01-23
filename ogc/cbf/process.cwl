cwlVersion: v1.2

# ==============================================================================
# CARDAMOM CBF File Generator - OGC Application Package for NASA MAAP
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
    id: cardamom-cbf-generator
    label: CARDAMOM CBF File Generator

    doc: |
      Generates CARDAMOM CBF (CARDAMOM Binary Format) input files for
      carbon cycle data assimilation. Consumes STAC catalogs from CARDAMOM
      preprocessor downloaders (ECMWF, NOAA, GFED) and user-provided
      observational constraint files.

      CBF files contain:
      - Meteorological forcing variables (temperature, precipitation, radiation)
      - Observational constraints (LAI, GPP, biomass, soil moisture)
      - Single-value constraints (soil carbon, plant carbon use efficiency)
      - MCMC configuration parameters

      Output format: NetCDF files following CARDAMOM specifications,
      one file per valid land pixel.

      Processing pipeline:
      1. Discover meteorological variables from STAC catalog
      2. Load observational constraint data (optional, with graceful NaN-filling)
      3. Extract pixel-level data for each valid land pixel
      4. Generate CARDAMOM-ready CBF NetCDF files

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

      stac_api:
        type: string
        doc: |
          STAC API endpoint or catalog path for meteorological data discovery.

          Can be either:
          - HTTPS URL: https://stac.maap-project.org
          - Local file path: file:///path/to/catalog.json

          Typically points to STAC catalog output from ECMWF/NOAA/GFED downloaders.
          Example: file:///outputs/cardamom-ecmwf/catalog.json

      start:
        type: string
        doc: |
          Start date in YYYY-MM format (inclusive).

          Beginning of meteorological time period for CBF generation.
          Must correspond to available data in STAC catalog.

          Example: 2020-01

      end:
        type: string
        doc: |
          End date in YYYY-MM format (inclusive).

          End of meteorological time period for CBF generation.
          Must correspond to available data in STAC catalog.

          Example: 2020-12

      # ====== OPTIONAL PARAMETERS ======

      region:
        type: string?
        default: conus
        doc: |
          Geographic region for processing.

          Options:
            - conus: Continental United States + Canada
            - global: Global (all land pixels)

          Controls spatial bounds and pixel filtering.

      land_fraction_file:
        type: File?
        doc: |
          Land-sea fraction mask NetCDF file.

          Optional file containing land fraction (0-1) for each pixel.
          Only pixels with fraction > land_fraction_threshold are processed.
          If omitted, uses default global land mask.

      obs_driver_file:
        type: File?
        doc: |
          Observational constraints NetCDF file.

          Optional file containing observational data variables:
          - LAI: Leaf Area Index (m²/m²)
          - GPP: Gross Primary Productivity (gC/m²/day)
          - ABGB: Aboveground Biomass (gC/m²)
          - EWT: Equivalent Water Thickness (mm)

          Missing data is gracefully NaN-filled for degraded processing.

      som_file:
        type: File?
        doc: |
          Soil Organic Matter initialization file.

          Optional file containing initial soil carbon (gC/m²)
          for CARDAMOM carbon cycle initialization.

      fir_file:
        type: File?
        doc: |
          Fire emissions file.

          Optional file containing mean fire-induced carbon release.

      scaffold_file:
        type: File?
        doc: |
          CBF template file for copying attributes and structure.

          Optional NetCDF scaffold file containing CARDAMOM-standard
          variable definitions and attributes. Used as template for
          generated CBF files.

      # ====== PROCESSING OPTIONS ======

      verbose:
        type: boolean?
        default: false
        doc: |
          Enable verbose debug logging.

          Prints detailed progress for pixel processing and data discovery.

    # ========================================================================
    # Workflow Outputs (OGC Interface)
    # ========================================================================

    outputs:

      outputs_result:
        type: Directory
        doc: |
          Complete output directory containing generated CBF files.

          Directory structure:
            outputs/
            └── site*.cbf.nc              # CARDAMOM CBF files (one per pixel)

          Filename format:
            siteYY_YYNYY_YYWD_IDEXP#exp0.cbf.nc
            where:
              YY_YY = latitude (integer_decimal format)
              NYY_YYW = longitude (integer_decimal format)
              EXP# = experiment ID

          File contents:
            - Meteorological forcing variables (monthly time series)
            - Observational constraints (optional, monthly)
            - Single-value constraints (scalars: SOM, CUE, LAI, FIR)
            - MCMC configuration (iterations, samples)

          Each CBF file is CARDAMOM-ready for carbon cycle data assimilation.

        outputSource: generate_step/outputs_result

    # ========================================================================
    # Workflow Steps
    # ========================================================================

    steps:
      generate_step:
        run: "#main"
        in:
          stac_api: stac_api
          start: start
          end: end
          region: region
          land_fraction_file: land_fraction_file
          obs_driver_file: obs_driver_file
          som_file: som_file
          fir_file: fir_file
          scaffold_file: scaffold_file
          verbose: verbose
        out: [outputs_result]

  # ============================================================================
  # COMMANDLINETOOL (Execution Step)
  # ============================================================================

  - class: CommandLineTool
    id: main
    label: CBF File Generator Tool

    doc: |
      Executes the CARDAMOM CBF file generator within a Docker container.
      Discovers meteorological data from STAC catalog, loads observational
      constraints, and generates pixel-specific CBF files for CARDAMOM
      carbon cycle data assimilation.

      Note: No external credentials required (reads from STAC catalog and files).

    # ======================================================================
    # Runtime Requirements
    # ======================================================================

    requirements:
      DockerRequirement:
        dockerPull: ghcr.io/jpl-mghg/cardamom-preprocessor:latest

      ResourceRequirement:
        coresMin: 4
        ramMin: 16384   # 16GB RAM (pixel-level processing)
        tmpdirMin: 10240    # 10GB temporary storage
        outdirMin: 20480    # 20GB output storage

      NetworkAccess:
        networkAccess: false  # No external API calls needed

      EnvVarRequirement:
        envDef:
          PYTHONUNBUFFERED: "1"

    # ======================================================================
    # Tool Inputs (matched from Workflow)
    # ======================================================================

    inputs:

      stac_api:
        type: string
        inputBinding:
          prefix: --stac-api

      start:
        type: string
        inputBinding:
          prefix: --start

      end:
        type: string
        inputBinding:
          prefix: --end

      region:
        type: string?
        default: conus
        inputBinding:
          prefix: --region

      land_fraction_file:
        type: File?
        inputBinding:
          prefix: --land-fraction-file

      obs_driver_file:
        type: File?
        inputBinding:
          prefix: --obs-driver-file

      som_file:
        type: File?
        inputBinding:
          prefix: --som-file

      fir_file:
        type: File?
        inputBinding:
          prefix: --fir-file

      scaffold_file:
        type: File?
        inputBinding:
          prefix: --scaffold-file

      verbose:
        type: boolean?
        default: false
        inputBinding:
          prefix: --verbose

    # ======================================================================
    # Tool Outputs
    # ======================================================================

    outputs:

      outputs_result:
        type: Directory
        outputBinding:
          glob: outputs

    # ======================================================================
    # Command Execution
    # ======================================================================

    baseCommand: ["/app/ogc/cbf/run_cbf.sh"]

    successCodes: [0]
