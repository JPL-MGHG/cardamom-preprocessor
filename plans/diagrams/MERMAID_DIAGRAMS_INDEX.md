# Mermaid Diagrams Index

This directory contains both Draw.io diagrams and Mermaid markdown diagrams for the CARDAMOM preprocessor system. Mermaid diagrams are text-based and render natively in GitHub, Markdown editors, and many documentation systems.

## Benefits of Mermaid Diagrams

✅ **Version Control Friendly**: Pure text, easy to diff and track changes
✅ **GitHub Native**: Renders automatically in README and markdown files
✅ **No Tool Required**: View in any markdown viewer or editor
✅ **Lightweight**: Small file sizes
✅ **Quick to Edit**: Just text, no UI needed
❌ **Less Styleable**: Limited color/formatting options vs Draw.io

## Mermaid Diagram Files

### Architecture Diagrams (Mermaid)

#### 1. [system_architecture_mermaid.md](./system_architecture_mermaid.md)
**System Architecture Overview**

5-layer architecture diagram:
- Layer 1: Data Acquisition (ECMWF, NOAA, GFED)
- Layer 2: STAC Discovery (catalog and queries)
- Layer 3: Data Loading (meteorology and observations)
- Layer 4: Data Assembly (regridding, alignment)
- Layer 5: CBF Generation (pixel processing)

**Best for**: Understanding overall system design and data flow through layers
**Format**: Flowchart with color-coded layers

---

#### 2. [cbf_generator_mermaid.md](./cbf_generator_mermaid.md)
**CBF Generator Complete Workflow**

Detailed workflow from CLI to CBF output:
- Parallel meteorology (required) and observation (optional) paths
- Time coordinate alignment
- Graceful degradation strategy
- Pixel processing loop
- CBF file output

**Best for**: Understanding CBF generation workflow and validation strategy
**Format**: Complex flowchart with decision points

---

#### 3. [meteorology_discovery_mermaid.md](./meteorology_discovery_mermaid.md)
**Meteorology Discovery & Loading (4 Phases)**

Phase-by-phase breakdown:
- **Phase 1: Discovery** - STAC queries with metadata filtering
- **Phase 2: Loading** - Handle 3 temporal structures
- **Phase 3: Validation** - CRITICAL completeness check
- **Phase 4: Assembly** - Regrid, normalize, mask

**Best for**: Understanding STAC-based discovery and validation
**Format**: Flowchart with 4 distinct phases

---

#### 4. [time_coordinate_mermaid.md](./time_coordinate_mermaid.md)
**Time Coordinate Alignment: OLD vs NEW**

Comparison diagram:
- OLD architecture: Scaffold as time authority
- NEW architecture: Meteorology as time authority
- Issues with OLD approach
- Benefits of NEW approach

**Best for**: Understanding architectural shift in time authority
**Format**: Side-by-side comparison flowchart

---

#### 5. [observational_data_mermaid.md](./observational_data_mermaid.md)
**Observational Data Handling with Graceful Degradation**

Graceful degradation strategy:
- Input files (all optional)
- Loading strategy (NaN-fill)
- Variable processing and naming
- Pixel-level extraction
- Degradation scenarios

**Best for**: Understanding optional data handling and NaN-fill strategy
**Format**: Flowchart with multiple degradation paths

---

#### 6. [variable_registry_mermaid.md](./variable_registry_mermaid.md)
**Variable Registry System: Single Source of Truth**

Central registry documentation:
- Variable entry example
- Type classification (meteorological, fire_emissions, observational)
- Metadata fields per variable
- Helper functions
- Registry consumers

**Best for**: Understanding variable metadata organization
**Format**: Flowchart with central hub and consumers

---

#### 7. [stac_catalog_mermaid.md](./stac_catalog_mermaid.md)
**STAC Catalog Structure & Organization**

STAC directory structure:
- Type-based collections
- STAC item properties (standard and custom)
- Discovery process
- Collection update strategy

**Best for**: Understanding STAC organization and discovery
**Format**: Hierarchical diagram with structure and processes

---

## Comparison: Draw.io vs Mermaid

| Feature | Draw.io XML | Mermaid Markdown |
|---------|-------------|------------------|
| **Visual Styling** | Full color, shapes, positioning | Limited (basic colors) |
| **Interactive Editor** | Yes (draw.io web/desktop) | Text editor only |
| **Version Control** | XML format (harder to diff) | Pure text (easy to diff) |
| **GitHub Native** | Links to viewer needed | Renders natively |
| **File Size** | Larger (XML) | Smaller (text) |
| **Editing Speed** | Visual, slower | Fast (text) |
| **Collaboration** | Good with draw.io | Excellent with git |
| **Print Quality** | Excellent | Good |

## How to View Mermaid Diagrams

### GitHub / GitLab / Gitea
Mermaid diagrams render automatically in markdown files.

### VS Code
Install "Markdown Preview Mermaid Support" extension

### Local Markdown Viewer
Use any Markdown editor that supports Mermaid (many modern ones do)

### Online Editor
Visit [mermaid.live](https://mermaid.live) and paste the diagram code

## Converting Between Formats

### Draw.io to Mermaid
Mermaid diagrams provide **simplified, text-based versions** of Draw.io diagrams.
- Mermaid versions are easier to version control and edit
- Draw.io versions provide more detailed visual styling
- Both show the same information, just in different formats

### Adding New Diagrams

For new architecture diagrams:
1. **Create Draw.io diagram first** (detailed, styled)
2. **Create Mermaid version** (text-based, simplified)
3. Both provide different perspectives on the same system

## Quick Navigation

**I want to understand:**

- **Overall System Design** → [system_architecture_mermaid.md](./system_architecture_mermaid.md)
- **CBF File Generation** → [cbf_generator_mermaid.md](./cbf_generator_mermaid.md)
- **STAC Data Discovery** → [meteorology_discovery_mermaid.md](./meteorology_discovery_mermaid.md)
- **Variable Metadata** → [variable_registry_mermaid.md](./variable_registry_mermaid.md)
- **STAC Organization** → [stac_catalog_mermaid.md](./stac_catalog_mermaid.md)
- **Time Handling** → [time_coordinate_mermaid.md](./time_coordinate_mermaid.md)
- **Missing Data Strategy** → [observational_data_mermaid.md](./observational_data_mermaid.md)

## Related Documentation

- **Draw.io Diagrams**: See [README.md](./README.md) for full list and descriptions
- **Architecture Details**: [CLAUDE.md](../../CLAUDE.md) - Coding standards and design patterns
- **Variable System**: [cardamom_variables.py](../../src/cardamom_variables.py) - Registry implementation
- **STAC Implementation**: [stac_utils.py](../../src/stac_utils.py) - Catalog management
- **CBF Generation**: [cbf_main.py](../../src/cbf_main.py) - Generator implementation

## Statistics

- **Total Diagrams**: 17 (11 Draw.io + 7 Mermaid + shared index)
- **Architecture Diagrams**: 6 (3 Draw.io only + 6 Mermaid)
- **Variable Flow Diagrams**: 11 (all Draw.io, representative template updated)
- **Lines of Code**: ~1,500+ lines of Mermaid diagram definitions

## Recent Updates

**December 2025**: Created comprehensive Mermaid diagrams for all architecture components:
- Converted key Draw.io diagrams to Mermaid format
- Maintained consistency with draw.io versions
- Added this index for easy navigation
- Text-based format enables better version control

---

*For visual, feature-rich diagrams: Use Draw.io versions*
*For collaborative editing and version control: Use Mermaid versions*
*Both describe the same system architecture*
