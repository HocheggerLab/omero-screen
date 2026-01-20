# OMERO-Screen Project Overview

## Project Summary
OMERO-Screen is a comprehensive mono-repo for high-content image analysis of immunofluorescence microscopy data. It provides an end-to-end pipeline from image segmentation to statistical analysis and visualization.

## Architecture Overview

### Main Pipeline (`src/omero_screen/`)
- **Core functionality**: End-to-end high-content image analysis pipeline
- **Key components**:
  - `image_analysis.py`: Cellpose-based segmentation for nuclei and cells
  - `flatfield_corr.py`: Flatfield correction for microscopy images
  - `metadata_parser.py`: Extracts experimental metadata from OMERO
  - `plate_dataset.py`: Manages plate-based screening data
  - `cellcycle_analysis.py`: Cell cycle phase classification
  - `aggregator.py`: Aggregates single-cell measurements
  - `quality_control.py`: QC metrics for images
- **Segmentation models**: Uses Cellpose models, automatically selected based on cell line and magnification
- **Data flow**: OMERO images → Segmentation → Feature extraction → CSV export

### Package Structure

#### 1. `omero-utils` (Helper Functions)
- **Purpose**: Utility functions for OMERO server interaction
- **Key modules**:
  - `omero_connect.py`: Connection management
  - `attachments.py`: File upload/download from OMERO
  - `map_anns.py`: Metadata annotations
  - `images.py`: Image handling utilities
  - `omero_plate.py`: Plate-specific operations

#### 2. `cellview` (Database Layer)
- **Purpose**: DuckDB-based storage for single-cell measurements
- **Features**:
  - Import CSV data from OMERO-Screen pipeline
  - Organize by project → experiment → plate → condition
  - Support for biological replicates
  - Fast local querying and data export
- **Access**: CLI (`cellview` command) or Python API (`cellview.api`)

#### 3. `omero-screen-plots` (Analysis & Visualization)
- **Purpose**: Standardized plotting and statistical analysis
- **Plot types**:
  - `featureplot.py`: Box/violin plots for any measured feature
  - `cellcycleplot.py`: Cell cycle distribution analysis
  - `combplot.py`: Combined plots (scatter, histogram, etc.)
  - `countplot.py`: Cell counting statistics
  - `normalise.py`: Data normalization utilities
- **Data normalization**:
  - `scale_data()` function uses percentile-based clipping (1st-99th)
  - Scales to 16-bit range (0-65535)
  - Handles outliers from hot pixels/debris
- **Style**: Custom matplotlib style for consistent figures

#### 4. `omero-screen-napari` (UI & Classification)
- **Purpose**: Napari plugin for interactive visualization
- **Planned features**:
  - Interactive cell classification
  - Training data generation for ML models
  - Visual QC of segmentation results

## Key Technical Details

### Segmentation Strategy
- **Nucleus channel**: Cellpose nucleus model
- **Cell channel**: Cellpose cyto2 model (or custom trained)
- **Cytoplasm**: Calculated as cell mask minus nucleus mask
- **Model selection**: Based on metadata (cell line, magnification)

### Data Processing
- **Flatfield correction**: Per-channel correction using pre-calculated masks
- **Feature extraction**: Area, intensity, shape metrics via scikit-image
- **Metadata**: Preserves plate layout, conditions, timepoints

### Quality Control
- Automatic mask upload to OMERO
- Image classification for QC
- Intensity and segmentation metrics

## Environment Configuration
- Uses `.env.{environment}` files (development, production, e2etest)
- Required: OMERO credentials, logging settings
- Test server: 127.0.0.2:4064 (parallel to main server)

## Testing Strategy
- **Unit tests**: Component-level testing
- **E2E tests**: Full pipeline validation with test OMERO server
- **Test data**: Example 2D/3D/timeseries plates included

## Commands & Workflows

### Main Pipeline
```bash
omero-screen  # Run full analysis pipeline
```

### Database Operations
```bash
cellview import-csv <file>  # Import data
cellview display plates     # Show available data
cellview export <plate_id>  # Export to CSV/Excel
```

### Testing
```bash
pytest -v  # Unit tests
omero-integration-test <test_name>  # E2E tests
```

## Development Notes
- Python 3.12+ required
- Uses uv for dependency management
- Pre-commit hooks for code quality
- Semantic versioning with conventional commits

## Project Memory System

This project maintains institutional knowledge in `.project_notes/` for consistency across sessions.

### Memory Files

- **bugs.md** - Bug log with dates, solutions, and prevention notes
- **decisions.md** - Architectural Decision Records (ADRs) with context and trade-offs
- **key_facts.md** - Project configuration, credentials, ports, important URLs
- **issues.md** - Work log with ticket IDs, descriptions, and URLs

### Memory-Aware Protocols

**Before proposing architectural changes:**
- Check `.project_notes/decisions.md` for existing decisions
- Verify the proposed approach doesn't conflict with past choices
- If it does conflict, acknowledge the existing decision and explain why a change is warranted

**When encountering errors or bugs:**
- Search `.project_notes/bugs.md` for similar issues
- Apply known solutions if found
- Document new bugs and solutions when resolved

**When looking up project configuration:**
- Check `.project_notes/key_facts.md` for credentials, ports, URLs, service accounts
- Prefer documented facts over assumptions

**When completing work on tickets:**
- Log completed work in `.project_notes/issues.md`
- Include ticket ID, date, brief description, and URL

**When user requests memory updates:**
- Update the appropriate memory file (bugs, decisions, key_facts, or issues)
- Follow the established format and style (bullet lists, dates, concise entries)

### Style Guidelines for Memory Files

- **Prefer bullet lists over tables** for simplicity and ease of editing
- **Keep entries concise** (1-3 lines for descriptions)
- **Always include dates** for temporal context
- **Include URLs** for tickets, documentation, monitoring dashboards
- **Manual cleanup** of old entries is expected (not automated)
