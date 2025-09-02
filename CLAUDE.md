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
