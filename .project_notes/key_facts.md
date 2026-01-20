# Project Key Facts

Essential project configuration, URLs, ports, and constants. Check here before making assumptions.

## IMPORTANT: Security Guidelines

**DO NOT store sensitive information in this file:**
- Passwords, API keys, tokens, or secrets
- Production credentials
- Private keys or certificates

**DO store:**
- Public URLs and endpoints
- Port numbers and service configurations
- Non-sensitive configuration values
- Links to secure credential storage (e.g., "See .env.production for DB password")

---

## OMERO Server Configuration

### Production Server
- **Host**: (Add your production OMERO server host)
- **Port**: 4064
- **Web URL**: (Add your OMERO.web URL)
- **Credentials**: See `.env.production` file (not in git)

### Test Server (E2E Tests)
- **Host**: 127.0.0.2
- **Port**: 4064
- **Purpose**: Parallel OMERO server for integration tests
- **Setup**: See `tests/e2e/README.md` for startup instructions

## Package Structure

### Main Packages
- **omero-screen**: Core analysis pipeline (`src/omero_screen/`)
- **omero-utils**: OMERO helper functions (`packages/omero-utils/`)
- **cellview**: DuckDB database layer (`packages/cellview/`)
- **omero-screen-plots**: Plotting and analysis (`packages/omero-screen-plots/`)
- **omero-screen-napari**: Napari plugin (planned) (`packages/omero-screen-napari/`)

### Key Entry Points
- `omero-screen` command: Main pipeline CLI
- `cellview` command: Database operations CLI
- `omero-integration-test` command: E2E test runner

## Segmentation Models

### Cellpose Model Selection
- **Nucleus channel**: Uses `nuclei` model by default
- **Cell channel**: Uses `cyto2` model by default
- **Custom models**: Selected based on cell line + magnification metadata
- **Model storage**: (Add path to custom model storage if applicable)

## Data Processing Defaults

### Normalization (omero-screen-plots)
- **Method**: Percentile-based clipping (1st-99th percentile)
- **Output range**: 16-bit (0-65535)
- **Purpose**: Handle outliers from hot pixels and debris

### Quality Control
- **Mask upload**: Automatically uploaded to OMERO after segmentation
- **Image classification**: Uses QC metrics from `quality_control.py`

## Development Environment

### Python Version
- **Required**: Python 3.12+
- **Package manager**: uv (for dependency management)

### Environment Files
- `.env.development`: Development configuration
- `.env.production`: Production configuration
- `.env.e2etest`: E2E test configuration

### Testing
- **Unit tests**: `pytest -v`
- **E2E tests**: `omero-integration-test <test_name>`
- **Test data**: Example plates in `tests/data/`

## External Dependencies

### Key Libraries
- **Cellpose**: Cell segmentation
- **DuckDB**: Single-cell data storage
- **scikit-image**: Feature extraction
- **napari**: Interactive visualization (planned)

## Useful URLs

- **OMERO Documentation**: https://docs.openmicroscopy.org/
- **Cellpose**: https://cellpose.readthedocs.io/
- **DuckDB**: https://duckdb.org/docs/

---

## Notes

- Update this file when project configuration changes
- Keep entries concise (1-2 lines)
- Include URLs for easy navigation
- Manual cleanup of outdated entries is expected
