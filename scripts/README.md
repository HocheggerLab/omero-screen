# Scripts

Utility scripts for the omero-screen project.

## download_sample_data.py

Downloads sample plate data from Zenodo for use in examples and testing.

### Activation

chmod +x ./scripts/download_sample_data.py
### Usage

```bash
# Download sample data (skips if already exists)
python scripts/download_sample_data.py

# Force re-download even if file exists
python scripts/download_sample_data.py --force

# Show help
python scripts/download_sample_data.py --help
```

### What it does

- Downloads `sample_plate_data.csv` from Zenodo (https://zenodo.org/records/16636600)
- Places the file in `packages/omero-screen-plots/examples/data/`
- Creates the data directory if it doesn't exist
- Shows download progress and file size information
- Skips download if file already exists (unless `--force` is used)

### File Details

- **Source**: Zenodo record 16636600
- **Size**: ~91.5 MB
- **Rows**: ~466k measurement records
- **Format**: CSV with single-cell measurements from OMERO Screen analysis

This sample data can be used with the plotting examples in `packages/omero-screen-plots/examples/` to demonstrate the visualization capabilities of the omero-screen-plots package.
