# Installation

## Requirements

- Python 3.12 or higher
- OMERO Screen installation (for full pipeline integration)
- Access to screening data in CSV format or cellview database

## Install from Source

Since OmeroScreen Plots is part of the omero-screen monorepo, the recommended installation method is through the main project:

```bash
# Clone the repository
git clone https://github.com/Helfrid/omero-screen.git
cd omero-screen

# Install using uv (recommended)
uv sync --dev
source .venv/bin/activate
```

## Standalone Installation

If you only need the plotting functionality:

```bash
cd packages/omero-screen-plots
uv pip install -e .
```

## Dependencies

OmeroScreen Plots requires the following main dependencies:

- **pandas**: Data manipulation and analysis
- **matplotlib**: Core plotting functionality
- **seaborn**: Statistical visualizations
- **numpy**: Numerical operations
- **scipy**: Statistical tests
- **scikit-posthocs**: Post-hoc statistical tests

All dependencies are automatically installed with the package.

## Verify Installation

```python
import omero_screen_plots
print(omero_screen_plots.__version__)
```

## Troubleshooting

### Import Errors

If you encounter import errors, ensure that:
1. You're in the correct virtual environment
2. The package is installed in editable mode (`-e`)
3. Your Python path includes the package directory

### Missing Dependencies

Run `uv sync` to ensure all dependencies are installed correctly.

## Next Steps

- Check out the [Quick Start Guide](quickstart.md) for basic usage
- Explore [Plot Types](plot_types.md) for available visualizations
- See [Examples](examples/basic_plots.md) for real-world use cases
