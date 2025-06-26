# omero-screen-plots Test Suite

This directory contains unit tests for the omero-screen-plots package.

## Test Structure

- `conftest.py` - Test fixtures and shared test data
- `test_cellcycleplot.py` - Tests for legacy cellcycle_plot function
- `test_combplot.py` - Tests for legacy combplot functions
- `test_countplot.py` - Tests for count_plot function
- `test_featurplot.py` - Tests for feature_plot function
- `test_new_plots.py` - Tests for new plotting functions (cellcycle_standard_plot, etc.)
- `test_plot_classes.py` - Tests for plot class implementations

## Testing Philosophy

As this is a research project, the tests focus on high-level smoke testing:
- Verify that plotting functions execute without errors
- Test with realistic data from `examples/plots_test_data.csv`
- Check that plots handle edge cases gracefully
- No detailed visual output validation (matplotlib makes this difficult)

## Key Test Fixtures

- `cell_cycle_data` - Full dataset loaded from CSV
- `filtered_data` - Pre-filtered data with normalized columns added

## Running Tests

```bash
# Run all omero-screen-plots tests
pytest tests/unit_tests/omero_screen_plots_tests/

# Run specific test file
pytest tests/unit_tests/omero_screen_plots_tests/test_new_plots.py

# Run with verbose output
pytest tests/unit_tests/omero_screen_plots_tests/ -v
```

## Test Coverage

The tests cover:
- All new plotting functions (standard, stacked, grouped cell cycle plots)
- All combined plot types (histogram, scatter, simple, full)
- Plot class implementations
- Error handling for invalid inputs
- Edge cases (minimal data, missing columns, single conditions)
