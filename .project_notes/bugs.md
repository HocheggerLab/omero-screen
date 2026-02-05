# Bug Log

Track recurring bugs, their solutions, and prevention strategies. This helps avoid solving the same problem twice.

## Format

```markdown
### YYYY-MM-DD - Brief Bug Description
- **Issue**: What went wrong
- **Root Cause**: Why it happened
- **Solution**: How it was fixed
- **Prevention**: How to avoid it in the future
```

## Examples

### 2025-01-15 - OMERO Connection Timeout in E2E Tests
- **Issue**: E2E tests fail intermittently with "Connection refused" errors to test OMERO server
- **Root Cause**: Test OMERO server on 127.0.0.2:4064 not started before tests run
- **Solution**: Add pre-test check to ensure OMERO server is running, or start it automatically in test setup
- **Prevention**: Document test server requirements in test README, add startup script

### 2025-01-10 - Cellpose Model Selection Fails for New Cell Lines
- **Issue**: Segmentation fails with KeyError when processing new cell lines not in config
- **Root Cause**: Model selection logic expects all cell lines to be pre-configured
- **Solution**: Add fallback to default Cellpose model (cyto2) when cell line not found
- **Prevention**: Add validation for cell line metadata, warn on unknown cell lines

---

## Active Bugs

(Add new bug entries below this line)

### 2026-02-05 - Database Schema Pollution from Suffixed Columns

- **Issue**: Database imports failed after importing a problematic plate. Error occurred when trying to import new plates via welldata_widget, with errors relating to column mismatches. The measurements table had accumulated 72 duplicate columns with numeric suffixes (`.0`, `.1`, `.2`) that caused INSERT statements to fail because new plates only provided values for clean column names.

- **Root Cause**:
  - Aggregated data files (`agg_data.csv`) sometimes contain duplicate columns from pandas merge operations, which pandas auto-renames with `.0`, `.1` suffixes
  - The `_ensure_intensity_columns_exist()` function in `cellview/importers/measurements.py` dynamically adds missing columns using `ALTER TABLE ADD COLUMN`
  - Once added, these suffixed columns became permanent in the database schema
  - The `clean_up_db()` function only removed orphaned **records** (rows with missing foreign keys), not orphaned **schema columns**
  - Future imports failed because the INSERT statement expected values for ALL table columns, including the spurious suffixed ones

- **Solution**:
  1. **Immediate fix**: Manually dropped all 72 problematic columns using `ALTER TABLE measurements DROP COLUMN "column.0"`
  2. **Permanent fix**: Added `clean_schema_columns()` function to `cellview/db/clean_up.py` that:
     - Detects columns with numeric suffixes using regex pattern `\.\d+$`
     - Automatically drops problematic columns from the measurements table
     - Integrated into both `clean_up_db()` and `deep_clean_db()` functions
     - Runs as first step before record cleanup
  3. Function is now called automatically when:
     - User runs import with `--clean` flag
     - User deletes a plate (cleanup runs automatically)
     - User runs deep cleanup after errors

- **Prevention**:
  - Schema cleaning is now part of the standard cleanup workflow
  - Consider adding pre-import validation in `_clean_agg_data()` (cellview/utils/state.py:365-464) to strip suffixed columns BEFORE they reach the import stage
  - Monitor for recurrence by checking measurements table schema periodically
  - Affected columns included: DAPI, pRb/pRB, gH2AX/γH2AX, CyclinA, Cdk4, EdU (all with various `.0`, `.1`, `.2` suffixes)
  - The `_clean_agg_data()` function already has logic to handle suffixed columns (lines 408-431) but may need strengthening to catch all edge cases
