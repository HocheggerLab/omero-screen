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
