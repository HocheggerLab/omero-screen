# Migration Guide: From Singleton to Dependency Injection

This guide explains how to migrate from the singleton pattern to dependency injection in the CellView package.

## Overview

The CellView package previously used a singleton pattern for state management through `CellViewState.get_instance()`. This has been refactored to support dependency injection for better testability, thread safety, and code maintainability.

## What Changed

### New Classes and Functions

1. **`CellViewStateCore`** - A new dependency-injectable version of state management
2. **`create_cellview_state()`** - Convenience function for creating state instances
3. **Updated importer classes** - All now accept optional state parameter
4. **New entry points** - `main_with_dependency_injection()` and `cellview_load_data_with_injection()`

### Backward Compatibility

- All existing code continues to work unchanged
- Singleton pattern is still available for legacy code
- Gradual migration is supported

## Migration Patterns

### 1. Basic State Creation

**Before (Singleton):**
```python
from cellview.utils.state import CellViewState

# Gets global singleton instance
state = CellViewState.get_instance(args)
```

**After (Dependency Injection):**
```python
from cellview.utils.state import create_cellview_state

# Creates new instance, no global state
state = create_cellview_state(args)
```

### 2. Importer Classes

**Before (Singleton):**
```python
from cellview.importers.measurements import MeasurementsManager

# Automatically uses singleton state
manager = MeasurementsManager(db_conn)
```

**After (Dependency Injection):**
```python
from cellview.importers.measurements import MeasurementsManager
from cellview.utils.state import create_cellview_state

# Inject state dependency
state = create_cellview_state(args)
manager = MeasurementsManager(db_conn, state)
```

### 3. Main Entry Points

**Before (Singleton):**
```python
from cellview.main import main

# Uses singleton pattern internally
main()
```

**After (Dependency Injection):**
```python
from cellview.main import main_with_dependency_injection

# Uses dependency injection internally
main_with_dependency_injection()
```

### 4. API Functions

**Before (Singleton):**
```python
from cellview.api import cellview_load_data

# Uses singleton pattern internally
df, vars = cellview_load_data(12345, 67890)
```

**After (Dependency Injection):**
```python
from cellview.api import cellview_load_data_with_injection

# Uses dependency injection internally
df, vars = cellview_load_data_with_injection(12345, 67890)
```

## Benefits of Migration

### 1. Thread Safety
```python
# Each thread can have its own state instance
import threading
from cellview.utils.state import create_cellview_state

def process_plate(plate_id):
    # Each thread gets isolated state
    args = argparse.Namespace(plate_id=plate_id, csv=None)
    state = create_cellview_state(args)
    # ... process with isolated state

# Safe concurrent processing
threads = []
for plate_id in [123, 456, 789]:
    t = threading.Thread(target=process_plate, args=(plate_id,))
    threads.append(t)
    t.start()
```

### 2. Better Testing
```python
def test_measurements_manager():
    # Create isolated test state
    state = create_cellview_state(None)
    state.df = create_test_dataframe()
    state.condition_id_map = {'A01': 1, 'A02': 2}

    # Inject into manager for testing
    manager = MeasurementsManager(mock_conn, state)

    # Test with controlled state
    manager.import_measurements()
    assert_expected_behavior()
```

### 3. State Isolation
```python
# Multiple independent processing pipelines
state1 = create_cellview_state(args1)  # For plate 123
state2 = create_cellview_state(args2)  # For plate 456

# No interference between states
manager1 = MeasurementsManager(conn, state1)
manager2 = MeasurementsManager(conn, state2)

# Process independently
manager1.import_measurements()  # Won't affect state2
manager2.import_measurements()  # Won't affect state1
```

## Migration Strategy

### Phase 1: Add Dependency Injection Support (✅ Complete)
- New `CellViewStateCore` class
- Updated all importer classes to accept optional state parameter
- New entry points with dependency injection
- Maintain full backward compatibility

### Phase 2: Migrate New Code (Recommended)
- Use dependency injection for all new features
- Update tests to use dependency injection
- Update documentation to prefer new patterns

### Phase 3: Gradual Legacy Migration (Optional)
- Identify critical code paths using singleton
- Migrate high-value areas (concurrent processing, complex tests)
- Keep backward compatibility for stable code

### Phase 4: Remove Singleton (Future)
- After sufficient migration time
- Remove `CellViewState` singleton implementation
- Remove legacy entry points
- **Note: This is not part of the current refactor**

## Code Examples

### Complete Example: Processing Multiple Plates Concurrently

```python
import concurrent.futures
import argparse
from cellview.utils.state import create_cellview_state
from cellview.importers.import_functions import import_data
from cellview.db.db import CellViewDB

def process_plate_with_di(plate_id: int) -> int:
    """Process a single plate using dependency injection."""
    try:
        # Create isolated state for this plate
        args = argparse.Namespace(plate_id=plate_id, csv=None)
        state = create_cellview_state(args)

        # Create database connection
        db = CellViewDB()

        # Import data with isolated state
        result = import_data(db, state)
        return result
    except Exception as e:
        print(f"Error processing plate {plate_id}: {e}")
        return 1

def main_concurrent_processing():
    """Process multiple plates concurrently using dependency injection."""
    plate_ids = [12345, 67890, 11111, 22222]

    # Process plates concurrently - each with isolated state
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(process_plate_with_di, plate_id): plate_id
            for plate_id in plate_ids
        }

        results = {}
        for future in concurrent.futures.as_completed(futures):
            plate_id = futures[future]
            try:
                result = future.result()
                results[plate_id] = result
                print(f"Plate {plate_id}: {'Success' if result == 0 else 'Failed'}")
            except Exception as e:
                print(f"Plate {plate_id} generated exception: {e}")
                results[plate_id] = 1

    return results

if __name__ == "__main__":
    results = main_concurrent_processing()
    print(f"Processing complete: {results}")
```

## Testing Migration

Use the provided tests to verify your migration:

```bash
# Run dependency injection tests
cd packages/cellview
python -m pytest tests/test_dependency_injection.py -v
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure to import from the correct modules
2. **State Not Passed**: Ensure state parameter is passed to importer classes
3. **Mixed Patterns**: Be consistent within a single code path (don't mix singleton and DI)

### Getting Help

- Review the test examples in `tests/test_dependency_injection.py`
- Check the new entry points in `main.py` and `api.py`
- Refer to the updated importer classes for parameter patterns

## Best Practices

1. **Use dependency injection for new code**
2. **Prefer `create_cellview_state()` over direct `CellViewStateCore` instantiation**
3. **Keep state instances scoped appropriately (per-request, per-thread, etc.)**
4. **Use dependency injection in tests for better isolation**
5. **Document when code uses dependency injection vs singleton**

## Conclusion

This migration provides a smooth transition path from singleton to dependency injection while maintaining full backward compatibility. The new pattern offers better testability, thread safety, and maintainability without disrupting existing code.
