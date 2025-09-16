# Environment Setup for Jupyter Examples

When running the omero-screen-plots examples in Jupyter notebooks, you may need to configure environment variables for OMERO connection.

## Problem

The plotting library automatically looks for `.env.development` files to load OMERO credentials. However, when installed as a package, it may not find your project's environment files.

## Solutions

### Option 1: Set Project Root (Recommended)

Set an environment variable pointing to your omero-screen project:

```python
import os
os.environ['OMERO_SCREEN_PROJECT_ROOT'] = '/path/to/your/omero-screen'

# Then import normally
import omero_screen_plots
```

### Option 2: Run from Project Directory

Start Jupyter from within the omero-screen project directory:

```bash
cd /path/to/your/omero-screen
jupyter lab packages/omero-screen-plots/examples/
```

### Option 3: Set Environment Variables Directly

```python
import os

# OMERO connection settings
os.environ['USERNAME'] = 'your_omero_username'
os.environ['PASSWORD'] = 'your_omero_password'
os.environ['HOST'] = 'your_omero_host'

# Logging settings (optional)
os.environ['LOG_LEVEL'] = 'INFO'
os.environ['ENABLE_CONSOLE_LOGGING'] = 'true'
os.environ['ENABLE_FILE_LOGGING'] = 'false'

# Then import
import omero_screen_plots
```

### Option 4: Copy Environment File

Copy your `.env.development` file to the current working directory:

```bash
cp /path/to/omero-screen/.env.development .
```

## Environment Variables

Required variables for OMERO connection:
- `USERNAME`: Your OMERO username
- `PASSWORD`: Your OMERO password
- `HOST`: OMERO server hostname
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `ENABLE_CONSOLE_LOGGING`: Enable console logging (true/false)
- `ENABLE_FILE_LOGGING`: Enable file logging (true/false)

## Troubleshooting

If you see an error like "No configuration found!", the system will show you:
- Where it looked for environment files
- What project root it detected
- Suggested solutions

The new robust discovery system tries multiple strategies:
1. `OMERO_SCREEN_PROJECT_ROOT` environment variable
2. Git repository root (looks for `.git` folder)
3. Project markers (`pyproject.toml`, `uv.lock`, `CLAUDE.md`, `packages`)
4. File-based discovery (development mode)
5. Current working directory (fallback)
