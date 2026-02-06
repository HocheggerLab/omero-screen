Configuration
=============

The project uses a flexible environment variable system that supports both development and production environments.

Environment Files
-----------------

*   The system first checks for an ``ENV`` environment variable.
*   If ``ENV`` is set (e.g., to ``production``), it loads configuration from ``.env.production``.
*   If ``ENV`` is not set, it defaults to ``development`` and loads ``.env.development``.
*   If no environment-specific file exists, it falls back to a default ``.env`` file.

Required Variables
------------------

Create a ``.env`` file in the root directory with the following variables:

**OMERO Server Configuration**

.. code-block:: bash

    USERNAME="omero-login-name"
    PASSWORD="omero-password"
    HOST="omero-server-host"
    PROJECT_ID=5313  # Project ID for "Screens" project
    DATA_PATH='omero-napari-data'

**Logging Configuration**

.. code-block:: bash

    LOG_LEVEL=DEBUG
    LOG_FILE_PATH=logs/app.log
    LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s
    ENABLE_CONSOLE_LOGGING=False
    ENABLE_FILE_LOGGING=True
    LOG_MAX_BYTES=1048576        # 1MB
    LOG_BACKUP_COUNT=5

**CellView Database Configuration**

.. code-block:: bash

    TEST_DATABASE=false
    DATABASE_PATH=~/cellview_date/cellview.db

**Image Cache Configuration (optional)**

.. code-block:: bash

    OMERO_SCREEN_IMAGE_CACHE_PATH=/path/to/home/.cache/omero_screen/images
    OMERO_SCREEN_IMAGE_CACHE_SIZE_LIMIT=4294967296
