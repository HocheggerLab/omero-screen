Developer Guide
===============

Development Setup
-----------------

This project uses `pre-commit <https://pre-commit.com/>`_ to validate changes.

.. code-block:: bash

    pre-commit install

Testing
-------

The project includes both unit tests and end-to-end (e2e) tests.

**Unit Tests**

Located in ``tests/unit_tests``.

.. code-block:: bash

    # Run all unit tests
    pytest -v

    # Run specific modules
    pytest tests/unit_tests/omero_screen_tests
    pytest tests/unit_tests/omero_utils_tests

**End-to-End Tests**

Located in ``tests/e2e_tests``. These simulate production-like conditions using a test server.

.. code-block:: bash

    # Run a specific e2e test
    omero-integration-test e2e_excel

Test Server
-----------

For local development, you can run a separate OMERO test server (requires Docker).

.. code-block:: bash

    # Start the test server
    ./scripts/manage_test_server.sh start

    # Check status
    ./scripts/manage_test_server.sh status

    # Stop server
    ./scripts/manage_test_server.sh stop

The test server runs on ``127.0.0.2:4064`` (User: ``root``, Pass: ``omero``).

Loading Test Data
-----------------

Use the ``load_plates.sh`` script to import data into the OMERO server.

.. code-block:: bash

    ./scripts/load_plates.sh -d /path/to/plates -x
