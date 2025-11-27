Data Structure and Hierarchy
============================

Cellview organizes high-content screening data using a hierarchical structure that maps to OMERO's data model but is optimized for analysis.

Hierarchy
---------

The data is organized in the following hierarchy:

1.  **Project**: The top-level container. A project can contain multiple experiments.
2.  **Experiment**: A logical grouping of plates that belong to the same experimental series.
3.  **Plate** (Repeat): A physical plate (e.g., 96-well plate). In the database, these are sometimes referred to as "repeats".
4.  **Well** (Condition): A specific well on a plate. This is where experimental conditions (cell lines, drugs, etc.) are defined.
5.  **Measurement**: The actual data points (e.g., cell intensity, size) derived from image analysis.

Inferring Hierarchy from OMERO
------------------------------

When importing data from OMERO, Cellview can automatically infer the Project and Experiment names from **Tags** attached to the Plate in OMERO.

-   **Project**: If a tag with the format `Project: <Name>` is found, `<Name>` is used as the project name.
-   **Experiment**: If a tag with the format `Experiment: <Name>` is found, `<Name>` is used as the experiment name.

If these tags are not present, you can specify the Project and Experiment names manually using the CLI arguments during import.

Database Structure
------------------

Internally, Cellview uses a DuckDB database with the following key tables:

-   **projects**: Stores project metadata.
-   **experiments**: Links experiments to projects.
-   **repeats**: Stores plate information, linked to experiments.
-   **conditions**: Stores well-level information (cell line, antibodies, etc.), linked to plates (repeats).
-   **measurements**: Stores single-cell data, linked to conditions.
-   **condition_variables**: Stores flexible key-value pairs for experimental conditions (e.g., "Drug_A": "10uM").
