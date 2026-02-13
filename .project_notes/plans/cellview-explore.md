We want an interactive data exploration tool that links CellView measurement data (scatter plots, cell cycle
 charts) with napari microscopy image viewing. The user types cellview --explore <plate_id>, a Jupyter notebook
 opens with pre-built cells that load data from CellView + images from OMERO, display them in napari, and provide
  plotly scatter plots where selecting cells highlights them in napari and navigates the camera.

 Target users are "terminal shy biologists" — the CLI command + template notebook pattern means they type one
 thing and then just run cells top-to-bottom.

 Architecture

 Create an explore/ subpackage inside cellview. The CLI generates a notebook programmatically (via nbformat),
 then launches JupyterLab. The notebook imports a bridge.py helper module containing ExploreSession — the class
 that manages data loading, napari viewer, plotly widgets, and the selection linking between them.

 Heavy dependencies (napari, plotly, ipywidgets) are only imported inside bridge.py at runtime, never at cellview
  import time. The CLI itself only needs nbformat + subprocess.

 Files to Create

 1. packages/cellview/src/cellview/explore/__init__.py

 Package marker. Exports ExploreSession for direct use.

 2. packages/cellview/src/cellview/explore/_cli.py

 - launch_explore(plate_id: int) -> None
 - Creates notebook via _notebook_builder.build_explore_notebook(plate_id)
 - Writes to ~/.cellview/explore/explore_plate_{plate_id}.ipynb
 - Launches jupyter lab <path> via subprocess.run()
 - Prints clear terminal instructions with Rich

 3. packages/cellview/src/cellview/explore/_notebook_builder.py

 - build_explore_notebook(plate_id: int) -> nbformat.NotebookNode
 - Programmatic notebook creation (no static .ipynb template to maintain)
 - Cells:
   a. Markdown: Title with plate_id
   b. Code: %gui qt, imports, session = ExploreSession(plate_id); session.connect()
   c. Markdown: Well selection instructions
   d. Code: session.show_well_selector()
   e. Markdown: Scatter plot instructions
   f. Code: session.show_scatter_plot()
   g. Markdown: Cell cycle chart instructions
   h. Code: session.show_cell_cycle_chart()
   i. Markdown: Manual exploration tips (session.df, session.viewer)

 4. packages/cellview/src/cellview/explore/bridge.py

 Core module — ExploreSession class:

 ExploreSession
 ├── __init__(plate_id)
 ├── connect()               # Load CellView data + OMERO conn + napari viewer
 ├── show_well_selector()    # ipywidgets Dropdown → _load_well()
 ├── show_scatter_plot()     # plotly FigureWidget + feature dropdowns
 ├── show_cell_cycle_chart() # plotly bar chart of cell cycle phases
 ├── _load_well(well_pos)    # Load images from OMERO, add to napari, add Points layer
 ├── _add_centroid_points()  # Map centroids to napari Points layer
 ├── _update_scatter()       # Rebuild scatter traces
 ├── _on_scatter_selection() # Lasso/box select → highlight Points, center camera
 ├── _on_scatter_click()     # Single click → navigate + zoom to cell
 └── _update_cell_cycle()    # Update bar chart for current well

 Data flow:
 - cellview_load_data(plate_id) → full DataFrame
 - Filter by well → _well_df
 - get_image_timepoint(conn, image_id, t=0) from omero_screen_napari.omero_image → numpy arrays (ZYXC), squeeze Z
  for 2D
 - Centroids from centroid-0-nuc (Y) and centroid-1-nuc (X) columns
 - Multi-image wells: stack images on axis 0, prepend image index to point coords

 Selection linking:
 - plotly FigureWidget traces have on_selection and on_click callbacks
 - Callback gets selected point indices via customdata → updates Points.face_color (green=unselected,
 red=selected)
 - Single click: also sets viewer.camera.center and viewer.camera.zoom
 - Uses Scattergl (WebGL) for performance even though datasets are small

 OMERO connection:
 - Uses BlitzGateway directly (not @omero_connect decorator, since we need a long-lived connection)
 - enableKeepAlive(60) to prevent timeout during interactive session
 - Reads credentials from env vars (same pattern as rest of codebase)

 Files to Modify

 5. packages/cellview/src/cellview/cli.py

 Add --explore argument:
 parser.add_argument(
     "--explore",
     type=int,
     metavar="PLATE_ID",
     help="Launch interactive Jupyter notebook for exploring plate data with napari.",
 )

 6. packages/cellview/src/cellview/main.py

 Add handler before DB connection setup (explore doesn't need DuckDB at startup):
 args = parse_args()

 # Handle explore before DB setup (it doesn't need the DB directly)
 if args.explore:
     from cellview.explore._cli import launch_explore
     launch_explore(args.explore)
     return

 7. packages/cellview/pyproject.toml

 Add optional dependency group:
 [project.optional-dependencies]
 test = ["pytest", "pytest-mock"]
 explore = [
     "nbformat>=5.9",
     "jupyterlab>=4.0",
     "plotly>=5.18",
     "ipywidgets>=8.0",
     "napari>=0.5",
     "omero-screen-napari",
 ]
 (These are already in the workspace via root pyproject.toml, so no install needed in practice)

 Key Reuse Points

 ┌───────────────────────────┬─────────────────────────────────┬─────────────────────────────────────────┐
 │           What            │              From               │                Used for                 │
 ├───────────────────────────┼─────────────────────────────────┼─────────────────────────────────────────┤
 │ cellview_load_data()      │ cellview.api                    │ Load plate DataFrame                    │
 ├───────────────────────────┼─────────────────────────────────┼─────────────────────────────────────────┤
 │ get_image_timepoint()     │ omero_screen_napari.omero_image │ Load images with diskcache              │
 ├───────────────────────────┼─────────────────────────────────┼─────────────────────────────────────────┤
 │ BlitzGateway              │ omero.gateway                   │ OMERO connection (same env var pattern) │
 ├───────────────────────────┼─────────────────────────────────┼─────────────────────────────────────────┤
 │ Segmentation mask loading │ omero_screen_napari.omero_image │ Load {image_id}_segmentation masks      │
 └───────────────────────────┴─────────────────────────────────┴─────────────────────────────────────────┘

 Testing

 Unit tests (tests/unit_tests/cellview_tests/explore/)

 1. test_notebook_builder.py
   - build_explore_notebook() returns valid NotebookNode
   - Correct number of cells, plate_id injected
   - nbformat.validate() passes
 2. test_cli.py
   - --explore argument parsed correctly
   - launch_explore() creates notebook file (mock subprocess)
 3. test_bridge.py
   - ExploreSession.__init__ stores plate_id
   - _add_centroid_points builds correct coordinate arrays from mock DataFrame
   - _on_scatter_selection updates face_color array correctly
   - _on_scatter_click sets camera center from point coordinates
   - Mock napari viewer + mock Points layer (no Qt needed)

 Manual verification

 1. cellview --explore <plate_id> → JupyterLab opens with notebook
 2. Run cells top-to-bottom → napari window appears with well images
 3. Select cells in scatter → napari highlights and navigates
 4. Change well → all views update

 Implementation Order

 1. explore/__init__.py — package marker
 2. explore/_notebook_builder.py — no heavy deps, testable in isolation
 3. explore/_cli.py — depends on builder, testable with mocked subprocess
 4. cli.py + main.py — wire up the --explore flag
 5. explore/bridge.py — the core linking logic (most complex)
 6. pyproject.toml — add optional deps
 7. Tests
