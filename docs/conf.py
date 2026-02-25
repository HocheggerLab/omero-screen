import os
import sys
import urllib.request
from pathlib import Path
from typing import Optional
import pandas as pd

# Add the package source to the Python path
sys.path.insert(0, os.path.abspath('../src'))
sys.path.insert(0, os.path.abspath('../packages/omero-screen-plots/src'))
sys.path.insert(0, os.path.abspath('../packages/cellview/src'))
sys.path.insert(0, os.path.abspath('../packages/omero-screen-napari/src'))
# Add extensions directory from omero-screen-plots
sys.path.insert(0, os.path.abspath('../packages/omero-screen-plots/docs'))

# Documentation build configuration
BUILD_PLOTS = os.environ.get('BUILD_PLOTS', 'true').lower() == 'true'
CACHE_DIR = Path(__file__).parent / '_build' / 'cache'

def ensure_cache_dir() -> None:
    """Ensure cache directory exists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

def download_example_data() -> Optional[pd.DataFrame]:
    """Download and cache the example dataset from Zenodo."""
    ensure_cache_dir()
    cache_file = CACHE_DIR / 'sample_plate_data.csv'

    if cache_file.exists():
        print(f"Using cached data: {cache_file}")
        return pd.read_csv(cache_file)

    print("Downloading example data from Zenodo...")
    url = "https://zenodo.org/records/16636600/files/sample_plate_data.csv?download=1"

    try:
        urllib.request.urlretrieve(url, cache_file)
        print(f"Data cached to: {cache_file}")
        return pd.read_csv(cache_file)
    except Exception as e:
        print(f"Warning: Failed to download data: {e}")
        return None

def get_example_data(subset: Optional[str] = None, n_samples: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Get example data with optional subsetting."""
    df = download_example_data()
    if df is None:
        return None

    # Apply subsetting based on plot type
    if subset == 'feature':
        n_samples = n_samples or 1000
        df = df.groupby('condition').apply(
            lambda x: x.sample(n=min(len(x), n_samples), random_state=42)
        ).reset_index(drop=True)
    elif subset == 'cellcycle':
        n_samples = n_samples or 2000
        df = df.groupby('condition').apply(
            lambda x: x.sample(n=min(len(x), n_samples), random_state=43)
        ).reset_index(drop=True)
    elif subset == 'scatter':
        n_samples = n_samples or 3000
        df = df.groupby('condition').apply(
            lambda x: x.sample(n=min(len(x), n_samples), random_state=44)
        ).reset_index(drop=True)
    elif n_samples:
        df = df.groupby('condition').apply(
            lambda x: x.sample(n=min(len(x), n_samples), random_state=45)
        ).reset_index(drop=True)

    return df

# -- Project information -----------------------------------------------------

project = 'omero-screen'
copyright = '2024, HocheggerLab'
author = 'HocheggerLab'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinxarg.ext',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'myst_parser',
    'matplotlib.sphinxext.plot_directive',
    '_ext.generate_plots',  # Auto-generate example plots
    'sphinxcontrib.video',
]

# Add support for both RST and Markdown
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'display_version': True,
}

# -- Intersphinx configuration ----------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'seaborn': ('https://seaborn.pydata.org/', None),
}

# -- Plot directive configuration --------------------------------------------
plot_include_source = True
plot_html_show_source_link = False
plot_formats = ['svg', 'png']
plot_html_show_formats = False
plot_default_fmt = 'svg'
plot_rcparams = {
    'figure.dpi': 300,
    'savefig.dpi': 300
}
plot_skip_if = f'not {BUILD_PLOTS}'
plot_close_figures = True

# -- Autodoc configuration ---------------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
autodoc_mock_imports = ["omero_screen"]
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
typehints_use_signature = True
typehints_use_signature_return = True
