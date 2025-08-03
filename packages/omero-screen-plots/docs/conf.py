# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import urllib.request
from pathlib import Path
from typing import Optional
import pandas as pd

# Add the package source to the Python path
sys.path.insert(0, os.path.abspath('../../src'))
# Add extensions directory
sys.path.insert(0, os.path.abspath('.'))

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
    """Get example data with optional subsetting.

    Args:
        subset (str): Type of subset ('feature', 'cellcycle', 'scatter', 'all')
        n_samples (int): Number of samples per condition

    Returns:
        pd.DataFrame: Example data
    """
    df = download_example_data()
    if df is None:
        return None

    # Apply subsetting based on plot type
    if subset == 'feature':
        # For feature plots, sample fewer cells for clarity
        n_samples = n_samples or 1000
        df = df.groupby('condition').apply(
            lambda x: x.sample(n=min(len(x), n_samples), random_state=42)
        ).reset_index(drop=True)
    elif subset == 'cellcycle':
        # Cell cycle plots need more data for phase distribution
        n_samples = n_samples or 2000
        df = df.groupby('condition').apply(
            lambda x: x.sample(n=min(len(x), n_samples), random_state=43)
        ).reset_index(drop=True)
    elif subset == 'scatter':
        # Scatter plots benefit from more points for pattern visibility
        n_samples = n_samples or 3000
        df = df.groupby('condition').apply(
            lambda x: x.sample(n=min(len(x), n_samples), random_state=44)
        ).reset_index(drop=True)
    elif n_samples:
        # General sampling
        df = df.groupby('condition').apply(
            lambda x: x.sample(n=min(len(x), n_samples), random_state=45)
        ).reset_index(drop=True)

    return df

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'OmeroScreen Plots'
copyright = '2025, Helfrid Hochegger'
author = 'Helfrid Hochegger'
release = '0.1.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'myst_parser',
    'matplotlib.sphinxext.plot_directive',
    '_ext.generate_plots',  # Auto-generate example plots
]

# Add support for both RST and Markdown
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'en'

# -- Autodoc configuration ---------------------------------------------------

# Automatically extract docstrings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Napoleon settings for Google-style docstrings
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

# Type hints configuration
typehints_use_signature = True
typehints_use_signature_return = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Theme options
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'display_version': True,
}

# Add logo and favicon if they exist
# html_logo = '_static/logo.png'
# html_favicon = '_static/favicon.ico'

html_context = {
    'display_github': True,
    'github_user': 'Helfrid',
    'github_repo': 'omero-screen',
    'github_version': 'main',
    'conf_py_path': '/packages/omero-screen-plots/docs/',
}

# -- Intersphinx configuration ----------------------------------------------
# Link to other projects' documentation
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'seaborn': ('https://seaborn.pydata.org/', None),
}

# -- Plot directive configuration --------------------------------------------
# Configuration for matplotlib plot directive
plot_include_source = True
plot_html_show_source_link = False
plot_formats = ['svg', 'png']  # Prefer SVG for quality
plot_html_show_formats = False
plot_default_fmt = 'svg'  # Use SVG by default
# Don't override the plot styles - let the package handle it
# But increase DPI for sharper documentation images
plot_rcparams = {
    'figure.dpi': 300,  # Higher resolution for docs (default is ~100)
    'savefig.dpi': 300  # Match save DPI
}

# Skip plots if BUILD_PLOTS is False or data download fails
plot_skip_if = f'not {BUILD_PLOTS}'
plot_close_figures = True
