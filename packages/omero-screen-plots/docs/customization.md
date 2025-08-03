# Customization & Styling

OmeroScreen Plots uses a custom matplotlib style (`hhlab_style01.mplstyle`) that provides consistent, publication-ready formatting for all plots. This page documents the default styling and shows how to customize it.

## Default Color Palette

The package uses a carefully curated 10-color palette designed for scientific visualization:

| Color Name | Hex Code | RGB | Preview |
|------------|----------|-----|---------|
| Blue | `#526C94` | (82, 108, 148) | <span style="color:#526C94">████</span> |
| Yellow | `#D8C367` | (216, 195, 103) | <span style="color:#D8C367">████</span> |
| Light Blue | `#75B1CE` | (117, 177, 206) | <span style="color:#75B1CE">████</span> |
| Pink | `#DC6B83` | (220, 107, 131) | <span style="color:#DC6B83">████</span> |
| Grey | `#D4D3CF` | (212, 211, 207) | <span style="color:#D4D3CF">████</span> |
| Turquoise | `#00BFB2` | (0, 191, 178) | <span style="color:#00BFB2">████</span> |
| Light Green | `#CCDBA2` | (204, 219, 162) | <span style="color:#CCDBA2">████</span> |
| Olive | `#889466` | (136, 148, 102) | <span style="color:#889466">████</span> |
| Lavender | `#C6B2D1` | (198, 178, 209) | <span style="color:#C6B2D1">████</span> |
| Purple | `#654875` | (101, 72, 117) | <span style="color:#654875">████</span> |

### Using Colors in Code

Colors are available through the `COLOR` enum:

```python
from omero_screen_plots.colors import COLOR

# Access individual colors
blue_color = COLOR.BLUE.value        # "#526C94"
yellow_color = COLOR.YELLOW.value    # "#D8C367"

# Use in custom plotting
import matplotlib.pyplot as plt
plt.plot(x, y, color=COLOR.BLUE.value)
```

## Typography & Layout

### Font Settings
- **Font Family**: Arial (sans-serif)
- **Default Size**: 6pt
- **Label Size**: 6pt (axes labels)
- **Tick Label Size**: 6pt
- **Legend Size**: 6pt
- **Title Size**: 3pt (positioned left, regular weight)

### Figure Dimensions
- **Default Size**: 2.4" × 1.3" (publication-ready)
- **DPI**: 150 (display), 300 (saved files)
- **Format**: Tight layout with 0.2" padding

### Axes Styling
- **Spines**: Left and bottom only (top and right hidden)
- **Line Width**: 0.5pt
- **Color**: Black text and axes
- **Ticks**: No tick marks (cleaner appearance)
- **Grid**: Disabled by default

### Legends
- **Background**: White with full opacity
- **Frame**: Visible border
- **Spacing**: Compact (0.25 units)
- **Scatter Points**: 3 points in legend

## Customizing Colors

### Method 1: Using order_and_colors Parameter

Most plot functions accept an `order_and_colors` dictionary:

```python
from omero_screen_plots.featureplot import feature_plot_simple

# Custom color scheme
custom_colors = {
    "Control": "#1f77b4",      # Blue
    "Treatment_A": "#ff7f0e",  # Orange
    "Treatment_B": "#2ca02c",  # Green
    "Treatment_C": "#d62728"   # Red
}

feature_plot_simple(
    data_path="data.csv",
    y_feature="intensity_mean_p21_nucleus",
    conditions=["Control", "Treatment_A", "Treatment_B", "Treatment_C"],
    condition_col="condition",
    order_and_colors=custom_colors,
    output_path="output/"
)
```

### Method 2: Using Default Colors with Custom Order

```python
from omero_screen_plots.colors import COLOR

# Use default palette but specify order
ordered_colors = {
    "DMSO": COLOR.BLUE.value,
    "Nutlin": COLOR.YELLOW.value,
    "Etoptoside": COLOR.LIGHT_BLUE.value,
    "Nocodazole": COLOR.PINK.value
}
```

### Method 3: Grouped Color Schemes

For grouped analyses, colors are automatically assigned within groups:

```python
feature_plot_simple(
    data_path="data.csv",
    y_feature="intensity_mean_p21_nucleus",
    conditions=["WT_Control", "WT_Drug", "KO_Control", "KO_Drug"],
    condition_col="condition",
    grouped=True,
    group_names=["Wild Type", "Knockout"],
    output_path="output/"
)
# Automatically uses alternating colors within each group
```

## Style Customization

### Temporary Style Override

```python
import matplotlib.pyplot as plt

# Temporarily override specific parameters
with plt.rc_context({'font.size': 8, 'axes.labelsize': 8}):
    feature_plot_simple(
        data_path="data.csv",
        y_feature="intensity_mean_p21_nucleus",
        conditions=["Control", "Treatment"],
        condition_col="condition",
        output_path="output/"
    )
```

### Creating Custom Style File

Create your own `.mplstyle` file based on `hhlab_style01.mplstyle`:

```python
import matplotlib.pyplot as plt

# Load custom style
plt.style.use("path/to/your/custom_style.mplstyle")

# Then use plotting functions normally
feature_plot_simple(...)
```

### Modifying Existing Plots

For fine-tuning individual plots:

```python
from omero_screen_plots.featureplot import feature_plot
import pandas as pd

# Create base plot
df = pd.read_csv("data.csv")
fig, ax = feature_plot(
    df=df,
    y_feature="intensity_mean_p21_nucleus",
    conditions=["Control", "Treatment"],
    condition_col="condition"
)

# Customize further
ax.set_title("Custom Title", fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("custom_plot.pdf")
```

## Best Practices

### Color Selection
- **Accessibility**: Default palette is colorblind-friendly
- **Contrast**: High contrast against white backgrounds
- **Consistency**: Use same colors for same conditions across figures
- **Limit**: Maximum 10 conditions per plot for clarity

### Typography
- **Size**: Keep text readable at publication scale
- **Hierarchy**: Distinguish between labels, titles, and annotations
- **Consistency**: Use same font sizes across related figures

### Layout
- **Spacing**: Ensure adequate white space around elements
- **Alignment**: Left-align titles for consistent appearance
- **Proportions**: Maintain aspect ratios appropriate for data type

### File Output
- **Format**: PDF for publications, PNG for presentations
- **Resolution**: 300 DPI minimum for print quality
- **Size**: Standard figure dimensions for journal requirements

## Common Customizations

### Journal-Specific Requirements

```python
# Nature/Science style (larger fonts)
with plt.rc_context({
    'font.size': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7
}):
    your_plot_function()

# Cell/Molecular Biology style (smaller, compact)
with plt.rc_context({
    'font.size': 6,
    'figure.figsize': (2.0, 1.5)
}):
    your_plot_function()
```

### High-Contrast Mode

```python
# Black and white for grayscale printing
bw_colors = {
    "Control": "#000000",      # Black
    "Treatment_1": "#666666",  # Dark grey
    "Treatment_2": "#CCCCCC"   # Light grey
}
```

### Presentation Mode

```python
# Larger sizes for presentations
with plt.rc_context({
    'font.size': 12,
    'axes.labelsize': 14,
    'figure.figsize': (6, 4),
    'figure.dpi': 100
}):
    your_plot_function()
```

The default styling is optimized for scientific publications but can be easily adapted for different contexts while maintaining the professional appearance and readability that makes OmeroScreen Plots effective for data communication.
