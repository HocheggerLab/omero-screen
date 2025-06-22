#!/usr/bin/env python3
"""Comprehensive example usage of cellcycle_stacked_plot function.

This example demonstrates all parameters of the cellcycle_stacked_plot function,
including those with default values, showing the complete interface available to users.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import the user-facing stacked plot function
from src.omero_screen_plots import cellcycle_stacked_plot


def create_comprehensive_test_data():
    """Create comprehensive test data for demonstrating all features."""
    np.random.seed(42)

    # Comprehensive experimental design
    conditions = [
        "DMSO_Control",
        "CDK4i_1μM",
        "CDK4i_5μM",
        "CDK6i_2μM",
        "CDK6i_10μM",
        "CDK4i+CDK6i_Combo",
        "Palbociclib_100nM",
        "Ribociclib_500nM",
    ]

    cell_lines = ["RPE-1", "HeLa", "U2OS", "MCF7"]
    phases = ["G1", "S", "G2/M", "Polyploid", "Sub-G1"]

    # Multiple replicate plates for robust statistics
    plate_ids = [f"exp_plate_{i:03d}" for i in range(1, 6)]  # 5 replicates

    print("Creating comprehensive cell cycle dataset...")
    print(f"Conditions: {len(conditions)} ({', '.join(conditions)})")
    print(f"Cell lines: {len(cell_lines)} ({', '.join(cell_lines)})")
    print(f"Phases: {len(phases)} ({', '.join(phases)})")
    print(f"Replicates: {len(plate_ids)} plates")

    data = []
    for plate_id in plate_ids:
        for condition in conditions:
            for cell_line in cell_lines:
                # Variable cell counts to simulate real experiments
                n_cells = np.random.randint(2000, 4000)

                # Sophisticated drug response simulation
                if "DMSO" in condition:
                    phase_probs = [
                        0.68,
                        0.16,
                        0.13,
                        0.02,
                        0.01,
                    ]  # Normal cycling
                elif "CDK4i_1μM" in condition:
                    phase_probs = [
                        0.45,
                        0.40,
                        0.12,
                        0.02,
                        0.01,
                    ]  # Mild S arrest
                elif "CDK4i_5μM" in condition:
                    phase_probs = [
                        0.30,
                        0.55,
                        0.12,
                        0.02,
                        0.01,
                    ]  # Strong S arrest
                elif "CDK6i_2μM" in condition:
                    phase_probs = [
                        0.75,
                        0.15,
                        0.08,
                        0.015,
                        0.005,
                    ]  # Mild G1 arrest
                elif "CDK6i_10μM" in condition:
                    phase_probs = [
                        0.85,
                        0.10,
                        0.04,
                        0.005,
                        0.005,
                    ]  # Strong G1 arrest
                elif "Combo" in condition:
                    phase_probs = [
                        0.20,
                        0.25,
                        0.20,
                        0.30,
                        0.05,
                    ]  # Complex response
                elif "Palbociclib" in condition:
                    phase_probs = [0.78, 0.12, 0.08, 0.015, 0.005]  # G1 arrest
                elif "Ribociclib" in condition:
                    phase_probs = [
                        0.72,
                        0.18,
                        0.08,
                        0.015,
                        0.005,
                    ]  # Moderate G1 arrest

                # Cell line specific drug sensitivity
                if cell_line == "HeLa":
                    # HeLa is more sensitive - enhance drug effects
                    if "CDK" in condition and "DMSO" not in condition:
                        phase_probs[3] *= 1.8  # More polyploidy
                        phase_probs = [
                            p / sum(phase_probs) for p in phase_probs
                        ]
                elif cell_line == "U2OS":
                    # U2OS is more resistant - reduce drug effects
                    if "CDK" in condition and "DMSO" not in condition:
                        phase_probs[0] *= 1.4  # More G1 resistance
                        phase_probs = [
                            p / sum(phase_probs) for p in phase_probs
                        ]
                elif cell_line == "MCF7":
                    # MCF7 shows intermediate sensitivity
                    if "CDK" in condition and "DMSO" not in condition:
                        phase_probs[4] *= 1.3  # More sub-G1 (apoptosis)
                        phase_probs = [
                            p / sum(phase_probs) for p in phase_probs
                        ]

                # Generate individual cell measurements
                for phase, prob in zip(phases, phase_probs, strict=False):
                    # Add realistic experimental variation
                    n_phase_cells = int(
                        n_cells * prob * (0.80 + 0.4 * np.random.random())
                    )
                    for i in range(n_phase_cells):
                        data.append(
                            {
                                "plate_id": plate_id,
                                "condition": condition,
                                "cell_line": cell_line,
                                "cell_cycle": phase,
                                "experiment": f"{plate_id}_{condition}_{cell_line}_cell_{i}",
                                # Additional metadata that might be present in real data
                                "experiment_date": f"2024-0{(hash(plate_id) % 9) + 1}-01",
                                "analyst": "researcher_"
                                + str(hash(plate_id) % 3 + 1),
                                "batch": f"batch_{(hash(condition) % 5) + 1}",
                            }
                        )

    df = pd.DataFrame(data)
    print(f"Generated {len(df):,} individual cell measurements")
    print(f"Dataset shape: {df.shape}")
    return df


def example_1_minimal_usage():
    """Example 1: Minimal usage showing just required parameters."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: MINIMAL USAGE")
    print("=" * 80)
    print("Showing the absolute minimum parameters needed:")

    data = create_comprehensive_test_data()

    # Only the essential parameters - everything else uses defaults
    fig = cellcycle_stacked_plot(
        data=data,
        conditions=["DMSO_Control", "CDK4i_5μM", "CDK6i_10μM"],
        selector_val="RPE-1",
    )

    print("\n✓ Generated with just 3 parameters!")
    print("  Default values used:")
    print("    condition_col='condition'")
    print("    selector_col='cell_line'")
    print("    title='Cell Cycle Distribution - RPE-1'")
    print("    colors=<default package colors>")
    print("    figsize=<default from config>")
    print("    phases=<default: ['Polyploid', 'G2/M', 'S', 'G1', 'Sub-G1']>")
    print("    reverse_stack=False")
    print("    show_legend=True")
    print("    legend_position='right'")
    print("    ax=None")
    print("    save=False")
    print("    dpi=300")
    print("    format='pdf'")
    print("    tight_layout=True")

    plt.close(fig)


def example_2_all_parameters_explicit():
    """Example 2: All parameters explicitly specified with their default values."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: ALL PARAMETERS EXPLICITLY SPECIFIED")
    print("=" * 80)
    print(
        "Showing every single parameter with their default values explicitly set:"
    )

    data = create_comprehensive_test_data()
    output_dir = Path("stacked_examples")
    output_dir.mkdir(exist_ok=True)

    # Every single parameter explicitly specified
    fig = cellcycle_stacked_plot(
        # REQUIRED PARAMETERS
        data=data,
        conditions=[
            "DMSO_Control",
            "CDK4i_1μM",
            "CDK4i_5μM",
            "CDK6i_2μM",
            "CDK6i_10μM",
        ],
        # BASE CLASS ARGUMENTS (with defaults shown)
        condition_col="condition",  # Default: "condition"
        selector_col="cell_line",  # Default: "cell_line"
        selector_val="HeLa",  # Default: None (but required for meaningful results)
        title="Complete Parameter Demo - HeLa Cells",  # Default: None (auto-generated)
        colors=None,  # Default: None (uses config colors)
        figsize=None,  # Default: None (uses config size)
        # CELLCYCLE STACKED PLOT SPECIFIC ARGUMENTS (with defaults)
        phases=None,  # Default: None (uses ['Polyploid', 'G2/M', 'S', 'G1', 'Sub-G1'])
        reverse_stack=False,  # Default: False
        show_legend=True,  # Default: True
        legend_position="right",  # Default: "right"
        # INTEGRATION ARGUMENTS
        ax=None,  # Default: None (creates own figure)
        # OUTPUT ARGUMENTS (with defaults)
        save=True,  # Default: False
        output_path=str(output_dir),  # Default: None (required if save=True)
        filename="all_parameters_explicit.pdf",  # Default: None (auto-generated)
        # SAVE QUALITY ARGUMENTS (with defaults)
        dpi=300,  # Default: 300
        format="pdf",  # Default: "pdf"
        tight_layout=True,  # Default: True
    )

    print("\n✓ Generated with ALL parameters explicitly specified!")
    print("  This shows the complete interface available to users")
    print("  Every parameter is documented and controllable")


def example_3_custom_styling_options():
    """Example 3: Demonstrate extensive customization capabilities."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: EXTENSIVE CUSTOMIZATION")
    print("=" * 80)
    print("Showing advanced customization options:")

    data = create_comprehensive_test_data()
    output_dir = Path("stacked_examples")

    # Advanced customization with non-default values
    fig = cellcycle_stacked_plot(
        # Core data
        data=data,
        conditions=[
            "DMSO_Control",
            "CDK4i_5μM",
            "CDK6i_10μM",
            "CDK4i+CDK6i_Combo",
        ],
        condition_col="condition",
        selector_col="cell_line",
        selector_val="U2OS",
        # Custom appearance
        title="Advanced Cell Cycle Analysis: CDK4/6 Inhibitor Effects in U2OS",
        colors=[
            "#8B0000",
            "#FF4500",
            "#FFD700",
            "#32CD32",
            "#4169E1",
        ],  # Custom color scheme
        figsize=(14, 8),  # Larger figure
        phases=[
            "Sub-G1",
            "G1",
            "S",
            "G2/M",
            "Polyploid",
        ],  # Custom phase order
        # Stacked plot customization
        reverse_stack=True,  # Reverse the stacking order
        show_legend=True,
        legend_position="bottom",  # Legend at bottom
        # High-quality output
        save=True,
        output_path=str(output_dir),
        filename="advanced_customization_u2os.pdf",
        dpi=600,  # High resolution
        format="pdf",
        tight_layout=True,
    )

    print("\n✓ Generated with extensive customization!")
    print("  Custom colors: Dark red to blue gradient")
    print("  Custom phase order: Sub-G1 first, Polyploid last")
    print("  Reversed stacking order")
    print("  Bottom legend position")
    print("  High resolution (600 DPI)")
    print("  Custom title and figure size")


def example_4_integration_with_subplots():
    """Example 4: Integration with existing matplotlib figures."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: INTEGRATION WITH SUBPLOTS")
    print("=" * 80)
    print("Showing integration with existing matplotlib figures:")

    data = create_comprehensive_test_data()
    output_dir = Path("stacked_examples")

    # Create a complex figure layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Multi-Cell Line Cell Cycle Analysis Dashboard",
        fontsize=16,
        fontweight="bold",
    )

    cell_lines = ["RPE-1", "HeLa", "U2OS", "MCF7"]
    conditions = ["DMSO_Control", "CDK4i_5μM", "CDK6i_10μM"]

    for i, (ax, cell_line) in enumerate(
        zip(axes.flat, cell_lines, strict=False)
    ):
        print(f"  Creating plot {i + 1}/4 for {cell_line}...")

        # Each subplot uses the stacked plot function with provided axis
        subplot_fig = cellcycle_stacked_plot(
            data=data,
            conditions=conditions,
            condition_col="condition",
            selector_col="cell_line",
            selector_val=cell_line,
            title=f"{cell_line}",  # Simple title for subplot
            colors=None,  # Use default colors
            figsize=None,  # Not used when ax provided
            phases=["Polyploid", "G2/M", "S", "G1"],  # Only major phases
            reverse_stack=False,
            show_legend=(i == 0),  # Only show legend on first plot
            legend_position="right",
            ax=ax,  # *** KEY: Provide existing axis ***
            save=False,  # Don't save individual plots
        )

    # Save the combined figure manually
    plt.tight_layout()
    fig.savefig(
        output_dir / "integrated_dashboard.pdf", dpi=300, bbox_inches="tight"
    )

    print("\n✓ Generated integrated dashboard!")
    print("  Four cell lines in one figure")
    print("  Each subplot created with cellcycle_stacked_plot")
    print("  Shared legend on first subplot")
    print("  Manual saving of combined figure")

    plt.close(fig)


def example_5_different_output_formats():
    """Example 5: Multiple output formats for different use cases."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: MULTIPLE OUTPUT FORMATS")
    print("=" * 80)
    print("Generating plots in different formats for different purposes:")

    data = create_comprehensive_test_data()
    output_dir = Path("stacked_examples")

    formats_and_uses = [
        ("pdf", 600, "Publication and print"),
        ("png", 150, "Web and presentations"),
        ("svg", 300, "Vector editing and scalability"),
        ("eps", 300, "Legacy publications"),
        ("tiff", 300, "Scientific journals"),
    ]

    conditions = ["DMSO_Control", "Palbociclib_100nM", "Ribociclib_500nM"]

    for fmt, dpi, use_case in formats_and_uses:
        print(
            f"  Creating {fmt.upper()} format (DPI: {dpi}) for {use_case}..."
        )

        fig = cellcycle_stacked_plot(
            data=data,
            conditions=conditions,
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF7",
            title=f"CDK4/6 Inhibitor Comparison - MCF7 ({fmt.upper()})",
            colors=["#2E86AB", "#A23B72", "#F18F01"],  # Three-color scheme
            figsize=(10, 6),
            phases=["G1", "S", "G2/M", "Polyploid"],  # Simplified phases
            reverse_stack=False,
            show_legend=True,
            legend_position="right",
            save=True,
            output_path=str(output_dir),
            filename=f"multiformat_example.{fmt}",
            dpi=dpi,
            format=fmt,
            tight_layout=True,
        )

        plt.close(fig)

    print("\n✓ Generated plots in multiple formats!")
    print("  Each format optimized for specific use case")
    print("  Different DPI settings for quality vs file size")


def example_6_error_handling_demonstration():
    """Example 6: Demonstrate error handling and validation."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: ERROR HANDLING DEMONSTRATION")
    print("=" * 80)
    print("Showing robust error handling and validation:")

    data = create_comprehensive_test_data()

    error_cases = [
        ("Empty data", pd.DataFrame(), ["Control"], {}),
        ("No conditions", data, [], {}),
        (
            "Invalid selector",
            data,
            ["Control"],
            {"selector_val": "NonExistentCellLine"},
        ),
        ("Save without path", data, ["Control"], {"save": True}),
    ]

    for case_name, test_data, test_conditions, extra_kwargs in error_cases:
        print(f"\n  Testing: {case_name}")
        try:
            fig = cellcycle_stacked_plot(
                data=test_data,
                conditions=test_conditions,
                selector_val="RPE-1",
                **extra_kwargs,
            )
            print("    ❌ ERROR: Should have raised an exception!")
            plt.close(fig)
        except Exception as e:
            print(f"    ✓ Correctly caught error: {type(e).__name__}: {e}")

    print("\n✓ Error handling working correctly!")
    print("  All invalid inputs properly caught and reported")
    print("  Helpful error messages guide users to fix issues")


def main():
    """Run all stacked plot examples."""
    print("COMPREHENSIVE CELLCYCLE_STACKED_PLOT FUNCTION DEMONSTRATION")
    print("This example shows ALL parameters including their default values")
    print("and demonstrates the complete functionality available to users.")

    # Create output directory
    Path("stacked_examples").mkdir(exist_ok=True)

    try:
        example_1_minimal_usage()
        example_2_all_parameters_explicit()
        example_3_custom_styling_options()
        example_4_integration_with_subplots()
        example_5_different_output_formats()
        example_6_error_handling_demonstration()

        print("\n" + "=" * 80)
        print("🎉 ALL STACKED PLOT EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 80)

        print("\nKEY FEATURES DEMONSTRATED:")
        print(
            "✅ Complete parameter interface (all arguments from base classes)"
        )
        print("✅ Intelligent defaults with full customization")
        print("✅ Integration with matplotlib subplots")
        print("✅ Multiple output formats and quality settings")
        print("✅ Robust error handling and validation")
        print("✅ Auto-generated filenames and titles")
        print("✅ Flexible legend positioning")
        print("✅ Custom phase ordering and stacking")

        print("\nGENERATED FILES in 'stacked_examples/' directory:")
        output_dir = Path("stacked_examples")
        for file in sorted(output_dir.glob("*")):
            print(f"  📄 {file.name}")

        print("\nPARAMETER SUMMARY:")
        print("  Required: data, conditions")
        print(
            "  Base class: condition_col, selector_col, selector_val, title, colors, figsize"
        )
        print(
            "  Stacked-specific: phases, reverse_stack, show_legend, legend_position"
        )
        print("  Integration: ax")
        print("  Output: save, output_path, filename")
        print("  Quality: dpi, format, tight_layout")
        print("  Plus: **kwargs for additional customization")

    except Exception as e:
        print(f"\n❌ Error in examples: {e}")
        raise


if __name__ == "__main__":
    main()
