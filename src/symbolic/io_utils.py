from os import PathLike
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

from src.symbolic.scm import StructuralCausalModel

from .base import DirectedAcyclicGraph


def plot_dag(
    dag: DirectedAcyclicGraph,
    node_config: Optional[Dict[Type, Dict[str, Any]]] = None,
    output_path: Optional[Union[str, PathLike]] = None,
    orientation: str = "vertical",
    node_shape: str = "circle",
    rank_sep: float = 0.5,
    node_sep: float = 0.5,
    show_edge_data: bool = False,
) -> graphviz.Digraph:
    """
    Plots a DirectedAcyclicGraph using Graphviz.

    Args:
        dag: The graph object to plot.
        node_config: Dictionary mapping Node classes to style attributes.
                     e.g. {SumNode: {'color': 'red', 'label': '+'}, ...}
                     Values for 'label' and 'color' can be strings or callables taking the node as input.
                     If None, uses dag.node_config.
        output_path: Path to save the image. If None, returns object for Jupyter display.
        orientation: "vertical" (Top-Down) or "horizontal" (Left-Right).
        node_shape: Default shape for nodes if not specified in config.
        rank_sep: Vertical separation between layers (inches).
        node_sep: Horizontal separation between nodes (inches).
        show_edge_data: Whether to show edge weights/data on the edges.
    """

    if node_config is None:
        node_config = dag.get_node_config()

    # 1. Determine Format from Path
    fmt = "svg"
    is_html = False
    if output_path is not None:
        path_obj = Path(output_path)
        fmt = path_obj.suffix.replace(".", "").lower()
        if fmt == "html":
            is_html = True
            fmt = "svg"
        elif not fmt:
            fmt = "svg"  # Default if no extension provided

    # 2. Configure Graphviz
    dot = graphviz.Digraph(
        format=fmt,
        node_attr={
            "shape": node_shape,
            "style": "filled",
            "fontname": "Helvetica",
            "fixedsize": "false",
        },
        engine="dot",
    )

    # Custom Graph Attributes for large graphs
    dot.attr(rankdir="TB" if orientation == "vertical" else "LR")
    dot.attr(ranksep=str(rank_sep))
    dot.attr(nodesep=str(node_sep))
    dot.attr(splines="false")  # Straight lines for better readability
    dot.attr(overlap="false")

    # 3. Add Nodes
    for node_id, node_data in dag._nodes.items():
        # Resolve Config for this node type
        style = {}
        best_cls = None
        for cls, config in node_config.items():
            if isinstance(node_data, cls):
                if best_cls is None or issubclass(cls, best_cls):
                    style = config
                    best_cls = cls

        # Attributes
        color = style.get("color", "white")
        label = style.get("label", str(node_id))
        shape = style.get("shape", node_shape)
        fontcolor = style.get("fontcolor", "black")
        fontname = style.get("fontname", "Helvetica")

        if callable(label):
            label = label(node_data)
        if callable(color):
            color = color(node_data)
        if callable(fontcolor):
            fontcolor = fontcolor(node_data)
        if callable(fontname):
            fontname = fontname(node_data)

        # Map common shape names if necessary, or pass through
        # graphviz supports: box, ellipse, circle, diamond, etc.

        dot.node(
            str(node_id),
            label=str(label),
            fillcolor=str(color),
            shape=str(shape),
            fontcolor=str(fontcolor),
            fontname=str(fontname),
            tooltip=repr(node_data),
        )

    # 4. Add Edges
    for source_id, targets in dag._adj.items():
        for target_id, edge_data in targets.items():
            edge_attrs = {}
            if show_edge_data and isinstance(edge_data, (int, float)):
                edge_attrs["label"] = f"{edge_data:.2f}"
                edge_attrs["fontsize"] = "8"
                edge_attrs["fontcolor"] = "black"
                edge_attrs["decorate"] = "true"  # Connect label to edge with a line if needed

            dot.edge(str(source_id), str(target_id), **edge_attrs)

    # 5. Render or Return
    if output_path is not None:
        out_file = Path(output_path)
        print(f"Rendering graph to {output_path}...")
        try:
            if is_html:
                svg_data = dot.pipe(format="svg").decode("utf-8")
                html_content = f"<!DOCTYPE html>\n<html>\n<head>\n<title>DAG Plot</title>\n</head>\n<body style='text-align: center; margin-top: 50px;'>\n{svg_data}\n</body>\n</html>"
                out_file.write_text(html_content, encoding="utf-8")
            else:
                dot.render(str(out_file.with_suffix("")), cleanup=True)
            print("Done.")
        except Exception as e:
            print(f"Graphviz render failed: {e}. Check if graphviz is installed on system.")

    return dot


def plot_scm(
    scm: StructuralCausalModel,
    output_path: Optional[Union[str, PathLike]] = None,
    orientation: str = "vertical",
) -> graphviz.Digraph:
    """
    Plots a StructuralCausalModel using Graphviz.
    Exogenous confounders are rendered as dashed bidirected arcs between their observable children.
    """
    fmt = "svg"
    is_html = False
    if output_path is not None:
        path_obj = Path(output_path)
        fmt = path_obj.suffix.replace(".", "").lower()

        if fmt == "html":
            is_html = True
            fmt = "svg"

        VALID_FORMATS = {
            "bmp",
            "canon",
            "cgimage",
            "cmap",
            "cmapx",
            "cmapx_np",
            "dot",
            "dot_json",
            "eps",
            "exr",
            "fig",
            "gd",
            "gd2",
            "gif",
            "gtk",
            "gv",
            "ico",
            "imap",
            "imap_np",
            "ismap",
            "jp2",
            "jpe",
            "jpeg",
            "jpg",
            "json",
            "json0",
            "pct",
            "pdf",
            "pic",
            "pict",
            "plain",
            "plain-ext",
            "png",
            "pov",
            "ps",
            "ps2",
            "psd",
            "sgi",
            "svg",
            "svg_inline",
            "svgz",
            "tga",
            "tif",
            "tiff",
            "tk",
            "vml",
            "vmlz",
            "vrml",
            "wbmp",
            "webp",
            "x11",
            "xdot",
            "xdot1.2",
            "xdot1.4",
            "xdot_json",
            "xlib",
        }
        if not is_html and fmt not in VALID_FORMATS:
            print(
                f"Warning: Extension '{fmt}' is not a valid graphviz format. Defaulting format to 'svg'."
            )
            fmt = "svg"

    dot = graphviz.Digraph(format=fmt, engine="dot")

    dot.attr(rankdir="TB" if orientation == "vertical" else "LR")
    dot.attr(ranksep="0.8")
    dot.attr(nodesep="0.6")
    dot.attr(splines="curved")

    # Observable nodes
    for node_id in sorted(scm.observable_variables):
        dot.node(
            str(node_id),
            label=f'<<font face="Times-BoldItalic"><b><i>{node_id}</i></b></font>>',
            shape="circle",
            style="filled",
            fillcolor="white",
            color="black",
            penwidth="1.5",
            fontname="Times-BoldItalic",
            fixedsize="true",
            width="0.45",
            height="0.45",
        )

    # Directed causal edges (observable parents only)
    for source_id, targets in scm._adj.items():
        if source_id in scm.exogenous_variables:
            continue
        for target_id in targets:
            if target_id in scm.observable_variables:
                dot.edge(
                    str(source_id),
                    str(target_id),
                    color="black",
                    penwidth="1.2",
                    arrowsize="0.7",
                )

    # Bidirected arcs for exogenous confounders
    for exo_id in sorted(scm.exogenous_variables):
        children = sorted(c for c in scm.get_children(exo_id) if c in scm.observable_variables)
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                dot.edge(
                    str(children[i]),
                    str(children[j]),
                    dir="both",
                    style="dashed",
                    color="black",
                    penwidth="1.2",
                    arrowsize="0.7",
                    constraint="false",
                )

    if output_path is not None:
        out_file = Path(output_path)
        print(f"Rendering SCM to {output_path}...")
        try:
            if is_html:
                svg_data = dot.pipe(format="svg").decode("utf-8")
                html_content = (
                    "<!DOCTYPE html>\n<html>\n<head>\n<title>SCM Plot</title>\n</head>\n"
                    "<body style='text-align: center; margin-top: 50px;'>\n"
                    f"{svg_data}\n</body>\n</html>"
                )
                out_file.write_text(html_content, encoding="utf-8")
            else:
                dot.render(str(out_file.with_suffix("")), cleanup=True)
            print("Done.")
        except Exception as e:
            print(f"Graphviz render failed: {e}. Check if graphviz is installed on system.")

    return dot


def _plot_grid(
    matrix: np.ndarray | torch.Tensor,
    cmap: str,
    vmin: float,
    vmax: float,
    title: str,
    filename: str = None,
) -> None:
    """Helper to plot a matrix as a grid with specific colormaps and bounds."""
    rows, cols = matrix.shape
    plt.figure(figsize=(8, 8))
    plt.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest")

    # Gridlines configuration
    ax = plt.gca()

    if rows * cols < 400:  # Only show labels if matrix is small enough
        ax.set_xticks(np.arange(cols))
        ax.set_yticks(np.arange(rows))
    else:
        # For large matrices, turn off labels to avoid clutter
        ax.set_xticks([])
        ax.set_yticks([])

    # Minor ticks at half-integers (-0.5, 0.5...) for gridlines
    # Use simpler grid generation for large matrices to avoid performance hit
    if rows * cols < 2500:
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=1)
    else:
        # If too big, skip the gridlines or they will just make it gray block
        pass

    # Ensure major gridlines are off (in case they are on by default)
    ax.grid(which="major", visible=False)
    # Remove minor tick marks (the little lines sticking out)
    ax.tick_params(which="minor", size=0)
    ax.tick_params(which="major", size=0)

    plt.title(title)
    plt.tight_layout()

    if filename:
        plt.savefig(filename)
        print(f"Plot saved to {filename}")
    else:
        plt.show()


def plot_matrix(
    matrix: np.ndarray | torch.Tensor, title: str = "Connection Matrix", filename: str = None
) -> None:
    """
    Utility function to visualize a boolean matrix as a grid.
    Black for 1 (True), White for 0 (False).
    """
    _plot_grid(matrix, cmap="binary", vmin=0, vmax=1, title=title)


def plot_diff_matrix(
    matrix: np.ndarray | torch.Tensor, title: str = "Difference Matrix", filename: str = None
) -> None:
    """
    Utility function to visualize a difference matrix as a grid.
    Values must be in the range [-1, 1]. Colors range from red (-1) to white (0) to green (1).
    """
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()

    if np.any(matrix < -1) or np.any(matrix > 1):
        raise ValueError("Matrix contains values outside the range [-1, 1].")

    # Custom colormap: Red (-1) -> White (0) -> Green (1)
    cmap = LinearSegmentedColormap.from_list("RdWhGn", ["red", "white", "green"])

    _plot_grid(matrix, cmap=cmap, vmin=-1, vmax=1, title=title)
