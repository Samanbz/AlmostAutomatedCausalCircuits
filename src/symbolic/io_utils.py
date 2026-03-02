from os import PathLike
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

import graphviz

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
    if output_path is not None:
        path_obj = Path(output_path)
        fmt = path_obj.suffix.replace(".", "")
        if not fmt:
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
        for cls, config in node_config.items():
            if isinstance(node_data, cls):
                style = config
                break

        # Attributes
        color = style.get("color", "white")
        label = style.get("label", str(node_id))
        shape = style.get("shape", node_shape)

        if callable(label):
            label = label(node_data)
        if callable(color):
            color = color(node_data)

        # Map common shape names if necessary, or pass through
        # graphviz supports: box, ellipse, circle, diamond, etc.

        dot.node(
            str(node_id),
            label=str(label),
            fillcolor=str(color),
            shape=str(shape),
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
        # graphviz.render adds suffix automatically, but dot.render with cleanup=True allows explicit paths
        print(f"Rendering graph to {output_path}...")
        try:
            # render(filename, subdirectory, view, cleanup)
            dot.render(str(out_file.with_suffix("")), cleanup=True)
            print("Done.")
        except Exception as e:
            print(f"Graphviz render failed: {e}. Check if graphviz is installed on system.")

    return dot
