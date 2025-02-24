from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec

from . import plot_helper, an
from .CCIData_class import CCIData


def network_plot(
    network,
    p_vals=None,
    diff_plot=False,
    normalise=True,
    remove_unconnected=True,
    show_labels=False,
    p_val_cutoff=0.05,
    edge_weight=50,
    text_size=15,
    node_size=2500,
    figsize=None,
    arrowsize=20,
    node_label_dist=1,
    p_val_text_size=10,
    node_colors=None,
    node_palette="tab20",
    outer_node_palette=None,
    show=True,
    show_legend=True,
    legend_size=12,
    title=None,
    title_size=14
):
    """Plots a network with optional edge significance highlighting and node coloring based on in-degree and out-degree difference.

    Args:
        network (pandas.DataFrame or numpy.ndarray): The adjacency matrix representing the network.
        p_vals (pandas.DataFrame or numpy.ndarray, optional): A matrix of p-values corresponding to the edges in `network`. If not provided, significance values will not be plotted. Defaults to None.
        diff_plot (bool, optional): Whether you are plotting the network difference, to show up and down-regulated edges. Defaults to False.
        normalise (bool, optional): Whether to normalize the network matrix before plotting. Defaults to True.
        remove_unconnected (bool, optional): Whether to remove cell types that do not interact with any cell types. Defaults to True.
        show_labels (bool, optional): Whether to show node labels. Defaults to True.
        p_val_cutoff (float, optional): The p-value cutoff for determining significant edges. Defaults to 0.05.
        edge_weight (float, optional): The base weight for edges. Defaults to 20.
        text_size (int, optional): The font size for node labels. Defaults to 15.
        node_size (int, optional): The size of the nodes. Defaults to 2500.
        figsize (tuple, optional): The size of the figure. Defaults to None.
        arrowsize (int, optional): The size of the arrow heads for edges. Defaults to 50.
        node_label_dist (float, optional): A factor for adjusting the distance between nodes and labels. Defaults to 1.
        p_val_text_size (int, optional): The font size for p-value labels. Defaults to 10.
        node_colors (dict, optional): A dictionary of colors for each node. Overwrites node_palette. Defaults to None.
        node_palette (str, optional): The name of the color palette to use for nodes. Defaults to "tab20".
        outer_node_palette (str, optional): The name of the color palette to use for outer nodes to show sender/reciever nodes. Defaults to None.
        show (bool, optional): Whether to show the plot or not. Defaults to True.
        show_legend (bool, optional): Whether to show legend. Defaults to False.
        legend_size (int, optional): Font size for legend. Defaults to 12.
        title (str, optional): Title of the plot. Defaults to None.
        title_size (int, optional): Font size for title. Defaults to 14.

    Returns:
        tuple: A tuple containing the figure and axis objects.
    """

    if not isinstance(network, pd.DataFrame):
        raise ValueError("Input should be a dataframe.")
    
    if figsize is None:
        if show_legend:
            figsize = (10, 8)
        else:
            figsize = (8, 8)

    # Adjust the figure layout to accommodate the legend on the right
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1], wspace=0.2)

    # Main plot area
    ax = fig.add_subplot(gs[0])
    plt.sca(ax)  # Set the current axis to the main plot area

    if remove_unconnected:
        network = network.loc[(network != 0).any(axis=1), (network != 0).any(axis=0)]
        network = network.loc[network.index.isin(
            network.columns), network.columns.isin(network.index)]

    if normalise:
        if network.min().min() >= 0:
            network = network / network.sum().sum()
        else:
            network = network / network.abs().sum().sum()
    network_abs = abs(network)

    if normalise:
        network_abs = network_abs / network_abs.sum().sum()
    network_abs = network.astype(float)
    G_network = nx.from_pandas_adjacency(network_abs, create_using=nx.DiGraph)
    pos = nx.circular_layout(G_network)
    weights = nx.get_edge_attributes(G_network, "weight")

    # Calculate the in-degree and out-degree for each node
    in_degree = dict(G_network.in_degree(weight="weight"))
    out_degree = dict(G_network.out_degree(weight="weight"))

    in_out_diff = {node: in_degree[node] - out_degree[node] for node in G_network.nodes}

    max_diff = max(abs(value) for value in in_out_diff.values())
    color_scale = np.linspace(-max_diff, max_diff, 256)

    if outer_node_palette is None:
        # Create a color scale based on the in-degree and out-degree difference
        cmap_colors = [(1, 0, 0), (0.7, 0.7, 0.7), (0, 0, 1)]  # Blue, Grey, Red
        outer_node_cmap = LinearSegmentedColormap.from_list("custom_cmap", cmap_colors)
    else:
        outer_node_cmap = plt.get_cmap(outer_node_palette)

    
    edge_colors = []
    # Map node colors to the in-degree and out-degree difference
    if sum(abs(value) for value in in_out_diff.values()) == 0:
        edge_colors = ['grey' for node in G_network.nodes]
    else:
        edge_colors = [
            outer_node_cmap(int(np.interp(in_out_diff[node], color_scale, range(256))))
            for node in G_network.nodes
        ]

    if node_colors is not None:
        node_colors_list = [node_colors[node] for node in G_network.nodes]
    else:
        if node_palette is not None:
            node_colors_list = \
                list(plt.get_cmap(node_palette).colors)[:len(G_network.nodes)]
        else:
            node_colors_list = ["grey" for node in G_network.nodes]

    if p_vals is None or diff_plot == False:
        # Create a non-significant matrix
        p_vals = network_abs.replace(network_abs.values, 1, inplace=False)
    else:
        # Prevent removal of pvals of 0
        p_vals[p_vals == 0] = 1e-300

    # Get edges that are significant
    G_p_vals = nx.from_pandas_adjacency(p_vals, create_using=nx.DiGraph)
    G_network_updown = nx.from_pandas_adjacency(network, create_using=nx.DiGraph)

    non_sig_up = [
        (u, v) for (u, v, d) in G_p_vals.edges(data=True)
        if d["weight"] > p_val_cutoff
        and u in G_network_updown
        and v in G_network_updown[u]
        and G_network_updown[u][v]["weight"] > 0]
    non_sig_up = [edge for edge in non_sig_up if edge in weights.keys()]

    non_sig_down = [
        (u, v) for (u, v, d) in G_p_vals.edges(data=True)
        if d["weight"] > p_val_cutoff
        and u in G_network_updown
        and v in G_network_updown[u]
        and G_network_updown[u][v]["weight"] < 0]
    non_sig_down = [edge for edge in non_sig_down if edge in weights.keys()]

    sig_up = [
        (u, v) for (u, v, d) in G_p_vals.edges(data=True)
        if d["weight"] <= p_val_cutoff
        and u in G_network_updown
        and v in G_network_updown[u]
        and G_network_updown[u][v]["weight"] > 0]
    sig_up = [edge for edge in sig_up if edge in weights.keys()]

    sig_down = [
        (u, v) for (u, v, d) in G_p_vals.edges(data=True) if d["weight"] <= p_val_cutoff
        and u in G_network_updown
        and v in G_network_updown[u]
        and G_network_updown[u][v]["weight"] < 0]
    sig_down = [edge for edge in sig_down if edge in weights.keys()]

    edge_thickness_non_sig_up = []
    edge_thickness_non_sig_down = []
    edge_thickness_sig_up = []
    edge_thickness_sig_down = []

    for edge in weights.keys():
        if edge in non_sig_up:
            edge_thickness_non_sig_up.append(weights[edge] * edge_weight)
        else:
            edge_thickness_non_sig_up.append(0)

        if edge in non_sig_down:
            edge_thickness_non_sig_down.append(weights[edge] * edge_weight)
        else:
            edge_thickness_non_sig_down.append(0)

        if edge in sig_up:
            edge_thickness_sig_up.append(weights[edge] * edge_weight)
        else:
            edge_thickness_sig_up.append(0)

        if edge in sig_down:
            edge_thickness_sig_down.append(weights[edge] * edge_weight)
        else:
            edge_thickness_sig_down.append(0)

    # if node_colors is None:
    nx.draw_networkx_nodes(
        G_network,
        pos,
        node_size=node_size,
        node_color=node_colors_list,
        edgecolors=edge_colors,
        linewidths=8.0,
    )

    if diff_plot:
        # Draw non-self edges first
        nx.draw_networkx_edges(
            G_network,
            pos,
            node_size=node_size * 2,
            connectionstyle="arc3,rad=0.08",
            width=edge_thickness_non_sig_up,
            arrows=True,
            arrowstyle="->",
            arrowsize=arrowsize,
            edge_color="pink"
        )
        
        # Same pattern for non-sig down edges
        nx.draw_networkx_edges(
            G_network,
            pos,
            node_size=node_size * 2,
            connectionstyle="arc3,rad=0.08",
            width=edge_thickness_non_sig_down,
            arrows=True,
            arrowstyle="->",
            arrowsize=arrowsize,
            edge_color="lightgreen"
            )

    else:
        # Non-self edges
        nx.draw_networkx_edges(
            G_network,
            pos,
            node_size=node_size * 2,
            connectionstyle="arc3,rad=0.08",
            width=edge_thickness_non_sig_up,
            arrows=True,
            arrowstyle="->",
            arrowsize=arrowsize,
        )

        # Same for non-sig down edges
        nx.draw_networkx_edges(
            G_network,
            pos,
            node_size=node_size * 2,
            connectionstyle="arc3,rad=0.08",
            width=edge_thickness_non_sig_down,
            arrows=True,
            arrowstyle="->",
            arrowsize=arrowsize,
        )

    # Significant up edges
    nx.draw_networkx_edges(
        G_network,
        pos,
        node_size=node_size * 2,
        connectionstyle="arc3,rad=0.08",
        width=edge_thickness_sig_up,
        arrows=True,
        arrowstyle="->",
        arrowsize=arrowsize,
        edge_color="purple",
    )

    # Significant down edges  
    nx.draw_networkx_edges(
        G_network,
        pos,
        node_size=node_size * 2,
        connectionstyle="arc3,rad=0.08",
        width=edge_thickness_sig_down,
        arrows=True,
        arrowstyle="->",
        arrowsize=arrowsize,
        edge_color="green",
    )

    edge_labels = nx.get_edge_attributes(G_p_vals, "weight")
    edge_labels = {
        key: edge_labels[key] for key in G_network.edges().keys() if key in edge_labels
    }

    # Add edge labels for significant edges
    for key, value in edge_labels.items():
        if value > p_val_cutoff:
            edge_labels[key] = ""
        else:
            edge_labels[key] = round(value, 3)

    def offset(d, pos, dist=0.05, loop_shift=0.1):
        for (u, v), obj in d.items():
            if u != v:
                par = dist * (pos[v] - pos[u])
                dx, dy = par[1], -par[0]
                x, y = obj.get_position()
                obj.set_position((x + dx, y + dy))
            else:
                x, y = obj.get_position()
                obj.set_position((x, y + loop_shift))

    d = nx.draw_networkx_edge_labels(
        G_network, 
        pos, 
        edge_labels, 
        font_size=p_val_text_size, 
        connectionstyle="arc3,rad=0.08"
        )

    offset(d, pos)

    pos.update(
        (x, [y[0] * 1.4 * node_label_dist, y[1] * (1.25 + 0.05) * node_label_dist])
        for x, y in pos.items()
    )

    if show_labels:
        nx.draw_networkx_labels(
            G_network,
            pos,
            font_weight="bold",
            font_color="black",
            font_size=text_size,
            clip_on=False,
            horizontalalignment="center",
        )

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    
    if show_legend:
        # Create a color bar in the bottom-right corner
        color_bar_ax = fig.add_axes([0.77, 0.2, 0.05, 0.2])
        if sum(abs(value) for value in in_out_diff.values()) != 0:
            sm = plt.cm.ScalarMappable(cmap=outer_node_cmap,
                                    norm=plt.Normalize(vmin=-max_diff, vmax=max_diff))
            sm.set_array([])
            cbar = plt.colorbar(sm, cax=color_bar_ax)
            cbar.set_ticks([])  # Remove the ticks
            cbar.set_label('Net sender ← → Net receiver', fontsize=legend_size)

        # Add the legend in the top-right corner
        legend_elements = []
        if diff_plot:
            legend_elements.extend([
                plt.Line2D([0], [0], color='pink', lw=2, 
                           label='Non-significant positive'),
                plt.Line2D([0], [0], color='lightgreen', lw=2, 
                           label='Non-significant negative'),
                plt.Line2D([0], [0], color='purple', lw=2, 
                           label='Significant positive'),
                plt.Line2D([0], [0], color='green', lw=2, 
                           label='Significant negative')
            ])

        if node_colors is not None:
            for node, color in node_colors.items():
                if node in network.index:
                    legend_elements.append(
                        plt.Line2D([0], [0], marker='o', color='w',
                                markerfacecolor=color, markersize=10, label=node)
                    )
        else:
            for i in range(len(G_network.nodes)):
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w',
                            markerfacecolor=node_colors_list[i], markersize=10, 
                            label=list(G_network.nodes.keys())[i])
                )

        if legend_elements:
            legend_ax = fig.add_axes([0.7, 0.55, 0.2, 0.3])
            legend_ax.axis('off')
            legend_ax.legend(handles=legend_elements, loc='center', 
                             fontsize=legend_size)

    if title is not None:
        plt.suptitle(title, fontsize=title_size, y = 0.875, fontweight="bold")

    plt.tight_layout()
    if show:
        plt.show()
    else:
        return fig, ax


def chord_plot(
    network,
    min_int=0.001,
    n_top_ccis=10,
    colors=None,
    show=True,
    title=None,
    title_size=14,
    label_size=10,
    figsize=None,
    show_legend=False,
    legend_size=12
):
    """Plots a chord plot of a network

    Args:
        network (pandas.DataFrame or numpy.ndarray): The adjacency matrix representing the network.
        min_int (float): Minimum interactions to display cell type. Defaults to 0.01.
        n_top_ccis (int): Number of top cell types to display. Defaults to 10.
        colors (dict): Dict of colors for each cell type to use for the plot. Defaults to None.
        show (bool): Whether to show plot or not. Defaults to True.
        title (str): Title of the plot. Defaults to None.
        title_size (int): Font size of the title. Defaults to 14.
        label_size (int): Font size of the labels. Defaults to None.
        figsize (tuple): Size of the figure. Defaults to None.
        show_legend (bool): Whether to show legend. Defaults to False.
        legend_size (int): Font size for legend. Defaults to 12.

    Returns:
        tuple: A tuple containing the figure and axis objects.
    """

    if not isinstance(network, pd.DataFrame):
        raise ValueError("Input should be a dataframe.")

    network = network.transpose()
    
    if figsize is None:
        if show_legend:
            figsize = (10, 8)
        else:
            figsize = (8, 8)
    
    # Create figure with gridspec to accommodate legend
    fig = plt.figure(figsize=figsize)
    if show_legend:
        gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0)
        ax = fig.add_subplot(gs[0])
    else:
        ax = plt.axes([0, 0, 1, 1])

    flux = network.values

    total_ints = flux.sum(axis=1) + flux.sum(axis=0) - flux.diagonal()
    keep = total_ints > min_int
    # Limit of 10 for good display #
    if sum(keep) > n_top_ccis:
        keep = np.argsort(-total_ints)[0:n_top_ccis]
    flux = flux[:, keep]
    flux = flux[keep, :].astype(float)
    cell_names = network.index.values.astype(str)[keep]
    nodes = cell_names

    color_list = []
    if colors is not None:
        for cell in cell_names:
            color_list.append(colors[cell])
    else:
        color_list = None

    nodePos = plot_helper.chordDiagram(flux, ax, lim=1.25, colors=color_list)
    ax.axis("off")
    prop = dict(fontsize=label_size, ha="center", va="center")

    for i in range(len(cell_names)):
        x, y = nodePos[i][0:2]
        if label_size != 0:
            ax.text(x, y, nodes[i], rotation=nodePos[i][2], **prop)

    if show_legend and colors is not None:
        # Add the legend in the right subplot
        legend_ax = fig.add_subplot(gs[1])
        legend_ax.axis('off')
        
        legend_elements = []
        for node, color in colors.items():
            if node in cell_names:
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w',
                            markerfacecolor=color, markersize=10, label=node)
                )
        
        if legend_elements:
            legend_ax.legend(handles=legend_elements, loc='center', 
                             fontsize=legend_size)

    if title is not None:
        fig.suptitle(title, fontsize=title_size, y=0.95, fontweight="bold")

    plt.tight_layout()

    if show:
        plt.show()
    else:
        return fig, ax


def dissim_hist(
    dissimilarity_scores,
    x_label_size=18,
    y_label_size=24,
    x_tick_size=14,
    y_tick_size=12,
    figsize=(6, 5),
    show=True,
    title=None,
    title_size=14
):
    """Plots a histogram of dissimilarity scores.

    Args:
        dissimilarity_scores (dict): A dictionary of dissimilarity scores.
        x_label_size (int): Font size for x-axis label. Defaults to 18.
        y_label_size (int): Font size for y-axis label. Defaults to 24.
        x_tick_size (int): Font size for ticks. Defaults to 14.
        y_tick_size (int): Font size for ticks. Defaults to 12.
        figsize (tuple): Size of the figure. Defaults to (10, 8).
        show (bool): Whether to show the plot or not. Defaults to True.
        title (str): Title of the plot. Defaults to None.
        title_size (int): Font size of the title. Defaults to 14.

    Returns:
        matplotlib.figure.Figure: The figure
    """

    fig = plt.figure(figsize=figsize)
    plt.style.use('default')
    plt.hist(list(dissimilarity_scores.values()))
    plt.xlim(0, 1)
    plt.xlabel("Dissimilarity Score", fontsize=x_label_size)
    plt.ylabel("Count", fontsize=y_label_size)
    plt.tick_params(axis='x', which='major', labelsize=x_tick_size)
    plt.tick_params(axis='y', which='major', labelsize=y_tick_size)

    if title is not None:
        plt.title(title, fontsize=title_size, pad=20)

    if show:
        plt.show()
    else:
        plt.close(fig)
        return fig


def lr_top_dissimilarity(
    dissimilarity_scores,
    n=10,
    top=True,
    x_label_size=18,
    y_label_size=24,
    x_tick_size=14,
    y_tick_size=12,
    figsize=(6, 5),
    show=True,
    title=None,
    title_size=14
):
    """Plots a bar plot of LR pairs with highest/lowest dissimilarity scores.

    Args:
        dissimilarity_scores (dict): A dictionary of dissimilarity scores.
        n (int): Number of LR pairs to plot.
        top (bool): If True, plot LR pairs with highest dissimilarity scores. If False, plot LR pairs with lowest dissimilarity scores.
        x_label_size (int): Font size for x-axis label. Defaults to 18.
        y_label_size (int): Font size for y-axis label. Defaults to 24.
        x_tick_size (int): Font size for ticks. Defaults to 14.
        y_tick_size (int): Font size for ticks. Defaults to 12.
        figsize (tuple): Size of the figure. Defaults to (10, 8).
        show (bool): Whether to show the plot or not. Defaults to True.
        title (str): Title of the plot. Defaults to None.
        title_size (int): Font size of the title. Defaults to 14.

    Returns:
        matplotlib.figure.Figure: The figure
    """

    reverse = not top
    sorted_items = sorted(
        dissimilarity_scores.items(), key=lambda x: x[1], reverse=reverse
    )
    top_n_items = sorted_items[-n:]
    keys, values = zip(*top_n_items)

    fig = plt.figure(figsize=figsize)
    plt.style.use('default')
    plt.barh(keys, values)
    plt.xlabel("Dissimilarity Score", fontsize=x_label_size)
    plt.ylabel("LR Pair", fontsize=y_label_size)
    plt.tick_params(axis='x', which='major', labelsize=x_tick_size)
    plt.tick_params(axis='y', which='major', labelsize=y_tick_size)
    
    if title is not None:
        plt.title(title, fontsize=title_size, pad=20)

    if show:
        plt.show()
    else:
        plt.close(fig)
        return fig
    

def lr_barplot(
    sample,
    assay="raw",
    n=15,
    x_label_size=18,
    y_label_size=24,
    x_tick_size=14,
    y_tick_size=12,
    figsize=(6, 5),
    show=True,
    title=None,
    title_size=14
):
    """Plots a bar plot of LR pairs and their proportions for a sample.

    Args:
        sample (CCIData): The CCIData object.
        assay (str): The assay to use. Defaults to "raw".
        n (int): Number of LR pairs to plot. If None, plot all LR pairs. Defaults to 15.
        x_label_size (int): Font size for x-axis label. Defaults to 18.
        y_label_size (int): Font size for y-axis label. Defaults to 24.
        x_tick_size (int): Font size for tick labels. Defaults to 14.
        y_tick_size (int): Font size for tick labels. Defaults to 12.
        figsize (tuple): Size of the figure. Defaults to (10, 8).
        title (str) (optional): Title for the plot. Defaults to None.
        title_size (int): Font size of the title. Defaults to 14.

    Returns:
        matplotlib.figure.Figure: The figure
    """

    if assay not in sample.assays:
        raise ValueError("Assay not found in sample.")

    interactions = [(lr, df.sum().sum()) for (lr, df) \
        in sample.assays[assay]['cci_scores'].items()]
    interactions.sort(key=lambda x: x[1])
    interactions = interactions[-n:]
    keys, values = zip(*interactions)
    values = [value / sum(values) for value in values]

    fig = plt.figure(figsize=figsize)
    plt.style.use('default')
    plt.barh(keys, values)
    plt.xlabel("Relative Interaction Strength", fontsize=x_label_size)
    plt.ylabel("LR Pair", fontsize=y_label_size)
    plt.tick_params(axis='x', which='major', labelsize=x_tick_size)
    plt.tick_params(axis='y', which='major', labelsize=y_tick_size)

    if title:
        plt.title(title, pad=20, fontsize=title_size)
    
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.close(fig)
        return fig


def lrs_per_celltype(
    sample,
    sender = None,
    receiver = None,
    assay="raw",
    key="cci_scores",
    p_vals=None,
    n=15,
    x_label_size=18,
    y_label_size=24,
    x_tick_size=14,
    y_tick_size=12,
    figsize=(6, 5),
    show=True,
    title=None,
    title_size=14
):
    """Plots a bar plot of LR pairs and their proportions for a sender and receiver cell type pair along with p_values (optional).

    Args:
        sample (CCIData): The CCIData object.
        sender (str): The sender cell type. Defaults to None.
        receiver (str): The receiver cell type. Defaults to None.
        assay (str): The assay to use. Defaults to "raw".
        key (str): The key to use. Defaults to "cci_scores".
        p_vals (dict): A dictionary of p-values. Defaults to None.
        n (int): Number of LR pairs to plot. Defaults to 15.
        x_label_size (int): Font size for x-axis label. Defaults to 18.
        y_label_size (int): Font size for y-axis label. Defaults to 24.
        x_tick_size (int): Font size for tick labels. Defaults to 14.
        y_tick_size (int): Font size for tick labels. Defaults to 12.
        figsize (tuple): Size of the figure. Defaults to (10, 8).
        title (str) (optional): Title for the plot. Defaults to None.
        title_size (int): Font size of the title. Defaults to 14.

    Returns:
        matplotlib.figure.Figure: The figure
    """

    pairs = sample.get_lr_proportions(sender, receiver, assay, key)
    keys = list(pairs.keys())[:n]
    values = list(pairs.values())[:n]
    keys.reverse()
    values.reverse()
    if p_vals is not None:
        p_val_pairs = an.get_p_vals_per_celltype(p_vals, sender, receiver)
        labels = [p_val_pairs[key] for key in keys]

        # make labels readable (if less than 0.00001, show as <0.00001)
        for i in range(len(labels)):
            if labels[i] < 0.001:
                labels[i] = "<0.001"
            else:
                labels[i] = f"{labels[i]:.3f}"

        # Define colors based on p-values
        colors = [
            '#1f77b4' if val < 0.05 else 'grey' for val in [
                p_val_pairs[key] for key in keys]]

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    plt.style.use('default')

    if p_vals is None:
        ax.barh(keys, values)
    else:
        bars = ax.barh(keys, values, color=colors)
        ax.bar_label(bars, labels)

    ax.set_xlabel("Proportion", fontsize=x_label_size)
    ax.set_ylabel("LR Pair", fontsize=y_label_size)
    ax.tick_params(axis='x', which='major', labelsize=x_tick_size)
    ax.tick_params(axis='y', which='major', labelsize=y_tick_size)

    if title:
        plt.title(title, pad=20, fontsize=title_size)
        
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.close(fig)
        return fig
