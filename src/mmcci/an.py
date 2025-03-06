import pandas as pd
import numpy as np
import networkx as nx
import scipy as sp
import scanpy
import gseapy as gp
import seaborn as sns
import statistics
import subprocess
import umap
import importlib.resources
import anndata as ad
import os

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from gseapy import barplot, dotplot
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, ClusterWarning
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn import preprocessing as pp
from sklearn.decomposition import PCA
from warnings import simplefilter

from . import sc, pl, tl
from .CCIData_class import CCIData

simplefilter("ignore", Warning)


def calculate_dissim(
    sample1: CCIData,
    sample2: CCIData,
    assay: str = "raw",
    key: str = "cci_scores",
    lmbda: float = 0.5,
    ):
    """Calculates a dissimilarity score between two samples for each common LR pair.
    
    Args:
        sample1, sample2 (CCIData): The two samples to compare.
        assay (str) (optional): The assay to use for the comparison. Defaults to 'raw'.
        key (str) (optional): The key to use for the comparison. Defaults to  'cci_scores'.
        lmbda (float) (optional): The weighting factor for the comparison. Defaults to 0.5.

    Returns:
        dict: A dictionary where keys are common LR pairs and values are the dissimilarity scores.
    """
    
    if assay not in sample1.assays:
        raise ValueError(f"Assay {assay} not found in sample1.")
    if assay not in sample2.assays:
        raise ValueError(f"Assay {assay} not found in sample2.")
    
    if key not in sample1.assays[assay]:
        raise ValueError(f"Key {key} not found in sample1.")
    if key not in sample2.assays[assay]:
        raise ValueError(f"Key {key} not found in sample2.")
    
    dissims = {}
    
    for lr_pair in set(sample1.assays[assay][key].keys()).intersection(
        set(sample2.assays[assay][key].keys())
    ):
        dissims[lr_pair] = sc.dissimilarity_score(
            sample1.assays[assay][key][lr_pair],
            sample2.assays[assay][key][lr_pair],
            lmbda=lmbda
        )
        
    return dissims


def get_network_diff(
    network1: pd.DataFrame,
    network2: pd.DataFrame,
    perm_test: bool = True,
    num_perms: int = 100000
    ):
    """Calculates the difference between two networks. If perm_test is True, also performs permutation testing to assess the significance of the differences.

    Args:
        network1, network2 (pd.DataFrame): Two matrices to compare (as DataFrames).

    Returns:
        dict: A dictionary containing the difference between the two networks and p-values if perm_test is True.
    """

    dfs = tl.align_dataframes(network1, network2)
    diff = dfs[0] - dfs[1]
    p_vals = None
    
    if perm_test:
        abs_diff = abs(diff)

        result_matrix = abs_diff.values.copy()

        def permtr(x):
            return np.apply_along_axis(
                np.random.permutation,
                axis=0,
                arr=np.apply_along_axis(np.random.permutation, axis=1, arr=x),
            )

        # Permute and test for matrix1
        perm = [permtr(abs_diff) for _ in range(num_perms)]
        sums = np.sum([result_matrix < perm_result for perm_result in perm], axis=0)

        # Calculate p-values
        p_vals = sums / num_perms

        p_vals = pd.DataFrame(p_vals, index=abs_diff.index, columns=abs_diff.columns)
    
    return {"diff": diff, "p_vals": p_vals}
        

def cell_network_clustering(
    sample: CCIData, 
    assay: str = "raw",
    n_clusters: int = 0, 
    method: str = "KMeans"
    ):
    """Groups ligand-receptor (LR) pairs into clusters based on their interaction network similarity.
    
    This function:
    1. Separates LR pairs into those with single vs multiple interactions
    2. Calculates dissimilarity scores between all pairs of LR networks
    3. Performs clustering using either KMeans or Hierarchical clustering
    4. Determines optimal cluster number if not specified
    5. Creates new assays in the sample for each cluster

    Args:
        sample (CCIData): Sample containing LR interaction networks to cluster
        assay (str): Name of assay containing interaction data to analyze
        n_clusters (int): Number of desired clusters. If 0, determined automatically
        method (str): Clustering method - either "KMeans" or "Hierarchical"

    Returns:
        CCIData: Sample with new assays added for each cluster found

    Raises:
        ValueError: If specified assay not found in sample
    """

    if assay not in sample.assays:
        raise ValueError(f"Assay {assay} not found in sample.")

    # Dictionary to store LR pairs with only one interaction
    single_interaction_pairs = {}

    def has_non_zero_values(df):
        """Check if dataframe contains any non-zero values"""
        return not (df == 0).all().all()

    # Get all LR pairs with non-zero interactions
    multi_interaction_pairs = {
        key: value for key, value in sample.assays[assay]['cci_scores'].items()
        if has_non_zero_values(value)
    }
    
    # Separate out LR pairs with only a single interaction
    for lr_pair, interaction_matrix in list(multi_interaction_pairs.items()):
        # Check if matrix has only one non-zero value (excluding diagonal)
        diagonal_zeros = (interaction_matrix.values.diagonal() == 0).sum() 
        total_zeros = (interaction_matrix.values == 0).sum()
        matrix_size = interaction_matrix.shape[0]
        
        if (diagonal_zeros == matrix_size - 1 and 
            total_zeros == (matrix_size * matrix_size) - 1):
            single_interaction_pairs[lr_pair] = interaction_matrix
            del multi_interaction_pairs[lr_pair]

    # Process LR pairs with multiple interactions
    if multi_interaction_pairs:
        multi_dissimilarity_matrix = pd.DataFrame(
            index=multi_interaction_pairs.keys(), 
            columns=multi_interaction_pairs.keys()
        )
        
        print("Computing dissimilarity scores for complex interaction networks...")
        with tqdm(total=len(multi_interaction_pairs), desc="Processing") as pbar:
            for pair1, matrix1 in multi_interaction_pairs.items():
                for pair2, matrix2 in multi_interaction_pairs.items():
                    dissim_score = sc.dissimilarity_score(
                        matrix1, matrix2, lmbda=0.5, only_non_zero=True
                    )
                    multi_dissimilarity_matrix.loc[pair1, pair2] = dissim_score
                pbar.update(1)
                
        multi_interaction_clusters = _lr_cluster_helper(
            multi_dissimilarity_matrix, multi_interaction_pairs, n_clusters, method
        )

    # Process LR pairs with single interactions 
    if single_interaction_pairs:
        # Initialize an empty dataframe to store the results
        result_df_one = pd.DataFrame(
            index=single_interaction_pairs.keys(), columns=single_interaction_pairs.keys()
        )
        # Iterate through the keys and compare the dataframes
        print("Computing Dissimilarity Scores for single interactions...")
        with tqdm(total=len(single_interaction_pairs), desc="Processing") as pbar:
            for key1, df1 in single_interaction_pairs.items():
                for key2, df2 in single_interaction_pairs.items():
                    result = sc.dissimilarity_score(
                        df1, df2, lmbda=0.5, only_non_zero=True
                    )
                    
                    # Store the result in the result_df
                    result_df_one.loc[key1, key2] = result
                pbar.update(1)
                
        total = None
        for lr_pair in single_interaction_pairs.keys():
            if single_interaction_pairs[lr_pair].sum().sum() > 0:
                if total is not None:
                    total = total + single_interaction_pairs[lr_pair] / \
                    single_interaction_pairs[lr_pair].sum().sum()
                else:
                    total = single_interaction_pairs[lr_pair] / \
                        single_interaction_pairs[lr_pair].sum().sum()
                total = total.fillna(0)

        total = total / total.sum().sum()
        total = total.fillna(0)
        
        n_clusters = (
            total.values.diagonal()
            == 0
        ).sum()
        single_interaction_clusters = _lr_cluster_helper(
            result_df_one, single_interaction_pairs, n_clusters
        )
        single_interaction_clusters["cluster"] = (
            single_interaction_clusters["cluster"]
            + max(multi_interaction_clusters["cluster"])
            + 1
        )

    # Combine cluster assignments
    final_cluster_assignments = pd.concat(
        [multi_interaction_clusters, single_interaction_clusters]
    )

    # Create new assays for each cluster in the sample
    for lr_pair in final_cluster_assignments.index:
        cluster_id = final_cluster_assignments["cluster"][lr_pair]
        cluster_assay = f"cluster_{cluster_id}"
        
        # Initialize cluster assay if needed
        if cluster_assay not in sample.assays:
            sample.assays[cluster_assay] = {
                'cci_scores': {},
                'p_values': {}
            }
            
        # Add interaction matrices to cluster assay    
        sample.assays[cluster_assay]['cci_scores'][lr_pair] = \
            sample.assays[assay]['cci_scores'][lr_pair]
        sample.assays[cluster_assay]['p_values'][lr_pair] = \
            sample.assays[assay]['p_values'][lr_pair]

    # Calculate overall interactions for each cluster
    for assay_name in sample.assays:
        if assay_name.startswith('cluster_'):
            sample = sample.calc_overall(assay=assay_name)
    
    return sample


def _lr_cluster_helper(result_df, sample, n_clusters=0, method="KMeans"):
    """
    Args:
        result_df (pd.DataFrame): A DataFrame containing dissimilarity scores for LRs
        sample (dict): A dictionary containing LR matrices.
        n_clusters (int) (optional): The desired number of clusters. If 0, the optimal number is determined using silhouette analysis. Defaults to 0.
        method (str) (optional): The clustering method to use. Defaults to 'KMeans'.

    Returns:
        pd.DataFrame: A DataFrame with the cluster assignments for each sample.
    """

    # Compute distance matrix from disimilarity matrix
    result_df = result_df.astype("float64")
    result_df = result_df.fillna(0)
    distances = pdist(result_df.values, metric="euclidean")
    dist_matrix = squareform(distances)
    dist_matrix = pd.DataFrame(
        pp.MinMaxScaler(feature_range=(0, 100)).fit_transform(
            pd.DataFrame(dist_matrix).T.values
        )
    )

    print("Computing Principal Components of weighted graph ...")
    # Perform PCA on weighted edges of interaction network
    flatten_dfs = [df.to_numpy().flatten() for df in sample.values()]
    flatten_dfs = pd.DataFrame(flatten_dfs)
    flatten_dfs = flatten_dfs.fillna(0)
    scaler = pp.StandardScaler()
    data_scaled = scaler.fit_transform(flatten_dfs)
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)

    # Concatenate PC-1 and PC-2 with distance matrix
    pc_com_dist_matrix = pd.concat([dist_matrix, pd.DataFrame(data_pca)], axis=1)

    print("Performing Clustering and Ranking within clusters...")
    if n_clusters > 0:
        # Number of clusters (adjust as needed)
        n_clusters = n_clusters

        if method == "Hierarchial":
            # Perform hierarchical clustering
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
            clusters = model.fit_predict(pc_com_dist_matrix)
        if method == "KMeans":
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(pc_com_dist_matrix)

    if n_clusters == 0:
        if method == "Hierarchial":
            # Evaluate silhouette score for different numbers of clusters
            silhouette_scores = []
            for n_clusters in range(2, 11):
                clusterer = AgglomerativeClustering(
                    n_clusters=n_clusters, linkage="ward"
                )
                cluster_labels = clusterer.fit_predict(pc_com_dist_matrix)
                silhouette_avg = silhouette_score(pc_com_dist_matrix, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            # Perform hierarchical clustering
            model = AgglomerativeClustering(
                # Add 2 to account for starting with k=2
                n_clusters=np.argmax(silhouette_scores) + 2,
                linkage="ward",
            )  # as indexing starts from 0
            clusters = model.fit_predict(pc_com_dist_matrix)

        if method == "KMeans":
            # Find optimal numer of clusters Davies-Bouldin index
            db_scores = []
            for k in range(2, 11):
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(pc_com_dist_matrix)
                db_scores.append(davies_bouldin_score(pc_com_dist_matrix, labels))
            # Add 2 to account for starting with k=2
            optimal_clusters = np.argmin(db_scores) + 2
            kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
            clusters = kmeans.fit_predict(pc_com_dist_matrix)

    clusters = pd.DataFrame(clusters)
    clusters.index = sample.keys()
    clusters.rename(columns={0: "cluster"}, inplace=True)

    # Rank LRs in each cluster based on increasing dissimilarity
    dist_matrix.columns = list(sample.keys())
    dist_matrix.index = list(sample.keys())
    columns = ["LRs", "cluster"]
    final_clusters = pd.DataFrame(columns=columns)

    for i in range(0, len(set(clusters["cluster"]))):
        clusters_index = list(clusters[clusters["cluster"] == i].index)
        dist_matrix_cluster = dist_matrix[dist_matrix.index.isin(clusters_index)]
        similarity_matrix = 1 / (1 + dist_matrix_cluster)
        linkage_matrix = hierarchy.linkage(similarity_matrix, method="ward")
        sorted_indices = hierarchy.leaves_list(linkage_matrix)
        cluster_df = pd.DataFrame(clusters_index)
        cluster_df = cluster_df.iloc[sorted_indices[::-1]].reset_index(drop=True)
        cluster_df["cluster"] = i
        final_clusters = pd.concat([final_clusters, cluster_df])
        
    final_clusters = final_clusters.iloc[:, 1:]
    final_clusters = final_clusters.rename(columns={0: "LRs"})
    final_clusters = final_clusters[["LRs", "cluster"]]
    final_clusters.set_index("LRs", inplace=True)
    
    return final_clusters

    
def lr_interaction_clustering(
    sample,
    resolution=0.5,
    cluster_palette="Dark2_r",
    cell_type_palette="tab20",
    cell_colors=None,
    spot_size=1.5,
    spatial_plot=True, 
    proportion_plot=True,
    return_adata=False,
    **kwargs
    ):
    """Clustering of spatial LR interaction scores on AnnData objects processed through stLearn.

    Args:
        sample (AnnData): An AnnData object that has been run through stLearn.
        resolution (float) (optional): The resolution to use for the clustering. Defaults to 0.5.
        cluster_palette (str) (optional): Name of matplotlib colormap to use for clusters. Defaults to 'Dark2_r'.
        cell_type_palette (str) (optional): Name of matplotlib colormap to use for cell types. Defaults to 'tab20'.
        cell_colors (dict) (optional): Dictionary mapping cell types to colors. If not provided, colors will be generated from cell_type_palette. Defaults to None.
        spot_size (float) (optional): The size of the spots in the spatial plot. Defaults to 1.5.
        spatial_plot (bool) (optional): Whether to show the spatial plot. Defaults to True.
        proportion_plot (bool) (optional): Whether to show the proportion plot. Defaults to True.
        return_adata (bool) (optional): Whether to return the AnnData object with the clustering results. Defaults to False.
        **kwargs: Additional keyword arguments to pass to the scanpy spatial plot function.
    
    Returns:
        AnnData: An AnnData object with the clustering results.
    """

    if sample.adata is None:
        raise ValueError("No AnnData object found in sample.")
    
    LR = pd.DataFrame(sample.adata.obsm["lr_scores"])
    LR.columns = list(sample.adata.uns["lr_summary"].index)
    LR.index = sample.adata.obs.index

    LR = scanpy.AnnData(LR)
    scanpy.pp.normalize_total(LR, inplace=True)
    scanpy.pp.log1p(LR)
    scanpy.pp.pca(LR)
    scanpy.pp.highly_variable_genes(LR, flavor="seurat", n_top_genes=2000)
    scanpy.pp.neighbors(LR, use_rep="X_pca", n_neighbors=15)
    scanpy.tl.leiden(LR, resolution=resolution)
    
    scanpy.tl.rank_genes_groups(LR, groupby='leiden', method='wilcoxon')
    # scanpy.pl.rank_genes_groups(LR, n_genes=25, sharey=False)
    scanpy.tl.dendrogram(LR, groupby='leiden')
    scanpy.pl.rank_genes_groups_dotplot(LR, n_genes=10, groupby='leiden')
    
    LR.obsm = sample.adata.obsm
    LR.uns = sample.adata.uns
    LR.obs["leiden"] = LR.obs["leiden"].astype("int64")

    scanpy.pp.pca(LR)
    scanpy.pp.highly_variable_genes(LR, flavor="seurat", n_top_genes=2000)
    scanpy.pp.neighbors(LR, use_rep="X_pca", n_neighbors=15)
    scanpy.tl.umap(LR)
    
    sample.adata.obs["LR_Cluster"] = LR.obs["leiden"].astype("str")
    
    if spatial_plot:
        scanpy.pl.spatial(sample.adata, color="LR_Cluster", size=spot_size, 
                     palette=cluster_palette, **kwargs)
        
    if proportion_plot:
        if "cell_type" in sample.adata.uns:
            # make any value less than 0.1 == 0 in adata.uns['cell_type']
            sample.adata.uns['cell_type'] = sample.adata.uns['cell_type'].applymap(
                lambda x: 0 if x < 0.1 else x)
            
            merged_df = sample.adata.uns['cell_type'].merge(
                sample.adata.obs["LR_Cluster"], left_index=True, right_index=True)
            proportions = merged_df.groupby('LR_Cluster').mean()
            proportions = proportions.div(proportions.sum(axis=1), axis=0)
        else:
            proportions = sample.adata.obs.groupby('LR_Cluster')['cell_type'] \
                .value_counts(normalize=True).unstack()

        # Generate colors from cell_type_palette if cell_colors not provided 
        if cell_colors is None:
            unique_types = proportions.columns
            default_colors = plt.get_cmap(cell_type_palette).colors
            cell_colors = {ctype: default_colors[i % len(default_colors)] 
                        for i, ctype in enumerate(unique_types)}

        fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 1]})

        bars = proportions.plot(kind="bar", stacked=True, ax=ax1, 
                                color=proportions.columns.to_series().map(cell_colors))  
        # Use colormap based on cell types
        ax1.set_ylabel("Proportion")
        ax1.legend().remove()  # Remove default legend

        # Create a legend without plot elements
        handles, labels = bars.get_legend_handles_labels()  # Get handles and labels 
        # from the bar plot
        ax2.legend(handles, labels, loc='center')
        ax2.axis('off')

        plt.tight_layout()
        plt.show()

    if return_adata:
        return LR


def run_gsea(
    sample: CCIData = None,
    assay: str = "raw",
    lrs=None,
    organism="human",
    gene_sets=["KEGG_2021_Human", "MSigDB_Hallmark_2020"],
    show_dotplot=False,
    show_barplot=True,
    top_term=5,
    figsize=(3,5),
    use_background=True,
    background_list=None,
    return_results=True
):
    """Runs GSEA analysis on a sample.

    Args:
        sample (CCIData) (optional): The sample to run GSEA on. If not given, lrs are used. Defaults to None.
        assay (str) (optional): The assay to use for the GSEA analysis. Defaults to 'raw'.
        lrs (list) (optional): A list of LR pairs to use for GSEA analysis instead of sample. Defaults to None.
        organism (str) (optional): The organism to use. Defaults to 'human'.
        gene_sets (list) (optional): The gene sets to use for gseapy analysis. Defaults to ['KEGG_2021_Human', 'MSigDB_Hallmark_2020'].
        show_dotplot (bool) (optional): Whether to show the dotplot. Defaults to False.
        show_barplot (bool) (optional): Whether to show the barplot. Defaults to True.
        top_term (int) (optional): The number of top terms to show. Defaults to 5.
        figsize (tuple) (optional): The size of the figure. Defaults to (3,5).
        return_results (bool) (optional): Whether to return the results DataFrame.  Defaults to True.

    Returns:
        pd.DataFrame or None: A DataFrame with the GSEA results if return_results=True, otherwise None.
    """

    gene_list = set()

    if lrs is None:
        if assay not in sample.assays:
            raise ValueError(f"Assay {assay} not found in sample.")
        lrs = sample.assays[assay]["cci_scores"].keys()

    for lr in lrs:
        gene1, gene2 = lr.split("_")
        gene_list.add(gene1)
        gene_list.add(gene2)

    gene_list = list(gene_list)
    
    background = None
    
    if use_background:
        if background_list is not None:
            background = background_list
        else:
            path = os.path.dirname(os.path.realpath(__file__))
            background = pd.read_csv(f"{path}/data/connectomeDB2020_lit.txt", sep="\t", header=None)
            background = list(set(background[0].tolist() + background[1].tolist()))

    enr = gp.enrichr(
        gene_list=gene_list,
        gene_sets=gene_sets,
        organism=organism,
        outdir=None,
        background=background
    )
        
    if show_dotplot:
        try:
            ax = dotplot(
                enr.results,
                column="Adjusted P-value",
                x="Gene_set",
                size=10,
                top_term=top_term,
                figsize=figsize,
                xticklabels_rot=45,
                show_ring=True,
                marker="o",
            )
        except:
            print("Could not plot dotplot.")
        
    if show_barplot:
        colours = list(plt.cm.Dark2.colors)
        colour_dict = {gene_set: colours[i] for i, gene_set in enumerate(gene_sets)}
    
        try:
            ax = barplot(
                enr.results,
                column="Adjusted P-value",
                group="Gene_set",
                size=10,
                top_term=top_term,
                figsize=figsize,
                color=colour_dict
            )
        except:
            print("Could not plot barplot.")

    if return_results:
        return enr.results
    return None


def pathway_subset(
    sample: CCIData,
    assay: str = "raw", 
    gsea_results: pd.DataFrame = None, 
    terms: list = None, 
    strict: bool = False,
    assay_name: str = None
    ) -> CCIData:
    """Subsets a sample to only include interactions between genes in a set of pathways.

    Args:
        sample (CCIData): The sample to subset.
        assay (str) (optional): The assay to use for the subset. Defaults to 'raw'.
        gsea_results (pd.DataFrame): The GSEA results to use to subset the sample.
        terms (list): The terms to subset the sample with.
        strict (bool): Whether to only include interactions between genes in the same pathway.

    Returns:
        CCIdata: The sample with an added assay containing the subsetted interactions.
    """

    genes = []
    grouped_cci_scores = {}
    grouped_p_values = {}
    
    if assay_name is None:
        # make assay name based on terms
        assay_name = "+".join(terms)    
        
    if assay not in sample.assays:
        raise ValueError(f"Assay {assay} not found in sample.")

    for term in terms:
        if term not in gsea_results['Term'].tolist():
            raise ValueError(f"Term {term} not found in GSEA results.")
        filtered_df = gsea_results[gsea_results['Term'] == term]
        gene_list = filtered_df['Genes'].tolist()

        for gene in gene_list:
            genes.extend(gene.lower().split(";"))

    for key in sample.assays[assay]['cci_scores'].keys():
        lig, rec = key.lower().split("_")
        if strict:
            if lig in genes and rec in genes:
                grouped_cci_scores[key] = sample.assays[assay]['cci_scores'][key]
                grouped_p_values[key] = sample.assays[assay]['p_values'][key]
        else:
            if lig in genes or rec in genes:
                grouped_cci_scores[key] = sample.assays[assay]['cci_scores'][key]
                grouped_p_values[key] = sample.assays[assay]['p_values'][key]
                
    print(f"Number of interactions in {assay_name}: {len(grouped_cci_scores)}")

    sample.assays[assay_name] = {}
    sample.assays[assay_name]['cci_scores'] = grouped_cci_scores
    sample.assays[assay_name]['p_values'] = grouped_p_values
    sample = sample.calc_overall(assay_name)
    
    return sample


def add_lr_module_score(sample, lr_list, key_name="score"):
    """Adds a module score to an AnnData object run through stLearn based on the interactions in a list of ligand-receptor pairs.

    Args:
        sample (AnnData): The AnnData object to add the score to. Must be processed through stLearn.
        lr_list (list): The list of ligand-receptor pairs to use.
        key_name (str): The key to use for the score.

    Returns:
        AnnData: The AnnData object with the module score added.
    """

    lr_counts = pd.DataFrame(sample.obsm['lr_sig_scores'])
    lr_counts.index = sample.obs.index
    lr_counts.columns = sample.uns['lr_summary'].index

    adata = ad.AnnData(lr_counts)
    scanpy.tl.score_genes(adata, gene_list=lr_list)
    sample.obs[key_name] = adata.obs['score']

    return sample
