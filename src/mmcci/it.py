import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import copy

from scipy import stats
from tqdm import tqdm

from . import sc, tl
from .CCIData_class import CCIData


def get_lr_pairs(
    samples: list,
    assay = "raw",
    method = ">=50%"
    ) -> list:
    """Identifies the LR pairs present in a list of samples according to the given method.

    Args:
        samples (list): A list of CCIData objects.
        assay (str) (optional): The assay or assays to use for identifying LR pairs. Defaults to "raw".
        method (str) (optional): The method to use for identifying LR pairs. Options are "all", ">=50%", ">50%", and "any". Defaults to ">=50%".

    Returns:
        list: A list of LR pairs that are present in a majority of samples
    """

    lr_pairs_counts = {}
    lr_pairs = []
    assays = []
    
    if type(assay) == str:
        assays = [assay for i in range(len(samples))]
    elif len(assay) != len(samples):
        raise ValueError("Assay list must be the same length as the samples list.")
    else:
        assays = assay

    for i in range(len(samples)):
        for lr_pair, matrix in samples[i].assays[assays[i]]['cci_scores'].items():
            if sum(sum(matrix.values)) != 0:
                lr_pairs_counts[lr_pair] = lr_pairs_counts.setdefault(lr_pair, 0) + 1

    for lr_pair, count in lr_pairs_counts.items():
        if method == "all":
            if count == len(samples):
                lr_pairs.append(lr_pair)
        elif method == ">=50%":
            if count >= len(samples) / 2:
                lr_pairs.append(lr_pair)
        elif method == ">50%":
            if count > len(samples) / 2:
                lr_pairs.append(lr_pair)
        elif method == "any":
            lr_pairs.append(lr_pair)
        else:
            raise ValueError("Method must be 'all', '>=50%', '>50%', or 'any'.")

    return lr_pairs


def calc_scale_factors(
    samples, 
    method="mean", 
    assay="raw", 
    group_key="platform"
    ) -> dict:
    """Calculates the scale factors for normalizing matrices between different platforms.
    
    Args:
        samples (list): A list of CCIData objects.
        method (str) (optional): The method to use for calculating scale factors.
        assay (str) (optional): The assay or assays to use for calculating scale factors. Defaults to "raw".
        group_key (str) (optional): The key to use for grouping samples. Defaults to "platform".
        
    Returns:
        dict: A dictionary where keys are LR pairs and values are the scale factors.
    """
    
    scale_factors = {}
    assays = []
    
    if type(assay) == str:
        assays = [assay for i in range(len(samples))]
    elif len(assay) != len(samples):
        raise ValueError("Assay list must be the same length as the samples list.")
    else:
        assays = assay
    
    # get all the unique group keys
    group_keys = set([sample.metadata[group_key] for sample in samples])
    group_samples = {group_key: [] for group_key in group_keys}
    
    for sample in samples:
        group_samples[sample.metadata[group_key]].append(sample)
    

    # get the scale factors for each group
    for group, sample_list in group_samples.items():
        total_counts = 0
        lr_pair_count = 0
        for i in range(len(sample_list)):
            for lr_pair, df in sample_list[i].assays[assays[i]]['cci_scores'].items():
                lr_pair_count += 1
                if method == "mean":
                    total_counts += df.mean().mean()
                elif method == "max":
                    total_counts += df.max().max()
                elif method == "median":
                    total_counts += df.median().median()
                elif method == "sum":
                    total_counts += df.sum().sum()
                else:
                    raise ValueError("Invalid method option.")

        total_counts = total_counts / lr_pair_count
        scale_factors[group] = total_counts
        
    # divide each group by the max scale factor
    max_scale_factor = max(scale_factors.values())
    for group, scale_factor in scale_factors.items():
        scale_factors[group] = max_scale_factor / scale_factor
        
    return scale_factors


def lr_integration(
    samples,
    method=">=50%",
    sum=False,
    strict=False,
    assay="raw",
    integrate_pvals=True,
    p_val_method="stouffer",
    metadata=None
    ) -> CCIData:
    """Integrates a list of samples into a single sample per lr pair.
    
    Args:
        samples (list): A list of CCIData objects.
        method (str) (optional): The method to use for identifying LR pairs. Options are "all", ">=50%", ">50%", and "any". Defaults to ">=50%".
        sum (bool) (optional): Whether to sum instead of multiply the matrices. Defaults to False.
        strict (bool) (optional): If True, only interactions where more than 50% of the values are non-zero will be multiplied. Defaults to False.
        assay (str) (optional): The assay or assays to use for integrating samples. Defaults to "raw".
        integrate_pvals (bool) (optional): Whether to integrate p-values (if possible). Defaults to True.
        p_val_method (str) (optional): The method to use for combining p-values. Options are "stouffer" and "fisher". Defaults to "stouffer".
        metadata (dict) (optional): Additional metadata to include in the integrated sample. Defaults to None.
        
    Returns:
        CCIData: The integrated sample.
    """
    
    if len(samples) == 0:
        raise ValueError("No samples provided.")
    
    assays = []
    
    if type(assay) == str:
        assays = [assay for i in range(len(samples))]
    elif len(assay) != len(samples):
        raise ValueError("Assay list must be the same length as the samples list.")
    else:
        assays = assay
    
    for i in range(len(samples)):
        if samples[i].assays[assays[i]] is None:
            raise ValueError("Sample does not have the specified assay.")
        
        if integrate_pvals:
            if 'p_values' not in samples[i].assays[assays[i]]:
                integrate_pvals = False
                print("No p-values found. Skipping p-value integration.")
                
    integrated_cci_scores = {}
    lr_dfs = {}

    if len(samples) >= 2:
        lr_pairs = sorted(get_lr_pairs(samples, assay=assays, method=method))
    else:
        raise ValueError("Integration needs at least two samples")

    for i in range(len(lr_pairs)):
        lr = lr_pairs[i]

        for j in range(len(samples)):
            if lr in samples[j].assays[assays[j]]['cci_scores']:
                if lr in lr_dfs:
                    lr_dfs[lr].append(samples[j].assays[assays[j]]['cci_scores'][lr])
                else:
                    lr_dfs[lr] = [samples[j].assays[assays[j]]['cci_scores'][lr]]

    with tqdm(total=len(lr_pairs), desc="Integrating LR CCI scores") as pbar:
        for lr, dfs in lr_dfs.items():
            if len(dfs) == 2:
                if sum:
                    dfs[0], dfs[1] = tl.align_dataframes(dfs[0], dfs[1])
                    integrated_cci_scores[lr] = dfs[0] + dfs[1]
                    integrated_cci_scores[lr] = integrated_cci_scores[lr] / 2
                    integrated_cci_scores[lr] = integrated_cci_scores[lr].fillna(0)
                else:
                    integrated_cci_scores[lr] = (dfs[0] * dfs[1]).fillna(0)
                    integrated_cci_scores[lr] = \
                        np.sqrt(integrated_cci_scores[lr]).fillna(0)
                        
            elif len(dfs) > 2:
                if sum:
                    integrated_cci_scores[lr] = dfs[0]
                    for i in range(1, len(dfs)):
                        integrated_cci_scores[lr], dfs[i] = tl.align_dataframes(
                            integrated_cci_scores[lr], dfs[i])
                        integrated_cci_scores[lr] = \
                            integrated_cci_scores[lr] + dfs[i]
                    integrated_cci_scores[lr] = \
                        integrated_cci_scores[lr] / len(dfs)
                    integrated_cci_scores[lr] = integrated_cci_scores[lr].fillna(0)
                else:
                    integrated_cci_scores[lr] = sc.multiply_non_zero_values(dfs, 
                                                                 strict=strict)
                    
            else:
                integrated_cci_scores[lr] = dfs[0]
            tqdm.update(pbar, 1)

    if integrate_pvals:
        integrated_p_values = {}
        lr_dfs = {}
        
        lr_pairs = set()
        for i in range(len(samples)):
            lr_pairs.update(samples[i].assays[assays[j]]['p_values'].keys())
        lr_pairs = list(lr_pairs)

        for i in range(len(lr_pairs)):
            lr = lr_pairs[i]

            for j in range(len(samples)):
                if lr in samples[j].assays[assays[j]]['p_values']:
                    if lr in lr_dfs:
                        lr_dfs[lr].append(samples[j].assays[assays[j]]['p_values'][lr])
                    else:
                        lr_dfs[lr] = [samples[j].assays[assays[j]]['p_values'][lr]]

        with tqdm(total=len(lr_dfs), desc="Integrating p values") as pbar:
            for lr, dfs in lr_dfs.items():
                integrated_p_values[lr] = \
                    _correct_pvals_matrix(dfs, method=p_val_method)
                pbar.update(1)
                
    if metadata is None:
        metadata = {'platform': 'integrated'}
        
    integrated = CCIData(
        cci_scores=integrated_cci_scores,
        p_values=integrated_p_values,
        other_metadata=metadata
    )
    
    return integrated


def integrate_networks(
    samples,
    sum=False,
    strict=False,
    assay="raw",
    network_key="network",
    integrate_pvals=False,
    p_val_method="stouffer",
    p_val_key="network_p_values",
    ) -> CCIData:
    """Integrates a list of samples that are single networks into a single network.
    
    Args:
        samples (list): A list of CCIData objects.
        sum (bool) (optional): Whether to sum instead of multiply the matrices. Defaults to False.
        strict (bool) (optional): If True, only interactions where more than 50% of the  values are non-zero will be multiplied. Defaults to False.
        assay (str) (optional): The assay or assays to use for integrating samples.  Defaults to "raw".
        network_key (str) (optional): The key or keys to use for identifying the  network. Defaults to "network".
        integrate_pvals (bool) (optional): Whether to integrate p-values (if possible). Defaults to False.
        p_val_method (str) (optional): The method to use for combining p-values. Options are "stouffer" and "fisher". Defaults to "stouffer".
        p_val_key (str) (optional): The key to use for identifying p-values. Defaults to "network_p_values".
        
    Returns:
        CCIData: The integrated sample.
    """
    
    if len(samples) < 2:
        raise ValueError("Integration needs at least two samples.")
    
    networks = []
    integrated_network = None
    integrated_p_values = None
    assays = []
    network_keys = []
    
    if type(assay) == str:
        assays = [assay for i in range(len(samples))]
    elif len(assay) != len(samples):
        raise ValueError("Assay list must be the same length as the samples list.")
    else:
        assays = assay
        
    if type(network_key) == str:
        network_keys = [network_key for i in range(len(samples))]
    elif len(network_key) != len(samples):
        raise ValueError("Network key list must be the same length as the samples list.")
    else:
        network_keys = network_key
    
    for i in range(len(samples)):
        if samples[i].assays[assays[i]] is None:
            raise ValueError("Sample does not have the specified assay.")
        
        if integrate_pvals:
            if p_val_key is None:
                raise ValueError("No p-value key provided.")
            
            if p_val_key not in samples[i].assays[assays[i]]:
                raise ValueError("Sample does not have the specified p-value key.")
            
        networks.append(samples[i].assays[assays[i]][network_keys[i]])
            
    if len(networks) == 2:
        if sum:
            networks[0], networks[1] = tl.align_dataframes(networks[0], networks[1])
            integrated_network = networks[0] + networks[1]
            integrated_network = integrated_network / 2
            integrated_network = integrated_network.fillna(0)
        else:
            integrated_network = (networks[0] * networks[1]).fillna(0)
            integrated_network = np.sqrt(integrated_network).fillna(0)
            
    elif len(networks) > 2:
        if sum:
            integrated_network = networks[0]
            for i in range(1, len(networks)):
                integrated_network, networks[i] = tl.align_dataframes(
                    integrated_network, networks[i])
                integrated_network = integrated_network + networks[i]
            integrated_network = integrated_network / len(networks)
            integrated_network = integrated_network.fillna(0)
        else:
            integrated_network = sc.multiply_non_zero_values(networks, strict=strict)

    if integrate_pvals:
        p_values = []
        
        for i in range(len(samples)):
            p_values.append(samples[i].assays[assays[i]][p_val_key])
        
        integrated_p_values = _correct_pvals_matrix(p_values, method=p_val_method)
        
    integrated = CCIData(
        network=integrated_network,
        network_p_values=integrated_p_values
    )

    return integrated


def _correct_pvals_matrix(
    dfs, 
    method="stouffer"
    ):
    
    result_df = dfs[0]

    for i in range(len(dfs)):
        dfs[i], result_df = tl.align_dataframes(
            dfs[i], result_df, fill_value=np.NaN)

    for i in range(len(dfs)):
        dfs[i], result_df = tl.align_dataframes(
            dfs[i], result_df, fill_value=np.NaN)

    result_df = result_df.astype(np.float64)
    for i, row in result_df.iterrows():
        for j in row.index:
            values = [df.loc[i, j] for df in dfs]
            values = [
                0.00000000001 if x == 0 else (
                    0.9999999999 if x == 1 else x) for x in values]
            values = [x for x in values if not np.isnan(x)]
            result_df.loc[i, j] = stats.combine_pvalues(values, method=method)[1]

    return result_df
