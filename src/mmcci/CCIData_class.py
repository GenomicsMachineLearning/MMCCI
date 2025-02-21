import pandas as pd
import numpy as np
import anndata as ad
import pickle
from typing import Dict, List, Optional, Union
from copy import deepcopy
import json
from tqdm import tqdm


class CCIData:
    """
    Class to store and manage Cell-Cell Interaction (CCI) data
    
    Attributes:
        metadata (Dict): Metadata for sample
        n_spots (int): Number of spots in the sample
        cci_scores (Dict): CCI score dataframe for each LR pair
        p_values (Dict): P-values dataframe for each LR pair
        adata (AnnData): AnnData object
        networks (Dict): Calculated CCI networks
        other_metadata (Dict): Other metadata
        assays (Dict): Assays for the sample
    """
    
    def __init__(
        self, 
        cci_scores: Dict = None, 
        p_values: Dict = None,
        n_spots: int = None, 
        platform: str = None,
        network: pd.DataFrame = None,
        network_p_values: pd.DataFrame = None,
        adata: ad.AnnData = None,
        other_metadata: Dict = None,
        assays: Dict = None
        ):
        self.metadata = {}
        self.assays = {}
        self.adata = adata
        self.assays['raw'] = {}
        
        self.metadata['n_spots'] = n_spots
        self.metadata['platform'] = platform
        
        if other_metadata:
            self.metadata.update(other_metadata)

        if cci_scores is not None:
            self.assays['raw']['cci_scores'] = cci_scores

            if p_values is not None:
                self.assays['raw']['p_values'] = p_values
                
        if network is not None:
            self.assays['raw']['network'] = network
            
            if network_p_values is not None:
                self.assays['raw']['network_p_values'] = network_p_values
                
        if assays:
            self.assays = assays
                
    
    def __repr__(self):
        """Print assays (with number of LR pairs) and metadata"""
        
        assays = {}
        for assay in self.assays.keys():
            if 'cci_scores' in self.assays[assay].keys():
                assays[assay] = f"{len(self.assays[assay]['cci_scores'])} LR pairs"
            elif 'network' in self.assays[assay].keys():
                assays[assay] = "network"
            else:
                assays[assay] = "none"
            
        if self.adata is not None:
            return f"CCIData object with assays: {assays}, metadata: {self.metadata}, \
                and AnnData object"
        
        return f"CCIData object with assays: {assays} and metadata: {self.metadata}"
    
    
    def __str__(self):
        """Print assays (with number of LR pairs) and metadata"""
        
        return self.__repr__()
        
        
    def get_sample_metadata(self, sample_id: str) -> Dict:
        """
        Get metadata for a sample
        
        Args:
            sample_id: Sample ID
        
        Returns:
            Metadata for the sample
        """
        return self.metadata.get(sample_id)
    
    
    def get_sample_n_spots(self, sample_id: str) -> Optional[int]:
        """
        Get number of spots for a sample
        
        Args:
            sample_id: Sample ID
        
        Returns:
            Number of spots for the sample
        """
        return self.n_spots.get(sample_id)
    
    
    def get_sample_cci_scores(self, sample_id: str) -> Optional[pd.DataFrame]:
        """
        Get CCI scores for a sample
        
        Args:
            sample_id: Sample ID
        
        Returns:
            CCI scores for the sample
        """
        return self.cci_scores.get(sample_id)
    
    
    def get_sample_p_values(self, sample_id: str) -> Optional[pd.DataFrame]:
        """
        Get p-values for a sample
        
        Args:
            sample_id: Sample ID
        
        Returns:
            P-values for the sample
        """
        return self.p_values.get(sample_id)
    
    
    def get_adata(self) -> Optional[ad.AnnData]:
        """
        Get AnnData object
        
        Returns:
            AnnData object
        """
        return self.adata
    
    
    def get_cell_types(self, assay: str = "raw") -> List[str]:
        """
        Get cell types in a sample
        
        Args:
            assay: Assay to get cell types from
        
        Returns:
            List of cell types in the sample
        """
        cell_types = []
        
        if 'cci_scores' in self.assays[assay].keys():
            for dfs in self.assays[assay]['cci_scores'].values():
                cell_types.extend(dfs.index)
        elif 'network' in self.assays[assay].keys():
            cell_types.extend(self.assays[assay]['network'].index)
        else:
            raise ValueError("No cell types found in sample.")
            
        return list(set(cell_types))
    
    
    def copy(self) -> 'CCIData':
        """
        Create a copy of the CCIData object
        
        Returns:
            Copy of the CCIData object
        """
        cci_data = CCIData(
            assays=deepcopy(self.assays),
            other_metadata=deepcopy(self.metadata),
        )
        
        return cci_data
    
    
    def rename_cell_types(self, 
                         replacements: Dict[str, str],
                         assay = None) -> 'CCIData':
        """Renames cell types in a CCIData.

        Args:
            replacements (dict): A dictionary of replacements, where the keys are the old cell type names and the values are the new cell type names.
            assay (str): The assay to rename the cell types in. If None, all assays are renamed.

        Returns:
            CCIData: A new CCIData object with the cell types renamed.
        """

        renamed_cci_data = self.copy()
        
        if assay is not None:
            for key in renamed_cci_data.assays[assay].keys():
                if key == 'network' or key == 'overall':
                    renamed_cci_data.assays[assay][key].rename(
                        index=replacements,
                        columns=replacements,
                        inplace=True)
                    
                else:
                    for lr_pair in renamed_cci_data.assays[assay][key].keys():
                        renamed_cci_data.assays[assay][key][lr_pair].rename(
                        index=replacements,
                        columns=replacements,
                        inplace=True)
            
        else:
            for assay in renamed_cci_data.assays.keys():
                for key in renamed_cci_data.assays[assay].keys():
                    if key == 'network' or key == 'overall':
                        renamed_cci_data.assays[assay][key].rename(
                            index=replacements,
                            columns=replacements,
                            inplace=True)
                        
                    else:
                        for lr_pair in renamed_cci_data.assays[assay][key].keys():
                            renamed_cci_data.assays[assay][key][lr_pair].rename(
                            index=replacements,
                            columns=replacements,
                            inplace=True)

        return renamed_cci_data
    
    
    def merge_cell_types(self,
                         cell_types: List[str],
                         new_cell_type: str,
                         assay = None) -> 'CCIData':
        """Merges cell types in a CCIData.
        
        Args:
            cell_types (list): A list of cell types to merge.
            new_cell_type (str): The name of the new cell type after merging.
            assay (str): The assay to merge the cell types in. If None, all assays are merged.
                
        Returns:
            CCIData: A new CCIData object with the cell types merged.
        """
        merged_cci_data = self.copy()
        assays = merged_cci_data.assays.keys()
        
        if assay is not None:
            assays = [assay]
            
        for assay in assays:
            for key in merged_cci_data.assays[assay].keys():
                if key == 'network' or key == 'overall':
                    df = merged_cci_data.assays[assay][key]
                    # Sum the rows
                    row_sums = df.loc[cell_types].sum()
                    # Sum the columns 
                    col_sums = df[cell_types].sum()
                    # Drop original cell types
                    df = df.drop(cell_types, axis=0)
                    df = df.drop(cell_types, axis=1) 
                    # Add new merged cell type
                    df.loc[new_cell_type] = row_sums
                    df[new_cell_type] = col_sums
                    merged_cci_data.assays[assay][key] = df
                elif key == 'cci_scores':
                    for lr_pair in merged_cci_data.assays[assay][key].keys():
                        df = merged_cci_data.assays[assay][key][lr_pair]
                        # Sum the rows
                        row_sums = df.loc[cell_types].sum()
                        # Sum the columns
                        col_sums = df[cell_types].sum()
                        # Drop original cell types
                        df = df.drop(cell_types, axis=0)
                        df = df.drop(cell_types, axis=1)
                        # Add new merged cell type
                        df.loc[new_cell_type] = row_sums
                        df[new_cell_type] = col_sums
                        merged_cci_data.assays[assay][key][lr_pair] = df
                elif key == 'p_values':
                    for lr_pair in merged_cci_data.assays[assay][key].keys():
                        df = merged_cci_data.assays[assay][key][lr_pair]
                        # Sum the rows
                        row_min = df.loc[cell_types].min()
                        # Sum the columns
                        col_min = df[cell_types].min()
                        # Drop original cell types
                        df = df.drop(cell_types, axis=0)
                        df = df.drop(cell_types, axis=1)
                        # Add new merged cell type
                        df.loc[new_cell_type] = row_min
                        df[new_cell_type] = col_min
        
        return merged_cci_data
    
    
    def subset_lrs(self,
                   lr_pairs: List[str],
                   assay = "raw",
                   new_assay = "subset") -> 'CCIData':
        """Subsets the LR pairs in a CCIData object.
    
        Args:
            lr_pairs (list): A list of LR pairs to include in the subsetted data.
            assay (str): The assay to subset the LR pairs in. If None, all assays are subsetted.
            new_assay (str): The name of the new assay after subsetting.
            
        Returns:
            CCIData: A new CCIData object with the LR pairs subsetted.
        """
        
        subsetted_cci_data = self.copy()
        subsetted_cci_data.assays[new_assay] = {}
        subsetted_cci_data.assays[new_assay]['cci_scores'] = \
            {lr: self.assays[assay]['cci_scores'][lr] for lr in lr_pairs}
        subsetted_cci_data.assays[new_assay]['p_values'] = \
            {lr: self.assays[assay]['p_values'][lr] for lr in lr_pairs}
        
        return subsetted_cci_data
    
    
    def scale(self, 
              scale_factor: float,
              assay = "raw",
              new_assay = "scaled") -> 'CCIData':
        """Scales the CCI scores in a CCIData object.

        Args:
            scale_factor (float): The factor to scale the CCI scores by.
            assay (str): The assay to scale the CCI scores in. If None, all assays are scaled.
            new_assay (str): The name of the new assay after scaling.

        Returns:
            CCIData: A new CCIData object with the CCI scores scaled.
        """

        scaled_cci_data = self.copy()
        scaled_cci_data.assays[new_assay] = self.assays[assay].copy()
        
        for lr_pair in self.assays[assay]['cci_scores'].keys():
            scaled_cci_data.assays[new_assay]['cci_scores'][lr_pair] = \
                self.assays[assay]['cci_scores'][lr_pair] * scale_factor
        
        return scaled_cci_data
    
    
    def scale_by_nspots(self, 
                        assay = "raw",
                        new_assay = "scaled") -> 'CCIData':
        """Scales the CCI scores in a CCIData object by the number of spots.

        Args:
            assay (str): The assay to scale the CCI scores in. If None, all assays are scaled.
            new_assay (str): The name of the new assay after scaling.

        Returns:
            CCIData: A new CCIData object with the CCI scores scaled.
        """

        if 'n_spots' not in self.metadata.keys():
            raise ValueError("'n_spots' not found in metadata.")
        
        scale_factor = 1e6 / self.metadata['n_spots']
        return self.scale(scale_factor, assay, new_assay)
    
    
    def filter_by_p_vals(self, 
                          cutoff: float = 0.05, 
                          assay = "raw",
                          new_assay = "filtered") -> 'CCIData':
        """Filters the CCI scores in a CCIData object by p-value.

        Args:
            cutoff (float): The p-value cutoff to filter the CCI scores by.
            assay (str): The assay to filter the CCI scores in. If None, all assays are filtered.
            new_assay (str): The name of the new assay after filtering.

        Returns:
            CCIData: A new CCIData object with the CCI scores filtered.
        """

        filtered_cci_data = self.copy()
        filtered_cci_data.assays[new_assay] = self.assays[assay].copy()
        
        for lr_pair, df in self.assays[assay]['cci_scores'].items():
                for i, row in df.iterrows():
                    for j in row.index:
                        p_vals = self.assays[assay]['p_values'][lr_pair]
                        if p_vals.loc[i, j] > cutoff:
                            df.loc[i, j] = 0
                filtered_cci_data.assays[new_assay]['cci_scores'][lr_pair] = df
                
        return filtered_cci_data
    
    
    def calc_overall(self, 
                     assay = "raw", 
                     name = "overall", 
                     normalize = True) -> 'CCIData':
        """Calculates the overall CCI scores in a CCIData object.

        Args:
            assay (str): The assay to calculate the overall CCI scores in. If None, all assays are calculated
            name (str): The name of the network after calculating the overall CCI scores.
            normalize (bool): If True, normalize each LR CCI score network. Defaults to True.
            
        Returns:
            CCIData: A new CCIData object with the overall CCI scores calculated.
        """
        
        overall_cci_data = self.copy()
        
        sample = overall_cci_data.assays[assay]['cci_scores']
        
        total = None
        for lr_pair in sample.keys():
            if sample[lr_pair].sum().sum() > 0:
                if total is not None:
                    if normalize:
                        total = total + sample[lr_pair] / sample[lr_pair].sum().sum()
                    else:
                        total = total + sample[lr_pair]
                    total = total.fillna(0)
                else:
                    if normalize:
                        total = sample[lr_pair] / sample[lr_pair].sum().sum()
                    else:
                        total = sample[lr_pair]
                    total = total.fillna(0)

        total = total / total.sum().sum()
        total = total.fillna(0)

        overall_cci_data.assays[assay][name] = total
        
        return overall_cci_data
    
    
    def get_lr_proportions(
        self,
        sender: str = None,
        reciever: str = None,
        assay: str = "raw",
        key: str = "cci_scores"
        ) -> dict:
        """Calculates the proportion of each LR pair in a sample for a specific cell type sender and receiver.

        Args:
            sender (str): The sender cell type.
            reciever (str): The receiver cell type.
            assay (str) (optional): The assay to use. Defaults to 'raw'.
            key (str) (optional): The key to use. Defaults to 'cci_scores'.

        Returns:
            dict: A list of LR pairs and proportion of its weighting.
        """

        if assay not in self.assays:
            raise ValueError(f"Assay {assay} not found in sample.")
        
        if key not in self.assays[assay]:
            raise ValueError(f"Key {key} not found in sample.")
        
        if sender is None and reciever is None:
            raise ValueError("Please provide a sender or receiver cell type.")
        
        lr_pairs = self.assays[assay][key].keys()
        for lr in lr_pairs:
            if type(self.assays[assay][key][lr]) != pd.DataFrame:
                del self.assays[assay][key][lr]

        if sender is not None:
            subset = {
                lr: df.loc[[sender]] for lr, df in self.assays[assay][key].items()
                if sender in df.index
                }
            
        if reciever is not None:
            subset = {
                lr: df[[reciever]] for lr, df in self.assays[assay][key].items()
                if reciever in df.columns
                }
            
        subset = {
            key: df
            for key, df in subset.items()
            if not df.map(lambda x: x == 0).all().all()
        }

        subset = {
            key: df
            for key, df in subset.items()
            if not df.map(lambda x: x == 0).all().all()
        }

        lr_props = {}
        total = 0
        
        if sender is not None and reciever is not None:
            for lr_pair in set(subset.keys()):
                score = subset[lr_pair].at[sender, reciever]
                total += score

            for lr_pair in set(subset.keys()):
                lr_props[lr_pair] = subset[lr_pair].at[sender, reciever] / total
                
        if sender is None:
            # sum across all sender cell types
            for lr_pair in set(subset.keys()):
                score = subset[lr_pair].loc[:, reciever].sum()
                total += score
                
            for lr_pair in set(subset.keys()):
                lr_props[lr_pair] = subset[lr_pair].loc[:, reciever].sum() / total
                
        if reciever is None:
            # sum across all receiver cell types
            for lr_pair in set(subset.keys()):
                score = subset[lr_pair].loc[sender, :].sum()
                total += score
                
            for lr_pair in set(subset.keys()):
                lr_props[lr_pair] = subset[lr_pair].loc[sender, :].sum() / total

        lr_props = dict(sorted(lr_props.items(), key=lambda item: item[1], 
                               reverse=True))

        return lr_props


    def get_p_vals(
        self,
        assay: str = "raw",
        key: str = "p_values",
        sender: str = None,
        reciever: str = None
        ) -> dict:
        """Returns the p-values for each LR pair in a sample for a specific sender and receiver cell type.
        
        Args:
            assay (str) (optional): The assay to use. Defaults to 'raw'.
            key (str) (optional): The key to use. Defaults to 'p_values'.
            sender (str) (optional): The sender cell type. Defaults to None.
            reciever (str) (optional): The receiver cell type. Defaults to None.
            
        Returns:
            dict: A dictionary of LR pairs and their p-values.
        """
        
        if assay not in self.assays:
            raise ValueError(f"Assay {assay} not found in sample.")
        
        if key not in self.assays[assay]:
            raise ValueError(f"Key {key} not found in sample.")
        
        if sender is None or reciever is None:
            raise ValueError("Please provide a sender and receiver cell type.")
        
        lr_pairs = self.assays[assay][key].keys()
        for lr in lr_pairs:
            if type(self.assays[assay][key][lr]) != pd.DataFrame:
                del self.assays[assay][key][lr]
                
        subset = {
            lr: df.loc[[sender], [reciever]] for lr, df in 
            self.assays[assay][key].items()
            if sender in df.index and reciever in df.columns
            }
        
        subset = {
            key: df
            for key, df in subset.items()
            if not df.map(lambda x: x == 0).all().all()
        }
        
        subset = {
            key: df
            for key, df in subset.items()
            if not df.map(lambda x: x == 0).all().all()
        }
        
        p_vals = {}
        for lr_pair in set(subset.keys()):
            p_vals[lr_pair] = subset[lr_pair].at[sender, reciever]
            
        return p_vals


    def save(self, path: str):
        """Saves CCIData object to JSON or pkl file.
        
        Args:
            path (str): Path to save the JSON or pkl file
        """
        if not path.endswith('.json'):
            path += '.json'
            
        if path.endswith('.json'):
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f)
                
        if path.endswith('.pkl'):
            with open(path, 'wb') as f:
                pickle.dump(self, f)

            
    def to_dict(self) -> dict:
        """Convert CCIData object to a JSON-serializable dictionary.
        
        Returns:
            dict: Dictionary representation of the CCIData object
        """
        data_dict = {
            'metadata': self.metadata,
            'assays': {}
        }
        
        # Convert assays
        for assay_name, assay in self.assays.items():
            data_dict['assays'][assay_name] = {}
            for key, value in assay.items():
                if key in ['cci_scores', 'p_values']:
                    # Convert dict of DataFrames
                    data_dict['assays'][assay_name][key] = {
                        k: v.to_dict() for k,v in value.items()
                    }
                elif isinstance(value, pd.DataFrame):
                    # Convert single DataFrame
                    data_dict['assays'][assay_name][key] = value.to_dict()
                else:
                    data_dict['assays'][assay_name][key] = value
                    
        return data_dict


    def create_pathway_assay(
        self,
        assay: str = "raw",
        gsea_results: pd.DataFrame = None,
        strict: bool = True,
        cutoff: float = 0.05,
        assay_name = "pathway"
        ) -> 'CCIData':
        """Creates a pathway assay from GSEA results.
        
        Args:
            assay (str): The assay to use. Defaults to 'raw'.
            gsea_results (pd.DataFrame): GSEA results. Defaults to None.
            strict (bool): If True, both ligand and receptor must be in the gene list. If False, only one must be in the gene list. Defaults to True.
            cutoff (float): The p-value cutoff to filter the GSEA results by. Defaults to 0.05.
            assay_name (str): The name of the new assay. Defaults to 'pathway'.
            
        Returns:
            CCIData: A new CCIData object with the pathway assay.
        """
        
        copy = self.copy()
        grouped_cci_scores = {}

        with tqdm(total=len(gsea_results), desc="Converting to pathways") as pbar:
            for term in gsea_results['Term']:
                
                filtered_df = gsea_results[gsea_results['Term'] == term]
                if filtered_df['Adjusted P-value'].values[0] > cutoff:
                    tqdm.update(pbar, 1)
                    continue
                
                gene_list = filtered_df['Genes'].tolist()
                genes = []
                for gene in gene_list:
                    genes.extend(gene.lower().split(";"))
                    
                cci_scores = []
                        
                for key in self.assays[assay]['cci_scores'].keys():
                    lig, rec = key.lower().split("_")
                    if strict:
                        if lig in genes and rec in genes:
                            cci_scores.append(self.assays[assay]['cci_scores'][key])
                    else:
                        if lig in genes or rec in genes:
                            cci_scores.append(self.assays[assay]['cci_scores'][key])
                            
                total = None      
                for df in cci_scores:
                    if df.sum().sum() > 0:
                        if total is not None:
                            total = total + df
                            total = total.fillna(0)
                        else:
                            total = df
                            total = total.fillna(0)
                            
                if total is None:
                    tqdm.update(pbar, 1)
                    continue
                
                total = total / total.sum().sum()
                total = total.fillna(0)

                grouped_cci_scores[term] = total
                tqdm.update(pbar, 1)
                        
        copy.assays[assay_name] = {'cci_scores': grouped_cci_scores}
        
        return copy
