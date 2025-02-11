# MMCCI: Multi-platform, Multi-sample Cell-Cell Interaction Integrative Analysis of Single Cell and Spatial Data

<table align="center">
  <tr>
    <td>
      <b>Package</b>
    </td>
    <td>
      <a href="https://pypi.python.org/pypi/mmcci/">
      <img src="https://img.shields.io/pypi/v/mmcci.svg" alt="PyPI Version">
      </a>
      <a href="https://pepy.tech/project/mmcci">
      <img src="https://static.pepy.tech/personalized-badge/mmcci?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads"
        alt="PyPI downloads">
    </td>
  </tr>
  <tr>
    <td>
     <b>Paper</b>
    </td>
    <td>
      <a href="https://doi.org/10.1101/2023.05.14.540710"><img src="https://zenodo.org/badge/DOI/10.1101/2023.05.14.540710.svg"
        alt="DOI"></a>
    </td>
  </tr>
  <tr>
    <td>
      <b>License</b>
    </td>
    <td>
      <a href="https://github.com/GenomicsMachineLearning/MMCCI/blob/main/LICENSE.txt"><img src="https://img.shields.io/badge/License-BSD-blue.svg"
        alt="LICENSE"></a>
    </td>
  </tr>
</table>
        
**MMCCI** is a fast and lightweight Python package for integrating and visualizing CCI networks within and between multiple platforms at the level of the individual LR pair. It works on **scRNA-seq** and **spatial transcriptomics** data samples that have been processed through the following CCI algorithms:
1. stLearn
2. CellChat
3. CellPhoneDB
4. NATMI
5. Squidpy

## Getting Started

### Installation

MMCCI can be installed with `pip`

```
pip install mmcci
```


### Documentation

Documentation and Tutorials are available here and we are commited to maintaining the software and addressing issues raised by users.

- There is a brain aging tutorial notebook [here](examples/brain_aging_example.ipynb)
- There is a melanoma tutorial notebook [here](examples/melanoma_example.ipynb)
- To understand how to load CCI results from different tools, look at this notebook [here](examples/loading_CCI_results.ipynb)

## CCI Integration

MMCCI allows users to integrate multiple CCI results together, both:
1. Samples from a single platform (eg. Visium)
2. Samples from multiple platforms (eg. Visium, Xenium and CosMX)

![Integration and Analysis Method](docs/images/analyses_pipeline.png)

## CCI Analysis

MMCCI provides multiple useful analyses that can be run on the integrated networks or from a single sample:
1. Network comparison between groups with permutation testing
2. CLustering of LR pairs with similar networks
3. Clustering of spots/cells with similar interaction scores
4. Sender-receiver LR querying
5. GSEA pathway analysis

### Pipeline Diagram

![MMCCI Pipeline](docs/images/pipeline.png)

## Citing MMCCI

If you have used MMCCI in your research, please consider citing us: 
```

```

