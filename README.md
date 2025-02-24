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
      <b>Documentation</b>
    </td>
    <td>
      <a href="https://mmcci.readthedocs.io/en/latest/">
      <img src="https://readthedocs.org/projects/mmcci/badge/?version=latest" alt="Documentation Status">
      </a>
    </td>
  </tr>
  <tr>
    <td>
     <b>Paper</b>
    </td>
    <td>
      <a href="https://www.biorxiv.org/content/10.1101/2024.02.28.582639v3"><img src="https://zenodo.org/badge/DOI/10.1101/2023.05.14.540710.svg"
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
        
**MMCCI** is a fast and lightweight Python package for integrating and visualizing CCI networks. It works on **scRNA-seq** and **spatial transcriptomics** data samples that have been processed through CCI algorithms including [stLearn](https://stlearn.readthedocs.io/en/latest/), [CellChat](http://www.cellchat.org/), [CellPhoneDB](https://www.cellphonedb.org/), [NATMI](https://github.com/asrhou/NATMI), and [Squidpy](https://squidpy.readthedocs.io/en/stable/).

![Integration and Analysis Method](docs/images/analyses_pipeline.png)

## Getting Started

### Installation

MMCCI can be installed with `pip`

```
pip install mmcci
```

### Documentation

Documentation is available at the [Read the Docs](https://mmcci.readthedocs.io/en/latest/)

## CCI Integration

MMCCI allows users to integrate multiple CCI results together, both:
1. Samples from a single platform (eg. Visium)
2. Samples from multiple platforms (eg. Visium, Xenium, CosMx, CODEX)

## CCI Analysis

MMCCI provides multiple useful analyses that can be run on the integrated networks or from a single sample:
1. Network comparison between groups with permutation testing
2. CLustering of LR pairs with similar networks
3. Clustering of spots/cells with similar interaction scores
4. Sender-receiver LR querying
5. GSEA pathway analysis

## Citing MMCCI

If you have used MMCCI in your research, please consider citing us: 
```

```

