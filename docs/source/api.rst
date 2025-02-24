API
===

CCIData Class
~~~~~~~

.. module:: mmcci.CCIData_class
.. currentmodule:: mmcci

.. autosummary::
    :toctree: api

    CCIData

Integration
~~~~~~~~~~~

.. module:: mmcci.it
.. currentmodule:: mmcci

.. autosummary::
    :toctree: api

    it.get_lr_pairs
    it.calc_scale_factors
    it.lr_integration
    it.integrate_networks

Analysis
~~~~~~~~

.. module:: mmcci.an
.. currentmodule:: mmcci

.. autosummary::
    :toctree: api

    an.calculate_dissim
    an.get_network_diff
    an.cell_network_clustering
    an.lr_interaction_clustering
    an.run_gsea
    an.pathway_subset
    an.add_lr_module_score

Scoring
~~~~~~~

.. module:: mmcci.sc
.. currentmodule:: mmcci

.. autosummary::
    :toctree: api

    sc.dissimilarity_score
    sc.multiply_non_zero_values

Plotting
~~~~~~~~

.. module:: mmcci.pl
.. currentmodule:: mmcci

.. autosummary::
    :toctree: api

    pl.network_plot
    pl.chord_plot
    pl.dissim_hist
    pl.lr_top_dissimilarity
    pl.lr_barplot
    pl.lrs_per_celltype

IO
~~~

.. module:: mmcci.io
.. currentmodule:: mmcci

.. autosummary::
    :toctree: api

    io.read_stLearn
    io.convert_stLearn
    io.read_CellPhoneDB
    io.read_Squidpy
    io.read_CellChat
    io.read_NATMI
    io.read_CCIData
    io.read_network
    io.from_dict
