Usage Examples
==============

Merging Samples
---------------

Combine multiple samples using the integration function:

.. code-block:: python

    from mmcci.integration import lr_integration
    integrated_data = lr_integration(samples, method=">=50%", strict=True)
    print(integrated_data)

Plotting Results
----------------

Visualize interaction data:

.. code-block:: python

    from mmcci.plotting import lr_barplot
    lr_barplot(sample_data, title="LR Interactions")
