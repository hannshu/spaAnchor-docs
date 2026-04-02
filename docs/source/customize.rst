Customize spaAnchor
-------------------

In this section, we will explain how to customize the foundational
``spaAnchor.HANN`` model to fit user's specific needs. 
``spaAnchor.HANN`` is the core of our search-and-regress paradigm, 
and users can easily inherit from this class to customize and build 
spaAnchor models tailored to their specific requirements. 

We highly recommendusers to read the ``spaAnchor.HANN`` (`API documentation <HANN_class.rst>`__) 
before customizing their own model.

spaAnchor's implementation is modular, and can majorly divided into four modules:  

1. :meth:`~spaAnchor.HANN.preprocess` for preprocessing the input :meth:`sc.AnnData` object and
construct spatial graph for generating cell-level and niche-level latent features.  

2. :meth:`~spaAnchor.HANN.projection` for genetating the cell-level latent features by using the
cross-slice shared features and use the cell-level latent features and spatial graph
to generate the niche-level latent features.   

3. :meth:`~spaAnchor.HANN.mapping` for HNN pairs identification using the previously generated
cell-level and niche-level latent features (We highly recommend users to keep the default HNN 
pair identification method, as it is a key step in the search-and-regress paradigm).

4. :meth:`~spaAnchor.HANN.translate` for feature translation from source slice to target slice.  


How to customize
~~~~~~~~~~~~~~~~

For customizing a spaAnchor model, a core architectural requirement 
is that all model classes must inherit from the ``spaAnchor.HANN`` 
base class. This standardizes the spaAnchor workflow, allowing users 
to focus solely on implementing module-specific logic without modifying 
the core pipeline.

Users can override any of the major modules to implement their 
custom logic. When doing so, please do not modify the existing parameters 
in the function signature. We highly recommend prefixing any new parameters 
with ``user_defined_`` or another unique identifier to avoid conflicts with 
existing parameters.

.. note::
   If you are not familiar with the internals of spaAnchor, we recommend 
   copying the source code of the module you wish to customize as a 
   start and modify it incrementally. This ensures that all 
   required behaviors are preserved while minimizing the risk of 
   introducing unintended behavior. The source code for each module is 
   available on our `GitHub repository <https://github.com/yuanstlab/spaanchor/blob/main/spaAnchor/modules/core.py>`_.


.. code:: python

    import spaAnchor as sa

    class custom_model(sa.HANN):

        def __init__(self, ..., **kwargs):
            super().__init__(**kwargs)
            ...

        ... # Override your customized function.


Below are the specific topics:

.. toctree::

    customize/preprocess
    customize/projection
    customize/translate


After building the model, user can run this customized model with the same workflow as the default model.

Please note that if the function user customized have new parameters without setting a
default value, user should specified this parameter when running this model. Such as for 
this customized KNN graph construction with a customized parameter ``user_defined_``, 
user should run this model with: 


.. code:: python

    model = custom_graph_construction(...)
    result_dict = model(adata_list, user_defined_k=6, ...)


All examples in this section can be loaded and used by ``spaAnchor.extra``.

This is a brief guide to help users customize spaAnchor. If you encounter
any issues or need help adapting the model for your specific use case,
please open an `issue <https://github.com/yuanstlab/spaanchor/issues>`__
on our GitHub repository. We are happy to help you extend the framework.

Moreover, if you have created a custom model that you'd like to share
with the community, we welcome you to submit a `pull
request <https://github.com/yuanstlab/spaanchor/pulls>`__.
