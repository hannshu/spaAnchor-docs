Customized spaAnchor
--------------------

Parameters for ``spaAnchor.HANN``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We will begin by introducing the parameters utilized in the foundational
``spaAnchor.HANN`` model. This model is the core of our
search-and-regress paradigm, and users can easily inherit from this
class to customize and build spaAnchor models tailored to their specific
requirements.

    seed : ``int``, default = ``0``
        Random seed.
    filter_cell : ``bool``, default = ``True``
        If True, filters out non-expressed cells.
    preprocess : ``bool``, default = ``True``
        If True, performs preprocessing steps on the input data.
        This includes highly variable gene selection, normalization, and log-transformation.
    n_hvg : ``int``, default = ``8000``
        Number of highly variable genes (HVGs) to select during preprocessing.
    spatial_key : ``str``, default = ``"spatial"``
        Key in the ``sc.AnnData`` object's ``.obsm`` that stores the spatial coordinates.
    reduction_opt : ``Literal["pca", "harmony", "outside"]``, default = ``"pca"``
        Method to use for dimensionality reduction of gene expression data.
        We provide two preprocess methods: PCA and Harmony.
        User can also do their customized preprocess of the shared features first.

        *   ``"pca"``: Principal Component Analysis.
        *   ``"harmony"``: Harmony batch correction followed by PCA.
        *   ``"outside"``: Assumes dimensionality reduction has already been performed
            and uses pre-existing embeddings.
    reduction_dim : ``int``, default = ``30``
        Number of dimensions for latent embedding.
        Applicable when ``reduction_opt`` is "pca" or "harmony".
    niche_threshold : ``Optional[int]``, default = ``None``
        Number of cells identified for each cell, using the niche-level embedding. 
        If input ``None``, this is set to max(300, int(cell_count * 3e-4)).
    cell_threshold : ``Optional[int]``, default = ``None``
        Number of cells identified for each cell, using the cell-level embedding. 
        If input ``None``, this is set to max(50, int(cell_count * 1.5e-4)).
    dis_threshold : float, default = 0.75
        Distance threshold used for filter less-confidence cell pairs.
    scale : ``bool``, default = ``False``
        If True, scales the predicted features for training regresser.
    imbalanced : ``bool``, default = ``False``
        If True, enables strategies to handle imbalanced datasets (loose the 
        ``niche_threshold`` and ``cell_threshold`` to identified more cells for
        the small dataset).
    feature_key : ``Optional[Union[str, List[str]]]``, default = ``None``
        Key(s) in the AnnData object's ``.obsm`` that store features for prediction.
        Can be a single string (to predict one type of feature) or a list of strings 
        (to predict multiple feature types).
        If None, ``.X`` will be used.
    emb_key : ``str``, default = ``"cell_latent"``
        Key of the embedding use for train the aggregate attention weight. 
    device : ``str``, default = ``"cuda"``
        The device on which to run the model (e.g., "cuda" for GPU or "cpu" for CPU).


Pre-built spaAnchor models
~~~~~~~~~~~~~~~~~~~~~~~~~~

spaAnchor provide four pre-built models:

-  ``spaAnchor.panel_expansion()`` for omics panel expansion.
   (`tutorial <https://spaanchor.readthedocs.io/en/latest/panel_expansion.html>`__)
-  ``spaAnchor.rna2protein()`` for cross-model translation.
   (`tutorial <https://spaanchor.readthedocs.io/en/latest/omics_translation.html>`__)
-  ``spaAnchor.diagonal_integration()`` for diagonal integration.
   (`tutorial <https://spaanchor.readthedocs.io/en/latest/diagonal_integration.html>`__)
-  ``spaAnchor.asymmetric_translation()`` for asymmetric multi-modal
   translation.
   (`tutorial <https://spaanchor.readthedocs.io/en/latest/asymmetric_translation.html>`__)

spaAnchor modules
~~~~~~~~~~~~~~~~~

While spaAnchor provides pre-built functions for common scenarios, it is
designed as a unified framework. We strongly encourage the user to adapt
and extend its core components to fit their unique experimental needs.

The spaAnchor pipeline is built on four major modules: 
- Preprocess: Cleans, normalizes the input data. 
- Projection (Representation learning): Generates the cell- and niche-level
latent embedding. The cell- and niche-level embedding should save at ``.obsm["cell_latent"]``
and ``.obsm["niche_latent"]`` for each slice's ``sc.AnnData`` object.
- Mapping (HNN identification): Learns the relationship between 
different slices using both cellular features and the spatial information. 
- Translation (Attention-based feature transfermation and Global 
prediction): Applies the learned mapping to predict unmeasured features 
in the target slice.

+-------------+--------------------------------------------------------------------------------------------+------------------------------------+
| Module      | Input                                                                                      | Output                             |
+=============+============================================================================================+====================================+
| Preprocess  | adata_list: List[sc.AnnData]                                                               | None                               |
+-------------+--------------------------------------------------------------------------------------------+------------------------------------+
| Projection  | adata_list: List[sc.AnnData]                                                               | List[sc.AnnData]                   |
+-------------+--------------------------------------------------------------------------------------------+------------------------------------+
| Mapping     | adata_list: List[sc.AnnData]                                                               | Dict[int, np.ndarray]              |
+-------------+--------------------------------------------------------------------------------------------+------------------------------------+
| Translation | adata_list: Union[List[sc.AnnData], Dict[str, sc.AnnData]], mapping_dict: List[np.ndarray] | Dict[int, Dict[str, pd.DataFrame]] |
+-------------+--------------------------------------------------------------------------------------------+------------------------------------+


How to customize
~~~~~~~~~~~~~~~~

We will takes the ``spaAnchor.diagonal_integration`` as an example. A
core architectural requirement is that all model classes should depend
on the ``spaAnchor.HANN`` class. This dependency is managed by the base
class to standardize the prediction workflow, allowing users to focus
solely on implementing model-specific logic.

.. code:: python

   import spaAnchor as sa

   class diagonal_integration(sa.HANN):

       def __init__(self, ..., **kws):
           super().__init__(**kws)
           ...

For diagonal integration, the model anchors the analysis on the cellular
histology feature. Consequently, the modification required is to
overload the ``projection()`` method to define how to extract cellular
feature from histology image.

.. code:: python

   class diagonal_integration(sa.HANN):

       def __init__(self, ..., **kws):
           super().__init__(**kws)
           ...


       def projection(self, adata_list) -> List[sc.AnnData]:
           model = get_vit256(pretrained_weights=MODEL_PATH, device=self.device)

           cell_embedding_list = []
           niche_embedding_list = []
           for adata in adata_list:
               niche_embedding, cell_embedding = get_cell_embedding(adata, model, eval_transforms(), self.spatial_key, 
                                   self.library_id, self.img_basis, self.scale_factor, self.batch_size, self.device) 
               cell_embedding_list.append(cell_embedding)
               niche_embedding_list.append(niche_embedding)
           del model; clean_cuda_cache()    # uninstall image model

           cell_embedding_list = np.vstack(cell_embedding_list)
           niche_embedding_list = np.vstack(niche_embedding_list)
           adatas = sc.concat(adata_list, label="batch")
           pca = PCA(n_components=self.reduction_dim, random_state=0)
           adatas.obsm["cell_latent"] = pca.fit_transform(cell_embedding_list)
           adatas.obsm["niche_latent"] = pca.fit_transform(np.vstack(niche_embedding_list))
           for i, adata in zip(adatas.obs["batch"].unique(), adata_list):
               adata.obsm["cell_latent"] = adatas[i == adatas.obs["batch"]].obsm["cell_latent"]
               adata.obsm["niche_latent"] = adatas[i == adatas.obs["batch"]].obsm["niche_latent"]
           latent_adata_list = adata_list

           for i in range(len(latent_adata_list)):
               adata_list[i].obsm["cell_latent"] = latent_adata_list[i].obsm["cell_latent"] 
               adata_list[i].obsm["niche_latent"] = latent_adata_list[i].obsm["niche_latent"]

           return adata_list
       
       ...

Then you can use this model for diagonal integration.

.. code:: python

   ...
   model = diagonal_integration()
   predicted_result = model([adata0.copy(), adata1.copy(), ...])
   ...

This is a brief guide to help you customize spaAnchor. If you encounter
any issues or need help adapting the model for your specific use case,
please open an `issue <https://github.com/yuanstlab/spaanchor/issues>`__
on our GitHub repository. We are happy to help you extend the framework.

Moreover, if you have created a custom model that you’d like to share
with the community, we welcome you to submit a `pull
request <https://github.com/yuanstlab/spaanchor/pulls>`__.
