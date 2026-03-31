Customized :meth:`~spaAnchor.HANN.preprocess`
---------------------------------------------

Preprocessing is a crucial step in the spaAnchor workflow, 
as it prepares the input data for subsequent analysis. 
The :meth:`~spaAnchor.HANN.preprocess` function is responsible for handling 
preprocessing tasks, including feature selection, data normalization,
and spatial graph construction.

Our default preprocessing pipeline includes filter non-expressed cells,
highly variable genes selection, total count normalization, log1p transformation, 
and spatial graph construction.

For users who wish to customize the preprocessing step, they can override the 
:meth:`~spaAnchor.HANN.preprocess` function in their custom model class and change the default 
preprocessing pipeline according to their specific needs.

such as:

.. code:: python

    import spaAnchor as sa

    class custom_hvg(sa.HANN):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def preprocess(
            self,
            adata_list: List[sc.AnnData],
            **kwargs
        ) -> List[torch.LongTensor]:
            graph_list = []

            for adata in adata_list:

                if (self.filter_cell): 
                    sc.pp.filter_cells(adata, min_genes=1)

                if (self.pp):
                    # sc.pp.highly_variable_genes(adata, n_top_genes=self.n_hvg, flavor="seurat_v3", subset=True)
                    # Such as using cell_ranger for highly variable gene selection.
                    sc.pp.highly_variable_genes(adata, n_top_genes=self.n_hvg, flavor="cell_ranger", subset=True)
                    sc.pp.normalize_total(adata, target_sum=1e4)
                    sc.pp.log1p(adata)

                graph_list.append(self.build_graph(adata.obsm[self.spatial_key], **kwargs))

            return graph_list


Moreover, we use Delaunay to generate spatial graph, if user want to use other
method such as KNN, they can override the :meth:`~spaAnchor.HANN.build_graph` function and change 
the spatial graph construction method.

.. code:: python

    import spaAnchor as sa

    class custom_graph_construction(sa.HANN):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def build_graph(self, coords: np.ndarray, user_defined_k: int, **kwargs) -> torch.Tensor:
            from sklearn.neighbors import kneighbors_graph

            adj = kneighbors_graph(coords, n_neighbors=user_defined_k, include_self=False).tocoo()
            edge_index = torch.tensor(np.array([adj.row, adj.col]), dtype=torch.long)

            return edge_index







