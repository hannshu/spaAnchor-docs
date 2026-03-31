Customize :meth:`~spaAnchor.HANN.translate`
-------------------------------------------

The translate module is responsible for predicting missing features 
in the target slice based on the source slice. It takes the identified HNN 
pairs to firstly transfer source slice unique features to target slice 
(attention-based feature translation) and then uses regression to predict 
those features for all cells in target slice.


Customize attention-based feature translation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


For attention-based feature aggregation, computational efficiency may be 
affected when the number of target features is large. To address this, 
spaAnchor supports an alternative mode in which the aggregation weights 
are learned from latent features rather than the expression feature space, 
and the resulting weights are then used to aggregate the expression features. 
This approach substantially reduces computational cost while maintaining 
prediction quality, and is particularly recommended for high-dimensional 
translation targets such as epigenomic peaks. This latent-space aggregation 
mode was adopted in our RNA-Epigenomics translation experiments.

.. code:: python

    class rna2epig(rna2protein):

        def __init__(
            self, 
            feature_key="epigenomics", 
            reduction_dim=100, 
            **kwargs
        ):
            super().__init__(
                feature_key=feature_key, 
                reduction_dim=reduction_dim,
                **kwargs
            )


        def translate(
            self, 
            source_emb, 
            source_feat, 
            target_emb, 
            target_feat, 
            hnn_pairs, 
            index, 
            columns, 
            **kwargs
        ):
            import pandas as pd
            from ..utils.translation import KNN_regression
            from ..utils.attn_model import attn_translation
            from ..utils.utils import lsi
            
            impute_result = attn_translation(
                source_x=source_emb,
                source_y=lsi(source_feat, 50),  # Train by LSI-reduced features
                target_x=target_emb,
                raw_edge_index=hnn_pairs,
                # To activate the latent feature learn -> expression feature aggregate
                # mode, user should input the source feature as `t_source_y`, and the 
                # model will learn to predict the target feature by aggregating source 
                # features according to the attention weight.
                t_source_y=source_feat
            )

            # Due to the extremely high dimension of the target feature, we use the 
            # KNN regression to do global prediction.
            impute_result = KNN_regression(impute_result, target_emb)
            return pd.DataFrame(impute_result, index=index, columns=columns)
    

Customize regression for global prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Although our default Ridge regression works well in most cases, users can also implement 
their own regression function for global prediction to better fit there data's pattern. 
Such as using MLP for translation.

.. code:: python
        
    import spaAnchor as sa
    import numpy as np
    from typing import Optional


    class custom_mlp(sa.HANN):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)


        def feature_regression(
            self,
            source: np.ndarray, 
            target: np.ndarray, 
            predict: Optional[np.ndarray] = None,
            top_pcs: int = 50, 
            # Although this parameter not used in this function, 
            # this still need to be kept for compatibility with 
            # the original function signature.
            n_jobs: int = -1,
            **kwargs
        ) -> np.ndarray:
            
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            from sklearn.neural_network import MLPRegressor

            predict = predict if (isinstance(predict, np.ndarray)) else source
            assert (source.shape[1] == predict.shape[1]), "ERROR: source and predict must have same feature size."

            regressor = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=min(top_pcs, source.shape[1]), random_state=0)),
                ("mlp", MLPRegressor(random_state=0))
            ])
            return regressor.fit(source, target).predict(predict)




