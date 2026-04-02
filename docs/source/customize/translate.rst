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


Apart from overriding the regression, spaAnchor also allows users to 
define the regression model for each target feature modality by specifying 
the ``regressor_key`` parameter when initializing the spaAnchor model. 

We also provide a set of regression models in ``spaAnchor.regressor``, including:

- ``spaAnchor.regressor.ridge_regressor``: Ridge regression model (:class:`~sklearn.linear_model.Ridge` / :class:`~sklearn.linear_model.LogisticRegression` with penalty="l2" for classification).
- ``spaAnchor.regressor.xgboost_regressor``: XGBoost regression model (:class:`~xgboost.XGBRegressor` / :class:`~xgboost.XGBClassifier` for classification).
- ``spaAnchor.regressor.mlp_regressor``: MLP regression model (:class:`~sklearn.neural_network.MLPRegressor` / :class:`~sklearn.neural_network.MLPClassifier` for classification).
- ``spaAnchor.regressor.randomforest_regressor``: Random Forest regression model (:class:`~sklearn.ensemble.RandomForestRegressor` / :class:`~sklearn.ensemble.RandomForestClassifier` for classification).
- ``spaAnchor.regressor.svm_regressor``: Support Vector Machine regression model (:class:`~sklearn.svm.SVR` / :class:`~sklearn.svm.SVC` for classification).
- ``spaAnchor.regressor.lightgbm_regressor``: LightGBM regression model (:class:`~lightgbm.LGBMRegressor` / :class:`~lightgbm.LGBMClassifier` for classification).

For one-hot data type transformation, we also provide the classifier version 
of those models in ``spaAnchor.regressor``, such as ``spaAnchor.regressor.ridge_classifier``. 
For classifier, spaAnchor output the predicted probability for each class matching to the input one-hot matrix.
Users can directly use those regression/classification models by inputting the corresponding 
key to the ``regressor_key`` parameter.

Differ from spaAnchor's default regression model, user can set the parameters for those regression models
by inputting paras when initial them in the ``regressor_key`` parameter (for detailed parameter list, 
please refer to the corresponding model's documentation). 


.. code:: python
        
    import spaAnchor as sa

    feature_key = ["X", "mod1", "mod2", "ont_hot_mod1", ...]  # The feature modality user want to translate.
    regressor_key = {
        "X": None, # Use default regression (Ridge) for feature modality "X"
        # Use MLP regression for feature modality "mod1" with specified parameters
        # the parameters here are just for demonstration, user can set them according to their data's pattern.
        "mod1": sa.regressor.mlp_regressor(hidden_layer_sizes=(100, 50), max_iter=200), 
        "mod2": sa.regressor.xgboost_regressor(n_estimators=100, max_depth=5, learning_rate=0.1),
        "ont_hot_mod1": sa.regressor.randomforest_classifier(n_estimators=100, max_depth=5),
        ...
    }
    model = sa.HANN(feature_key=feature_key, regressor_key=regressor_key)   # Any our prebuilt model or customized model.
    transfer_result = model(adata_list)


Also if users have their own regression model, they can also directly pass the 
regression function to this parameter.

For user defined regression function, the function should have those preserved parameters:

.. code:: python

    import spaAnchor as sa
        
    def user_defined_regression(
        self,
        source: np.ndarray, 
        target: np.ndarray, 
        predict: Optional[np.ndarray] = None,
        top_pcs: int = 50, 
        n_jobs: int = -1,
        **kwargs
    ) -> np.ndarray:
        
        ...


    feature_key = ["mod", ...]
    regressor_key = {
        "mod": user_defined_regression, # Should assign the callable function object, not the result of the function.
        ...
    }
    model = sa.HANN(feature_key=feature_key, regressor_key=regressor_key)   # Any our prebuilt model or customized model.
    transfer_result = model(adata_list)

