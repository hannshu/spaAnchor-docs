from typing import List, Dict, Optional, Literal, Union, Callable
from numbers import Number
import numpy as np
import pandas as pd
import scanpy as sc
import torch

class HANN:
    r"""
    This class is the foundational implement for spaAnchor pipeline.
    For detailed usage, please refer to the tutorial.

    This class implements the four major steps of the spaAnchor pipeline, including:  

    1. Preprocessing and graph construction (:meth:`preprocess`).  

    2. Latent feature generation (Projection, :meth:`projection`).  

    3. HNN pairs identification (:meth:`mapping`).  

    4. Feature translation (:meth:`translate`).  

    Args:
        filter_cell (bool, optional): Whether to filter cells with zero expression 
            during preprocessing. (default: :obj:`True`)
        preprocess (bool, optional): Whether to perform preprocessing (HVG selection, normalization, log1p). 
            (default: :obj:`True`)
        n_hvg (int, optional): Number of highly variable genes for generate latent features. 
            (default: :obj:`8000`)
        spatial_key (str, optional): Key in :attr:`adata.obsm` where spatial 
            coordinates are stored. (default: :obj:`"spatial"`)
        reduction_opt (str, optional): Dimensionality reduction method. 
            User can customize the embedding by generate `cell_latent` and `niche_latent` in :attr:`adata.obsm` 
            before input or customize the `self.projection` function for generating 
            cell-level and niche-level latent features.
            Options: :obj:`"pca"`, :obj:`"harmony"`, :obj:`"outside"`. 
            (default: :obj:`"pca"`)
        reduction_dim (int, optional): Number of dimensions for the latent 
            representation. (default: :obj:`30`)
        niche_threshold (int, optional): Neighbor threshold for niche-level 
            mapping. If :obj:`None`, calculated based on dataset size. 
            (default: :obj:`None`)
        cell_threshold (int, optional): Neighbor threshold for cell-level 
            mapping. If :obj:`None`, calculated based on dataset size. 
            (default: :obj:`None`)
        dis_threshold (float, optional): Distance threshold for HNN identification. 
            (default: :obj:`0.75`)
        scale (bool or Number, optional): Scaling factor applied to target features 
            to ensure numerical stability. Target features are divided by this factor 
            to prevent excessively large input magnitudes. If set to :obj:`True`, 
            the scaling factor is set to :obj:`1000`. Alternatively, a specific 
            numerical value can be provided. (default: :obj:`False`)
            (default: :obj:`False`)
        imbalanced (bool, optional): If set to :obj:`True`, the neighbor 
            search space in HNN is dynamically expanded based on the ratio of 
            dataset sizes (e.g., :obj:`slice_0.shape[0] / slice_1.shape[0]`). 
            This ensures more robust matching when one slice is significantly 
            larger than the other. (default: :obj:`False`)
        feature_key (str or List[str], optional): Keys in :attr:`adata.obsm` 
            to be translated. (default: :obj:`["X"]`)
        regressor_key (Dict[str, Callable], optional): Mapping of feature keys to 
            specific regression functions. :obj:`None` to use the default Ridge
            regression. (default: :obj:`None`)
        regression_feat (str, optional): Which type of intrinsic feature
            set to use as the regressor input. raw count for :obj:`"X_raw"`, 
            preprocessed count for :obj:`"X_pp"`. (default: :obj:`"X_raw"`)
        emb_key (str, optional): Set the cell features used as input for attention-based
            feature transformation. (default: :obj:`"cell_latent"`)
        predict_full (bool, optional): If :obj:`True`, predicts the full feature 
            panel even for shared features. Set :obj:`False` for preserving legacy 
            spaAnchor output. (default: :obj:`False`)
        device (str, optional): Computation device (:obj:`"cuda"` or :obj:`"cpu"`). 
            (default: :obj:`"cuda"`)
    """

    def __init__(
        self,
        filter_cell: bool = True,
        preprocess: bool = True,
        n_hvg: int = 8e3,
        spatial_key: str = "spatial",
        reduction_opt: Literal["pca", "harmony", "outside"] = "pca",
        reduction_dim: int = 30,
        niche_threshold: Optional[int] = None,
        cell_threshold: Optional[int] = None,
        dis_threshold: float = 0.75,
        scale: Union[bool, Number] = False,
        imbalanced: bool = False,
        feature_key: Optional[Union[str, List[str]]] = None,
        regressor_key: Optional[Dict[str, Callable]] = None, 
        regression_feat: Literal["X_raw", "X_pp"] = "X_raw",
        emb_key: str = "cell_latent",
        predict_full: bool = False,
        device: str = "cuda" 
    ):
        pass

    def build_graph(self, coords: np.ndarray, **kwargs) -> torch.Tensor:
        r"""Constructs a spatial graph using Delaunay triangulation.

        Args:
            coords (numpy.ndarray): Spatial coordinates of cells.
            **kwargs (optional): Additional arguments.

        Returns:
            torch.Tensor: Edge index of the graph (:obj:`(2, E)`).
        """
        pass

    def preprocess(
        self,
        adata_list: List[sc.AnnData],
        **kwargs
    ) -> List[torch.Tensor]:
        r"""Prepares AnnData objects for HANN processing.

        Performs cell filtering, HVG selection, normalization, log transformation, 
        and stores counts in :attr:`.obsm`. Also calls :meth:`build_graph` for 
        spatial graph construction.

        Args:
            adata_list (List[sc.AnnData]): List of spatial AnnData objects.
            **kwargs (optional): Additional arguments.

        Returns:
            List[torch.Tensor]: List of edge indices for each slice.
        
        Raises:
            ValueError: If :attr:`AnnData.X` is empty or incorrectly formatted.
        """
        pass

    def projection_cell(
        self,
        adata_list: List[sc.AnnData],
        **kwargs
    ) -> List[sc.AnnData]:
        r"""Generates cell-level latent features.

        Args:
            adata_list (List[sc.AnnData]): Preprocessed AnnData objects.
            **kwargs (optional): Additional arguments.

        Returns:
            List[sc.AnnData]: AnnData objects with cell-level latent features stored at :obj:`adata.obsm["cell_latent"]`.
        """
        pass

    def projection(
        self,
        adata_list: List[sc.AnnData],
        graphs: List[torch.Tensor],
        **kwargs
    ) -> List[sc.AnnData]:
        r"""Generates latent features (cell-level and niche-level).

        Args:
            adata_list (List[sc.AnnData]): List of AnnData objects.
            graphs (List[torch.Tensor]): Spatial graphs for each slice.
            **kwargs (optional): Additional arguments.

        Returns:
            List[sc.AnnData]: AnnData objects updated with :obj:`cell_latent` and :obj:`niche_latent`.
        """
        pass

    def mapping(
        self,
        adata_list: List[sc.AnnData],
        **kwargs
    ) -> np.ndarray:
        r"""Finds cross-slice MNN pairs using both cell and niche latent spaces.

        Args:
            adata_list (List[sc.AnnData]): Two AnnData objects to be aligned.
            **kwargs (optional): Additional arguments.

        Returns:
            numpy.ndarray: Intersection of cell-wise and niche-wise MNN pairs.
        """
        pass

    def feature_regression(
        self,
        source: np.ndarray, 
        target: np.ndarray, 
        predict: Optional[np.ndarray] = None,
        top_pcs: int = 50, 
        n_jobs: int = -1,
        **kwargs
    ) -> np.ndarray:
        r"""Global prediction for the target slice.

        Args:
            source (numpy.ndarray): Intrinsic feature of the HANN searched cells.
            target (numpy.ndarray): HANN transformed feature matrix of the HANN searched cells.
            predict (numpy.ndarray, optional): Intrinsic feature for all cells.
                (default: :obj:`None`)
            top_pcs (int, optional): Number of PCs for dimensionality reduction. 
                (default: :obj:`50`)
            n_jobs (int, optional): Number of parallel jobs. (default: :obj:`-1`)
            **kwargs (optional): Additional arguments.

        Returns:
            numpy.ndarray: Predicted features.
        """
        pass

    def translate(
        self,
        source_emb: np.ndarray,
        source_feat: np.ndarray,
        target_emb: np.ndarray,
        target_feat: np.ndarray,
        hnn_pairs: np.ndarray,
        index: List[str],
        columns: List[str],
        regression_func: Callable = "self.feature_regression",
        **kwargs
    ) -> pd.DataFrame:
        r"""Translates features from source slice to target slice.

        This step including two stages:  

        1. Attention-based feature transformation using the HNN pairs.  

        2. Global regression to predict features for all cells in the target slice.  

        Args:
            source_emb (numpy.ndarray): Latent features of the source slice.
            source_feat (numpy.ndarray): Features of the source slice to be transferred.
            target_emb (numpy.ndarray): Latent features of the target slice.
            target_feat (numpy.ndarray): Intrinsic features of the target slice for regression.
            hnn_pairs (numpy.ndarray): HNN pairs between source and target slices.
            index (List[str]): Index for the output DataFrame.
            columns (List[str]): Feature names for the output DataFrame.
            regression_func (Callable, optional): Function to perform global regression. 
                Please note that this part need a :obj:`~typing.Callable` function object, not a function result.
                (default: :meth:`self.feature_regression <spaAnchor.HANN.feature_regression>`)
            **kwargs (optional): Additional arguments.

        Returns:
            pandas.DataFrame: Predicted features for the target slice.
        """
        pass

    def predict_missing_feat(
        self,
        source_adata: sc.AnnData,
        target_adata: sc.AnnData,
        hnn_pairs: np.ndarray,
        feat_key: str,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        r"""Predict missing features between a pair of slices.

        Args:
            source_adata (sc.AnnData): Source data slice.
            target_adata (sc.AnnData): Target data slice.
            hnn_pairs (numpy.ndarray): HNN pairs between source and target slices.
            feat_key (str): Key of the feature in :attr:`source_adata.obsm` to predict.
            **kwargs (optional): Additional arguments.

        Returns:
            pandas.DataFrame, optional: Predicted features or :obj:`None` if key is missing.
        """
        pass

    def forward(
        self,
        adata_list: List[sc.AnnData],
        tgt_id: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[int, Dict[str, pd.DataFrame]]:
        r"""Executes the spaAnchor pipeline.

        Args:
            adata_list (List[sc.AnnData]): List of spatial slices.
            tgt_id (numpy.ndarray, optional): A matrix indicates which slice need to translate 
                feature from which slice. The `tgt_id` matrix should be like:

                .. code-block:: python

                    tgt_id = np.array([
                        [True,  True,  False],
                        [False, True,  True ],
                        [True,  True,  True ]
                    ])


                `tgt_id[0][1]` is `True` indicating slice 0 should be translated by slice 1's information.  

                `tgt_id[1][0]` is `False` indicating slice 1 has nothing to translate from slice 0.    

                The diagonal of `tgt_id` matrix can be either `True` or `False`.
            **kwargs (optional): Additional arguments.

        Returns:
            Dict[int, Dict[str, pandas.DataFrame]]: A dictionary containing translated features. Access via :obj:`result[target_slice_index][feature_name]`.`feature_name` should be like `{what feature?}_from_{which slice?}`.
        """
        pass