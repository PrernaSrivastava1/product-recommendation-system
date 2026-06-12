"""
cf_recommender.py

Collaborative Filtering Recommendation System ? supports kNN-based and
SVD-based algorithms via the Surprise library.

Provides a unified interface for training, evaluation, cross-validation,
hyperparameter tuning, and generating personalised recommendations.

Classes
-------
AlgorithmMismatchError
    Raised when the algorithm class is incompatible with the requested fit method.
ModelNotTrainedError
    Raised when a method is called before the model has been trained.
CFRecommendationSystem
    Core class managing CF-based recommendation workflows.
"""

# ---------------------------------------------------------
# IMPORT LIBRARIES
# ---------------------------------------------------------

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import pandas as pd

from surprise.dataset import Dataset
from surprise.model_selection import GridSearchCV, cross_validate
from surprise.prediction_algorithms import (
    SVD,
    AlgoBase,
    KNNBaseline,
    KNNBasic,
    KNNWithMeans,
    KNNWithZScore,
    SVDpp,
)

from src.model_eval_functions import evaluate_model, get_recommendations
from src.utils import display_eval_metrics, smart_display, smart_display_df

# ---------------------------------------------------------
# EXCEPTIONS
# ---------------------------------------------------------


class AlgorithmMismatchError(Exception):
    """Raised when the chosen algorithm is incompatible with the fit method."""
    pass


class ModelNotTrainedError(Exception):
    """Raised when a prediction or evaluation is attempted before training."""
    pass


# ---------------------------------------------------------
# COLLABORATIVE FILTERING SYSTEM
# ---------------------------------------------------------


class CFRecommendationSystem:
    """
    A unified collaborative filtering recommender supporting kNN and SVD
    algorithm families.

    Supports training, evaluation, cross-validation, hyperparameter tuning
    via GridSearchCV, individual rating prediction, and top-N recommendation
    generation.

    Attributes
    ----------
    data : pd.DataFrame
        The full interaction DataFrame.
    algo_name : str
        Name of the algorithm class (e.g., 'KNNBasic', 'SVD').
    algo_class : Type[AlgoBase]
        The Surprise algorithm class to instantiate.
    params_grid : dict
        Default or grid parameters for the algorithm.
    model : AlgoBase | None
        Trained algorithm instance (populated after fit_knn / fit_svd).
    best_score : float | None
        Best RMSE from hyperparameter tuning.
    best_params : dict | None
        Best parameters found during tuning.
    gs : GridSearchCV | None
        The fitted GridSearchCV object.
    """

    __slots__ = (
        "data", "algo_name", "algo_class", "params_grid",
        "model", "best_score", "best_params", "gs",
    )

    def __init__(
        self,
        data: pd.DataFrame,
        algo_class: Type[AlgoBase],
        params_grid: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialise the CFRecommendationSystem.

        Parameters
        ----------
        data : pd.DataFrame
            Full interaction DataFrame (user_id, prod_id, rating).
        algo_class : Type[AlgoBase]
            A Surprise algorithm class (e.g., KNNBasic, SVD).
        params_grid : dict, optional
            Default parameters or a grid for tuning. Defaults to {}.
        """
        self.data = data
        self.algo_name = algo_class.__name__
        self.algo_class = algo_class
        self.params_grid = params_grid or {}
        self.model: Optional[AlgoBase] = None
        self.best_score: Optional[float] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.gs: Optional[GridSearchCV] = None

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------

    def fit_knn(self, trainset: Dataset, use_tuned_params: bool = False) -> None:
        """
        Train a kNN-based collaborative filtering model.

        Parameters
        ----------
        trainset : Dataset
            Surprise trainset object.
        use_tuned_params : bool
            If True, use parameters from the last hyperparameter tuning run.

        Raises
        ------
        AlgorithmMismatchError
            If algo_class is not a kNN variant.
        """
        if not issubclass(
            self.algo_class, (KNNBasic, KNNBaseline, KNNWithMeans, KNNWithZScore)
        ):
            raise AlgorithmMismatchError(
                f"{self.algo_class.__name__} is not a kNN algorithm. "
                "Use fit_svd() for SVD-based models."
            )

        source = self.best_params if (use_tuned_params and self.best_params) else self.params_grid
        model_params = {
            "k": source.get("k", 40),
            "min_k": source.get("min_k", 1),
            "sim_options": source.get("sim_options", {}),
            "verbose": False,
        }

        self.model = self.algo_class(**model_params)
        self.model.fit(trainset)
        print(f"  Trained {self.algo_name} | Parameters: {model_params}")

    def fit_svd(self, trainset: Dataset, use_tuned_params: bool = False) -> None:
        """
        Train an SVD-based matrix factorisation model.

        Parameters
        ----------
        trainset : Dataset
            Surprise trainset object.
        use_tuned_params : bool
            If True, use parameters from the last hyperparameter tuning run.

        Raises
        ------
        AlgorithmMismatchError
            If algo_class is not SVD or SVDpp.
        """
        if not issubclass(self.algo_class, (SVD, SVDpp)):
            raise AlgorithmMismatchError(
                f"{self.algo_class.__name__} is not an SVD algorithm. "
                "Use fit_knn() for kNN-based models."
            )

        source = self.best_params if (use_tuned_params and self.best_params) else self.params_grid
        model_params = {
            "n_factors": source.get("n_factors", 100),
            "n_epochs": source.get("n_epochs", 20),
            "lr_all": source.get("lr_all", 0.005),
            "reg_all": source.get("reg_all", 0.02),
            "random_state": source.get("random_state", 42),
        }

        self.model = self.algo_class(**model_params)
        self.model.fit(trainset)
        print(f"  Trained {self.algo_name} | Parameters: {model_params}")

    # ------------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------------

    def evaluate(
        self, trainset: Dataset, testset: Dataset, k: int = 10, th: float = 3.5
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Evaluate the model on train and test sets and display metric tables.

        Parameters
        ----------
        trainset : Dataset
            Surprise trainset.
        testset : Dataset
            Surprise testset (list of triples).
        k : int
            Top-K cut-off (default 10).
        th : float
            Relevance threshold (default 3.5).

        Returns
        -------
        tuple
            (train_metrics dict, test_metrics dict)
        """
        if self.model is None:
            raise ModelNotTrainedError("Call fit_knn() or fit_svd() first.")

        train_predictions = self.model.test(trainset.build_testset())
        train_pred_met, train_rank_met = evaluate_model(train_predictions, k=k, threshold=th)

        test_predictions = self.model.test(testset)
        test_pred_met, test_rank_met = evaluate_model(test_predictions, k=k, threshold=th)

        display_eval_metrics(
            train_pred_met, train_rank_met, test_pred_met, test_rank_met
        )

        return {**train_rank_met, **train_pred_met}, {**test_rank_met, **test_pred_met}

    def cross_val(
        self, data: Dataset, measures: Optional[List[str]] = None, cv: int = 5
    ) -> None:
        """
        Run k-fold cross-validation and print results.

        Parameters
        ----------
        data : Dataset
            Surprise Dataset object.
        measures : list, optional
            Metrics to evaluate (default ['rmse', 'mae']).
        cv : int
            Number of folds (default 5).
        """
        if self.model is None:
            raise ModelNotTrainedError("Call fit_knn() or fit_svd() first.")

        measures = measures or ["rmse", "mae"]
        cross_validate(
            algo=self.model, data=data, measures=measures,
            cv=cv, n_jobs=-1, verbose=True,
        )

    # ------------------------------------------------------------------
    # PREDICTION & RECOMMENDATION
    # ------------------------------------------------------------------

    def predict(
        self,
        int_data: Dict[str, List[Union[Tuple[str, float], str]]],
        has_interacted: bool,
    ) -> None:
        """
        Generate and display predicted ratings for specific user-item pairs.

        Parameters
        ----------
        int_data : dict
            {user_id: [(prod_id, true_rating), ...]} for interacted items, or
            {user_id: [prod_id, ...]} for non-interacted items.
        has_interacted : bool
            True when int_data contains (prod_id, true_rating) pairs;
            False when it contains only prod_ids.
        """
        if self.model is None:
            raise ModelNotTrainedError("Call fit_knn() or fit_svd() first.")

        phrase = "" if has_interacted else "Non-"
        smart_display(f"Rating Estimates for {phrase}Interacted Products")

        for user, interactions in int_data.items():
            predictions = []
            for interaction in interactions:
                if has_interacted:
                    iid, rui = interaction
                    pred = self.model.predict(uid=user, iid=iid, r_ui=rui, verbose=False)
                else:
                    iid = interaction
                    pred = self.model.predict(uid=user, iid=iid, verbose=False)
                predictions.append(pred)

            for pred in predictions:
                print(f"  user: {pred.uid:<14} item: {pred.iid:<12} "
                      f"r_ui = {str(pred.r_ui):<6} est = {pred.est:.2f}  {pred.details}")

    def tune_hyperparameters(
        self,
        param_grid: Dict[str, Any],
        measures: Optional[List[str]] = None,
        cv: int = 5,
    ) -> None:
        """
        Run GridSearchCV hyperparameter tuning and store the best parameters.

        Parameters
        ----------
        param_grid : dict
            Hyperparameter grid to search.
        measures : list, optional
            Metrics to optimise (default ['rmse']).
        cv : int
            Number of cross-validation folds (default 5).
        """
        measures = measures or ["rmse"]

        self.gs = GridSearchCV(
            self.algo_class, param_grid=param_grid, measures=measures, cv=cv, n_jobs=-1
        )
        self.gs.fit(self.data)
        self.best_score = self.gs.best_score[measures[0]]
        self.best_params = self.gs.best_params[measures[0]]

        smart_display("Hyperparameter Tuning Grid")
        print(f"  {param_grid}")

        smart_display("Hyperparameter Tuning Results")
        print(f"  {measures[0].upper()} : {self.best_score:.3f}")
        print(f"  Best params : {self.best_params}")

    def recommend(
        self,
        data: pd.DataFrame,
        int_data: Dict[str, List[Tuple[str, float]]],
        top_n: int = 5,
    ) -> None:
        """
        Generate and display top-N recommendations for each user.

        Parameters
        ----------
        data : pd.DataFrame
            Full interaction DataFrame.
        int_data : dict
            {user_id: [(prod_id, rating), ...]} ? used to identify users.
        top_n : int
            Number of recommendations per user (default 5).
        """
        if self.model is None:
            raise ModelNotTrainedError("Call fit_knn() or fit_svd() first.")

        for user in int_data.keys():
            recommendations = get_recommendations(
                data=data, algo=self.model, user_id=user, top_n=top_n
            )
            smart_display(f"Recommendations for User: {user}")
            smart_display_df(recommendations)


# ---------------------------------------------------------
# END OF MODULE
# ---------------------------------------------------------
