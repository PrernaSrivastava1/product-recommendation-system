"""
rank_recommender.py

Rank-Based Recommendation System ? uses Bayesian or arithmetic average
scoring to surface popular, well-rated products.

Classes
-------
InvalidMethodError
    Raised when an unsupported scoring method is requested.
RankRecommendationSystem
    Computes popularity scores, evaluates the model, and surfaces top-N
    product recommendations for all users.
"""

# ---------------------------------------------------------
# IMPORT LIBRARIES
# ---------------------------------------------------------

from typing import Dict, List, Optional, Tuple

import pandas as pd
from surprise import Dataset
from surprise.prediction_algorithms.predictions import Prediction

from src.model_eval_functions import evaluate_model
from src.utils import display_eval_metrics, smart_display, smart_display_df

# ---------------------------------------------------------
# EXCEPTIONS
# ---------------------------------------------------------


class InvalidMethodError(Exception):
    """Raised when an unrecognised scoring method is supplied."""
    pass


# ---------------------------------------------------------
# RANK RECOMMENDATION SYSTEM
# ---------------------------------------------------------


class RankRecommendationSystem:
    """
    A rank-based recommender that scores products by Bayesian or simple
    average ratings, then recommends the highest-scoring items.

    Bayesian averaging shrinks products with few ratings toward the global
    mean, preventing low-interaction items from dominating the top-N list.

    Attributes
    ----------
    method : str
        Scoring method ? 'bayesian' or 'average'.
    algo_name : str
        Short algorithm label used in comparison tables.
    key_column : str
        DataFrame column name holding the computed score.
    scores : pd.DataFrame | None
        Product-level score table populated by compute_scores().
    """

    __slots__ = ("method", "algo_name", "key_column", "scores")

    def __init__(self, method: str = "bayesian"):
        """
        Initialise the RankRecommendationSystem.

        Parameters
        ----------
        method : str
            'bayesian' (default) or 'average'.

        Raises
        ------
        InvalidMethodError
            If the method is not one of the supported options.
        """
        self.method = method.lower()
        if self.method not in ["bayesian", "average"]:
            raise InvalidMethodError("method must be 'bayesian' or 'average'.")

        self.algo_name: str = "BAvg" if self.method == "bayesian" else "Avg"
        self.key_column: str = (
            "bayesian_rating" if self.method == "bayesian" else "avg_rating"
        )
        self.scores: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def compute_scores(
        self, data: pd.DataFrame, gb_feature: str, filter_feature: str
    ) -> None:
        """
        Compute rank-based product scores.

        Calculates rating counts, simple averages, and (if method='bayesian')
        Bayesian averages for each product.

        Parameters
        ----------
        data : pd.DataFrame
            Interaction DataFrame with at least [user_id, prod_id, rating].
        gb_feature : str
            Column to group by (e.g. 'prod_id').
        filter_feature : str
            Column containing the numeric rating (e.g. 'rating').
        """
        model_scores = data.groupby(gb_feature)[filter_feature].agg(
            cnt_rating="count", avg_rating="mean"
        )

        global_rating = data[filter_feature].mean()
        kn = model_scores["cnt_rating"].mean()

        model_scores["bayesian_rating"] = (
            global_rating * kn + model_scores["cnt_rating"] * model_scores["avg_rating"]
        ) / (kn + model_scores["cnt_rating"])

        self.scores = model_scores

    def evaluate(
        self, trainset: Dataset, testset: Dataset, k: int = 10, th: float = 3.5
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Evaluate the rank-based model on train and test sets.

        Parameters
        ----------
        trainset : Dataset
            Surprise trainset object.
        testset : Dataset
            Surprise testset list.
        k : int
            Top-K cut-off for ranking metrics (default 10).
        th : float
            Rating threshold for relevance (default 3.5).

        Returns
        -------
        tuple
            (train_metrics dict, test_metrics dict)

        Raises
        ------
        ValueError
            If compute_scores() has not been called yet.
        """
        if self.scores is None:
            raise ValueError("Call compute_scores() before evaluate().")

        model_dict = self.scores[self.key_column].to_dict()
        global_rating = self.scores[self.key_column].mean()

        train_predictions = self._calculate_predictions(
            trainset.build_testset(), model_dict, global_rating
        )
        test_predictions = self._calculate_predictions(
            testset, model_dict, global_rating
        )

        train_pred_met, train_rank_met = evaluate_model(
            train_predictions, k=k, threshold=th
        )
        test_pred_met, test_rank_met = evaluate_model(
            test_predictions, k=k, threshold=th
        )

        display_eval_metrics(
            train_pred_met, train_rank_met, test_pred_met, test_rank_met
        )

        return {**train_rank_met, **train_pred_met}, {**test_rank_met, **test_pred_met}

    def recommend(self, top_n: Optional[int] = None, threshold: int = 0) -> None:
        """
        Display the top-N products by Bayesian or average rating.

        Parameters
        ----------
        top_n : int, optional
            Number of products to return. Defaults to all.
        threshold : int
            Minimum number of ratings required (default 0).

        Raises
        ------
        ValueError
            If compute_scores() has not been called yet.
        """
        if self.scores is None:
            raise ValueError("Call compute_scores() before recommend().")

        recommendations = self.scores[self.scores["cnt_rating"] > threshold]
        top_items = (
            recommendations[self.key_column].nlargest(top_n).round(2).reset_index()
        )
        top_items.columns = ["prod_id", "estimated_ratings"]

        smart_display("Top Recommendations (All Users)")
        smart_display_df(top_items)

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_predictions(
        data: Dataset, model_dict: Dict[str, float], global_rating: float
    ) -> List[Prediction]:
        """
        Build predictions using rank scores, falling back to global average.

        Parameters
        ----------
        data :
            Iterable of (user_id, item_id, true_rating) triples.
        model_dict : dict
            item_id -> rank score mapping.
        global_rating : float
            Global average used when an item is unseen.

        Returns
        -------
        list
            List of (uid, iid, true_r, est, details) tuples.
        """
        predictions = []
        for user_id, item_id, true_r in data:
            score = model_dict.get(item_id, global_rating)
            predictions.append((user_id, item_id, true_r, score, {}))
        return predictions


# ---------------------------------------------------------
# END OF MODULE
# ---------------------------------------------------------
