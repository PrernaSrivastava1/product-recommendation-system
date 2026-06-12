"""
hybrid_recommender.py

Hybrid Recommendation System ? blends collaborative filtering (CF) scores
with popularity-based rank scores (Bayesian average) to produce more robust,
personalised recommendations.

The weighted combination helps address the cold-start problem for new items
by falling back to globally popular products when CF confidence is low.

Classes
-------
WeightSumError
    Raised when CF and rank weights do not sum to 1.
ModelNotAvailableError
    Raised when required sub-models are not available.
HybridRecommendationSystem
    Combines a trained CFRecommendationSystem with a RankRecommendationSystem.
"""

# ---------------------------------------------------------
# IMPORT LIBRARIES
# ---------------------------------------------------------

from typing import Dict, List, Tuple, Union

import pandas as pd
from surprise.dataset import Dataset
from surprise.prediction_algorithms.predictions import Prediction
from surprise.prediction_algorithms.algo_base import AlgoBase

from src.model_eval_functions import evaluate_model, get_recommendations
from src.utils import display_eval_metrics, smart_display, smart_display_df
from src.cf_recommender import CFRecommendationSystem
from src.rank_recommender import RankRecommendationSystem

# ---------------------------------------------------------
# EXCEPTIONS
# ---------------------------------------------------------


class WeightSumError(Exception):
    """Raised when CF weight + rank weight ? 1.0."""
    pass


class ModelNotAvailableError(Exception):
    """Raised when a required sub-model has not been trained."""
    pass


# ---------------------------------------------------------
# HYBRID RECOMMENDATION SYSTEM
# ---------------------------------------------------------


class HybridRecommendationSystem:
    """
    Hybrid recommender that combines collaborative filtering with rank-based
    scoring for improved recommendation quality.

    The hybrid score for an item is computed as:
        hybrid_score = weight_cf * cf_score + weight_rank * rank_score

    This ensures that very popular products act as a 'safety net' when the CF
    model has limited interaction data for a user.

    Attributes
    ----------
    model_cf : AlgoBase
        Trained Surprise CF algorithm instance.
    model_rank : pd.DataFrame
        Product-level score DataFrame from RankRecommendationSystem.
    model_rank_dict : dict
        Fast-lookup {prod_id: rank_score} dictionary.
    global_rating : float
        Fallback rank score for unseen products.
    weight_cf : float
        Weight applied to CF scores (default 0.8).
    weight_rank : float
        Weight applied to rank scores (default 0.2).
    algo_name : str
        Combined name label for comparison tables.
    key_column : str
        Column name of the rank score ('bayesian_rating' or 'avg_rating').
    """

    __slots__ = (
        "model_cf", "model_rank", "model_rank_dict", "global_rating",
        "weight_cf", "weight_rank", "algo_name", "key_column",
    )

    def __init__(
        self,
        model_cf: CFRecommendationSystem,
        model_rank: RankRecommendationSystem,
        weight_cf: float = 0.8,
        weight_rank: float = 0.2,
    ):
        """
        Initialise the HybridRecommendationSystem.

        Parameters
        ----------
        model_cf : CFRecommendationSystem
            A trained CFRecommendationSystem instance.
        model_rank : RankRecommendationSystem
            A RankRecommendationSystem with computed scores.
        weight_cf : float
            Weight for CF scores (default 0.8). Must sum to 1 with weight_rank.
        weight_rank : float
            Weight for rank scores (default 0.2). Must sum to 1 with weight_cf.

        Raises
        ------
        WeightSumError
            If weight_cf + weight_rank ? 1.
        ModelNotAvailableError
            If either sub-model is None / untrained.
        """
        self.model_cf = model_cf.model
        self.model_rank = model_rank.scores
        self.key_column = model_rank.key_column
        self.model_rank_dict = self.model_rank[self.key_column].to_dict()
        self.global_rating = self.model_rank[self.key_column].mean()
        self.weight_cf = weight_cf
        self.weight_rank = weight_rank
        self.algo_name = f"{model_cf.algo_name} | {model_rank.algo_name}"

        if abs(self.weight_cf + self.weight_rank - 1) > 1e-9:
            raise WeightSumError(
                f"Weights must sum to 1 (got {weight_cf + weight_rank:.4f})."
            )

        if self.model_cf is None or self.model_rank is None:
            raise ModelNotAvailableError(
                "Both a trained CF model and rank scores are required."
            )

    # ------------------------------------------------------------------
    # HYBRID SCORING
    # ------------------------------------------------------------------

    def compute_hybrid_scores(
        self, data: pd.DataFrame, int_data: Dict[str, List[Tuple[str, float]]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute hybrid scores for each user's candidate products.

        Parameters
        ----------
        data : pd.DataFrame
            Full interaction DataFrame.
        int_data : dict
            {user_id: [(prod_id, rating), ...]}

        Returns
        -------
        dict
            {user_id: {prod_id: hybrid_score, ...}}
        """
        cf_scores: dict = {}
        hybrid_scores: dict = {}

        for user in int_data.keys():
            cf_scores[user] = self._calculate_cf_scores(
                data=data, algo=self.model_cf, user_id=user
            )
            hybrid_scores[user] = {
                item: self._calculate_hybrid_score(cf_score, item)
                for item, cf_score in cf_scores[user].items()
            }

        return hybrid_scores

    # ------------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------------

    def evaluate(
        self, trainset: Dataset, testset: Dataset, k: int = 10, th: float = 3.5
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Evaluate hybrid model on train and test sets.

        Parameters
        ----------
        trainset : Dataset
            Surprise trainset.
        testset : Dataset
            Surprise testset list.
        k : int
            Top-K cut-off (default 10).
        th : float
            Relevance threshold (default 3.5).

        Returns
        -------
        tuple
            (train_metrics dict, test_metrics dict)
        """
        train_predictions = self.model_cf.test(trainset.build_testset())
        hybrid_train_predictions = self._calculate_hybrid_predictions(train_predictions)

        test_predictions = self.model_cf.test(testset)
        hybrid_test_predictions = self._calculate_hybrid_predictions(test_predictions)

        train_pred_met, train_rank_met = evaluate_model(
            hybrid_train_predictions, k=k, threshold=th
        )
        test_pred_met, test_rank_met = evaluate_model(
            hybrid_test_predictions, k=k, threshold=th
        )

        display_eval_metrics(
            train_pred_met, train_rank_met, test_pred_met, test_rank_met
        )

        return {**train_rank_met, **train_pred_met}, {**test_rank_met, **test_pred_met}

    # ------------------------------------------------------------------
    # PREDICTION & RECOMMENDATION
    # ------------------------------------------------------------------

    def predict(
        self,
        int_data: Dict[str, List[Union[Tuple[str, float], str]]],
        has_interacted: bool,
    ) -> None:
        """
        Show hybrid rating predictions for specific user-item pairs.

        Parameters
        ----------
        int_data : dict
            {user_id: [(prod_id, true_rating), ...]} or {user_id: [prod_id, ...]}
        has_interacted : bool
            True for interacted items (includes true rating); False otherwise.
        """
        phrase = "" if has_interacted else "Non-"
        smart_display(f"Hybrid Rating Estimates for {phrase}Interacted Products")

        predictions = []
        for user, interactions in int_data.items():
            for interaction in interactions:
                if has_interacted:
                    iid, rui = interaction
                    cf_pred = self.model_cf.predict(uid=user, iid=iid, r_ui=rui, verbose=False)
                else:
                    iid = interaction
                    cf_pred = self.model_cf.predict(uid=user, iid=iid, verbose=False)

                hybrid_score = self._calculate_hybrid_score(cf_pred.est, iid)
                predictions.append((cf_pred, hybrid_score))

        for cf_pred, hybrid_score in predictions:
            r_ui_str = f"{cf_pred.r_ui:.2f}" if cf_pred.r_ui is not None else "None"
            print(
                f"  user: {cf_pred.uid:<14} item: {cf_pred.iid:<12} "
                f"r_ui = {r_ui_str:<6} hybrid_est = {hybrid_score:.2f}  "
                f"{cf_pred.details}"
            )

    def recommend(
        self,
        data: pd.DataFrame,
        int_data: Dict[str, List[Tuple[str, float]]],
        top_n: int = 5,
    ) -> None:
        """
        Generate and display top-N hybrid recommendations for each user.

        Parameters
        ----------
        data : pd.DataFrame
            Full interaction DataFrame.
        int_data : dict
            {user_id: [(prod_id, rating), ...]}
        top_n : int
            Recommendations per user (default 5).
        """
        hybrid_scores = self.compute_hybrid_scores(data, int_data)

        for user, scores in hybrid_scores.items():
            top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

            recommendations_df = pd.DataFrame(
                top_items, columns=["prod_id", "estimated_ratings"]
            ).round({"estimated_ratings": 2})

            smart_display(f"Hybrid Recommendations for User: {user}")
            smart_display_df(recommendations_df)

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _calculate_cf_scores(
        self, data: pd.DataFrame, algo: AlgoBase, user_id: str
    ) -> Dict[str, float]:
        """Get CF predicted ratings for all unrated items for a user."""
        cf_recs = get_recommendations(data=data, algo=algo, user_id=user_id)
        return dict(zip(cf_recs["prod_id"], cf_recs["estimated_ratings"]))

    def _calculate_hybrid_score(self, cf_score: float, item_id: str) -> float:
        """Blend CF score with the product's rank score."""
        rank_score = self.model_rank_dict.get(item_id, self.global_rating)
        return self.weight_cf * cf_score + self.weight_rank * rank_score

    def _calculate_hybrid_predictions(
        self, predictions: List[Prediction]
    ) -> List[Prediction]:
        """Replace CF estimated ratings with blended hybrid scores."""
        return [
            (pred.uid, pred.iid, pred.r_ui,
             self._calculate_hybrid_score(pred.est, pred.iid), {})
            for pred in predictions
        ]


# ---------------------------------------------------------
# END OF MODULE
# ---------------------------------------------------------
