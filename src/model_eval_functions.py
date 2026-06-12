"""
model_eval_functions.py

Evaluation, recommendation, and analysis functions for the Amazon Product
Recommendation System.

Supports computing ranking/quality metrics, generating personalized
recommendations, and running baseline grid searches ? compatible with both
Jupyter notebooks and terminal execution.

Modules
-------
1. IMPORT LIBRARIES
2. METRIC FUNCTIONS
3. RECOMMENDATION FUNCTIONS
4. BASELINE GRID SEARCH
"""

# ---------------------------------------------------------
# 1. IMPORT LIBRARIES
# ---------------------------------------------------------

from collections import defaultdict, namedtuple

import numpy as np
import pandas as pd

from src.utils import smart_display, smart_display_df

# Import model evaluation utilities from Surprise
from surprise import accuracy
from surprise.model_selection import GridSearchCV

# Namedtuple that mirrors Surprise's internal Prediction format so that
# raw (uid, iid, true_r, est, details) tuples can be passed to accuracy.*
_Prediction = namedtuple("Prediction", ["uid", "iid", "r_ui", "est", "details"])


def _ensure_predictions(predictions: list) -> list:
    """
    Ensure predictions are proper namedtuples that Surprise's accuracy
    functions understand, regardless of whether they were passed as plain
    tuples or Prediction namedtuples.
    """
    result = []
    for p in predictions:
        if isinstance(p, _Prediction):
            result.append(p)
        elif isinstance(p, tuple) and len(p) == 5:
            result.append(_Prediction(*p))
        else:
            result.append(p)  # Let Surprise raise its own error if malformed
    return result


# ---------------------------------------------------------
# 2. METRIC FUNCTIONS
# ---------------------------------------------------------

def calculate_predictive_metrics(
    user_est_true: dict, k: int = 10, threshold: float = 3.5
) -> dict:
    """
    Compute Precision@K, Recall@K, and F1 Score@K for a recommendation model.

    Parameters
    ----------
    user_est_true : dict
        Maps user ID to list of (estimated_rating, true_rating) tuples.
    k : int
        Top-K cut-off (default 10).
    threshold : float
        Minimum rating to consider an item relevant (default 3.5).

    Returns
    -------
    dict
        {"Precision@K": ..., "Recall@K": ..., "F1 Score@K": ...}
    """
    precisions: dict = {}
    recalls: dict = {}

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        n_rel = sum(true_r >= threshold for _, true_r in user_ratings)
        n_rec_k = sum(est >= threshold for est, _ in user_ratings[:k])
        n_rel_and_rec_k = sum(
            (true_r >= threshold) and (est >= threshold)
            for est, true_r in user_ratings[:k]
        )

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    precision = round(sum(precisions.values()) / len(precisions), 3)
    recall = round(sum(recalls.values()) / len(recalls), 3)
    f1 = (
        round((2 * precision * recall) / (precision + recall), 3)
        if (precision + recall) != 0
        else 0
    )

    return {"Precision@K": precision, "Recall@K": recall, "F1 Score@K": f1}


def calculate_ranking_metrics(
    user_est_true: dict, k: int = 10, threshold: float = 3.5
) -> dict:
    """
    Compute MRR, MAP, and Hit Rate@K for a recommendation model.

    Parameters
    ----------
    user_est_true : dict
        Maps user ID to list of (estimated_rating, true_rating) tuples.
    k : int
        Top-K cut-off (default 10).
    threshold : float
        Minimum rating to consider an item relevant (default 3.5).

    Returns
    -------
    dict
        {"MRR": ..., "MAP": ..., "Hit Rate@K": ...}
    """
    mrr_sum = 0.0
    map_sum = 0.0
    hit_rate_count = 0
    num_users = len(user_est_true)

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # MRR ? reciprocal rank of first relevant item
        for rank, (est, true_r) in enumerate(user_ratings[:k], start=1):
            if true_r >= threshold:
                mrr_sum += 1 / rank
                break

        # MAP ? average precision over relevant items
        hits = 0
        avg_precision = 0.0
        for rank, (est, true_r) in enumerate(user_ratings[:k], start=1):
            if true_r >= threshold:
                hits += 1
                avg_precision += hits / rank

        if hits > 0:
            map_sum += avg_precision / hits
            hit_rate_count += 1

    return {
        "MRR": round(mrr_sum / num_users, 3),
        "MAP": round(map_sum / num_users, 3),
        "Hit Rate@K": round(hit_rate_count / num_users, 3),
    }


def evaluate_model(predictions: list, k: int = 10, threshold: float = 3.5) -> tuple:
    """
    Evaluate a recommendation model with both predictive and ranking metrics.

    Parameters
    ----------
    predictions : list
        List of predictions (Prediction namedtuples or plain 5-tuples).
    k : int
        Top-K cut-off (default 10).
    threshold : float
        Relevance threshold (default 3.5).

    Returns
    -------
    tuple
        (predictive_metrics dict, ranking_metrics dict)
    """
    predictions = _ensure_predictions(predictions)

    # RMSE using Surprise's built-in function
    rmse = round(accuracy.rmse(predictions, verbose=False), 3)

    # Build per-user (est, true_r) lists
    user_est_true: dict = defaultdict(list)
    for pred in predictions:
        user_est_true[pred.uid].append((pred.est, pred.r_ui))

    predictive_metrics = calculate_predictive_metrics(user_est_true, k, threshold)
    predictive_metrics = {"RMSE": rmse, **predictive_metrics}

    ranking_metrics = calculate_ranking_metrics(user_est_true, k, threshold)

    return predictive_metrics, ranking_metrics


# ---------------------------------------------------------
# 3. RECOMMENDATION FUNCTIONS
# ---------------------------------------------------------

def get_recommendations(
    data: pd.DataFrame, algo, user_id: str, top_n: int = None
) -> pd.DataFrame:
    """
    Generate top-N product recommendations for a user using a trained model.

    Parameters
    ----------
    data : pd.DataFrame
        Full interaction DataFrame with columns ['user_id', 'prod_id', 'rating'].
    algo :
        Trained Surprise algorithm instance.
    user_id : str
        Target user ID.
    top_n : int, optional
        Number of recommendations to return. Defaults to all unrated products.

    Returns
    -------
    pd.DataFrame
        Columns: ['prod_id', 'estimated_ratings', 'details']
    """
    user_item_matrix = data.pivot(
        index="user_id", columns="prod_id", values="rating"
    )

    if user_id not in user_item_matrix.index:
        return pd.DataFrame(columns=["prod_id", "estimated_ratings", "details"])

    non_interacted = (
        user_item_matrix.loc[user_id][user_item_matrix.loc[user_id].isnull()]
        .index.tolist()
    )

    recommendations = []
    for item_id in non_interacted:
        prediction = algo.predict(user_id, item_id)
        recommendations.append((item_id, prediction.est, prediction.details))

    recommendations.sort(key=lambda x: x[1], reverse=True)

    if top_n is None:
        top_n = len(non_interacted)

    return pd.DataFrame(
        recommendations[:top_n],
        columns=["prod_id", "estimated_ratings", "details"],
    ).assign(estimated_ratings=lambda df: df["estimated_ratings"].round(2))


# ---------------------------------------------------------
# 4. BASELINE GRID SEARCH
# ---------------------------------------------------------

def baseline_gridsearch(
    data: pd.DataFrame,
    algos: dict,
    param_grids: dict,
    measures: list = None,
    cv: int = 5,
) -> tuple:
    """
    Run GridSearchCV for multiple algorithms for baseline comparison.

    Parameters
    ----------
    data :
        Surprise Dataset object.
    algos : dict
        {name: algo_class} mapping.
    param_grids : dict
        {name: param_grid} mapping.
    measures : list, optional
        Evaluation measures (default ["rmse"]).
    cv : int
        Cross-validation folds (default 5).

    Returns
    -------
    tuple
        (best_model_name, best_model_params)
    """
    if measures is None:
        measures = ["rmse"]

    best_rmse = float("inf")
    best_model_name = None
    best_model_params = None
    models: dict = {}

    for name, algo in algos.items():
        param_grid = param_grids.get(name, {})
        model_gs = GridSearchCV(algo, param_grid, measures=measures, cv=cv, n_jobs=-1)
        model_gs.fit(data)
        models[name] = model_gs

        current_rmse = model_gs.best_score["rmse"]
        if current_rmse < best_rmse:
            best_rmse = current_rmse
            best_model_name = name
            best_model_params = model_gs.best_params["rmse"]

    print_baseline_gs_results(best_model_name=best_model_name, models=models)
    return best_model_name, best_model_params


def print_baseline_gs_results(best_model_name: str, models: dict) -> None:
    """
    Print baseline GridSearch results to the console / notebook.

    Parameters
    ----------
    best_model_name : str
        Name of the winning algorithm.
    models : dict
        {name: GridSearchCV} mapping of all evaluated models.
    """
    smart_display("Baseline GridSearch Results")

    for name, model in models.items():
        print(f"\n  Algorithm : {name}")
        print(f"  RMSE      : {round(model.best_score['rmse'], 3)}")
        print(f"  Params    : {model.best_params['rmse']}")

    width = 80
    msg = f" BASELINE MODEL SELECTION: {best_model_name} ".center(width, "?")
    smart_display(msg, bold=False)


# ---------------------------------------------------------
# END OF MODULE
# ---------------------------------------------------------
