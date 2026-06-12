"""
src/__init__.py

Amazon Product Recommendation System ? source package.

Exposes the core recommender classes and utility functions for convenient
importing throughout the project and in the Jupyter notebook.
"""

from src.utils import (
    smart_display,
    read_all_csv_files,
    select_interactions,
    display_eval_metrics,
    create_metrics_df,
    prepare_and_display_metrics,
)
from src.rank_recommender import RankRecommendationSystem
from src.cf_recommender import CFRecommendationSystem
from src.hybrid_recommender import HybridRecommendationSystem
from src.model_eval_functions import evaluate_model, get_recommendations

__all__ = [
    "smart_display",
    "read_all_csv_files",
    "select_interactions",
    "display_eval_metrics",
    "create_metrics_df",
    "prepare_and_display_metrics",
    "RankRecommendationSystem",
    "CFRecommendationSystem",
    "HybridRecommendationSystem",
    "evaluate_model",
    "get_recommendations",
]
