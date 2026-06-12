# -*- coding: utf-8 -*-
"""
utils.py

Utility functions for data loading, display, and metric formatting in the
Amazon Product Recommendation System.

Handles reading split CSV files, selecting user interactions for evaluation,
and displaying model metrics in a clean, consistent format ? both inside
Jupyter notebooks and from the terminal.

Modules
-------
1. IMPORT LIBRARIES
2. DISPLAY HELPERS
3. DATA LOADING
4. INTERACTION UTILITIES
5. METRICS DISPLAY
"""

# ---------------------------------------------------------
# 1. IMPORT LIBRARIES
# ---------------------------------------------------------

import os
import re
import random

import pandas as pd

# Try importing IPython display utilities; fall back gracefully if unavailable
try:
    from IPython.display import Markdown, display as _ipython_display
    import IPython
    _IN_IPYTHON = True
except ImportError:
    _IN_IPYTHON = False


# ---------------------------------------------------------
# 2. DISPLAY HELPERS
# ---------------------------------------------------------

def _is_jupyter() -> bool:
    """Return True when running inside a Jupyter / IPython kernel."""
    if not _IN_IPYTHON:
        return False
    try:
        shell = IPython.get_ipython()
        return shell is not None and "ZMQInteractiveShell" in type(shell).__name__
    except Exception:
        return False


def smart_display(text: str, bold: bool = True) -> None:
    """
    Display text as bold Markdown inside Jupyter, or as a plain print in terminal.

    Parameters
    ----------
    text : str
        The message to display.
    bold : bool
        If True and running in Jupyter, wrap the text in **bold** markdown.
    """
    if _is_jupyter():
        formatted = f"**{text}**" if bold else text
        _ipython_display(Markdown(formatted))
    else:
        separator = "-" * min(len(text) + 4, 80)
        print(f"\n{separator}")
        print(f"  {text}")
        print(separator)


def smart_display_df(df: pd.DataFrame) -> None:
    """
    Display a DataFrame using IPython rich display in Jupyter, or print in terminal.
    """
    if _is_jupyter():
        _ipython_display(df)
    else:
        print(df.to_string())
        print()


def smart_display_dict(d: dict) -> None:
    """Display a dictionary in Jupyter or as a formatted print in terminal."""
    if _is_jupyter():
        _ipython_display(d)
    else:
        for key, value in d.items():
            print(f"  {key}: {value}")
        print()


# ---------------------------------------------------------
# 3. DATA LOADING
# ---------------------------------------------------------

def read_all_csv_files(
    folder_path: str, header: int | list[int] | None = None
) -> pd.DataFrame:
    """
    Read all CSV files in a folder, sorted in natural order, and concatenate
    them into a single DataFrame.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing CSV files.
    header : int | list[int] | None, optional
        Row number(s) to use as column names. None means no header row.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame from all CSV files found.
    """
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    def _natural_sort_key(s: str) -> list:
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", s)
        ]

    csv_files.sort(key=_natural_sort_key)

    df_list = []
    smart_display("Reading CSV Files")

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, header=header)
        df_list.append(df)
        print(f"  Imported: {file}")

    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"\n  Data import complete ? {len(csv_files)} file(s) loaded.")
    return combined_df


# ---------------------------------------------------------
# 4. INTERACTION UTILITIES
# ---------------------------------------------------------

def select_interactions(
    trainset, num_users: int = 2, num_products: int = 2, seed: int = 42
) -> tuple:
    """
    Randomly select a subset of users and sample their interactions and
    non-interactions from the training set.

    Parameters
    ----------
    trainset :
        Surprise trainset object.
    num_users : int
        Number of users to sample (default 2).
    num_products : int
        Number of products to sample per user (default 2).
    seed : int
        Random seed for reproducibility (default 42).

    Returns
    -------
    tuple
        (user_interactions, user_non_interactions) ? dicts keyed by raw user ID.
    """
    random.seed(seed)

    random_inner_uids = random.sample(trainset.all_users(), num_users)

    user_interactions: dict = {}
    user_interacted_products: dict = {}
    user_non_interactions: dict = {}

    all_products = set(trainset.to_raw_iid(item) for item in trainset.all_items())

    for inner_uid in random_inner_uids:
        raw_user_id = trainset.to_raw_uid(inner_uid)
        user_ratings = trainset.ur[inner_uid]

        user_interacted_products[raw_user_id] = [
            (trainset.to_raw_iid(item_inner_id), rating)
            for item_inner_id, rating in user_ratings
        ]

        user_interactions[raw_user_id] = random.sample(
            user_interacted_products[raw_user_id], num_products
        )

        interacted_products = set(
            item_id for item_id, _ in user_interacted_products[raw_user_id]
        )
        non_interacted_products = list(all_products - interacted_products)

        if len(non_interacted_products) >= num_products:
            user_non_interactions[raw_user_id] = random.sample(
                non_interacted_products, num_products
            )

    smart_display("User Interactions")
    smart_display_dict(user_interactions)

    smart_display("User Non-Interactions")
    smart_display_dict(user_non_interactions)

    return user_interactions, user_non_interactions


# ---------------------------------------------------------
# 5. METRICS DISPLAY
# ---------------------------------------------------------

def display_eval_metrics(
    train_pred_met, train_rank_met, test_pred_met, test_rank_met
) -> None:
    """
    Display predictive and ranking evaluation metrics for both train and test sets.

    Parameters
    ----------
    train_pred_met : dict
        Predictive metrics for the training set.
    train_rank_met : dict
        Ranking metrics for the training set.
    test_pred_met : dict
        Predictive metrics for the test set.
    test_rank_met : dict
        Ranking metrics for the test set.
    """
    pred_metrics_df = pd.DataFrame(
        {"Trainset": train_pred_met, "Testset": test_pred_met}
    )
    rank_metrics_df = pd.DataFrame(
        {"Trainset": train_rank_met, "Testset": test_rank_met}
    )

    smart_display("Predictive Quality Metrics")
    smart_display_df(pred_metrics_df)
    smart_display("Ranking Quality Metrics")
    smart_display_df(rank_metrics_df)


def create_metrics_df(metrics: dict, algo_names: list) -> pd.DataFrame:
    """
    Build a formatted comparison DataFrame from a metrics dictionary.

    Parameters
    ----------
    metrics : dict
        Dict of algorithm name -> metric dict.
    algo_names : list
        List of algorithm display names.

    Returns
    -------
    pd.DataFrame
    """
    metrics_df = pd.DataFrame(metrics).T

    if len(metrics_df) != len(algo_names):
        raise ValueError(
            f"algo_names length ({len(algo_names)}) does not match "
            f"metrics rows ({len(metrics_df)})."
        )

    metrics_df.insert(0, "Algo", algo_names)
    metrics_df.insert(1, "|", ["|"] * len(metrics_df))

    return metrics_df


def prepare_and_display_metrics(metrics: dict, algo_names: list, title: str) -> None:
    """
    Prepare and display a model evaluation metrics table with a title.

    Parameters
    ----------
    metrics : dict
        Dict of algorithm name -> metric dict.
    algo_names : list
        List of algorithm display names.
    title : str
        Section heading to print above the table.
    """
    df = create_metrics_df(metrics, algo_names)
    smart_display(title)
    smart_display_df(df)


# ---------------------------------------------------------
# END OF MODULE
# ---------------------------------------------------------
