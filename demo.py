#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
demo.py
=======
Instant recommendation demo for the Amazon Product Recommendation System.

Loads the pre-trained SVD model and generates personalised product
recommendations without any retraining. Ideal for showcasing the project
in an interview, on your resume, or to stakeholders.

Usage
-----
    python demo.py                                   # random user, top 10
    python demo.py --user A3BMUBUC1N77U8             # specific user, top 10
    python demo.py --user A3BMUBUC1N77U8 --top 5    # specific user, top 5
    python demo.py --list-users                      # list 20 sample user IDs
    python demo.py --charts                          # open all EDA charts
    python demo.py --user A3BMUBUC1N77U8 --charts   # recommendations + charts
"""

import argparse
import os
import pickle
import platform
import random
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

# --- Path setup ------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "final_model_svd.pkl")
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "processed", "processed_data.pkl")
FIGURES_DIR  = os.path.join(PROJECT_ROOT, "reports", "figures")

# Ordered list of charts with friendly titles
CHARTS = [
    ("rating_distribution.png",       "Rating Distribution"),
    ("reviews_per_user.png",           "Reviews per User (log scale)"),
    ("reviews_per_product.png",        "Reviews per Product (log scale)"),
    ("user_product_interactions.png",  "User & Product Interaction Distribution"),
    ("model_performance_comparison.png","Model Performance Comparison"),
]


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def banner() -> None:
    print("\n" + "=" * 60)
    print("  Amazon Product Recommendation System")
    print("  Powered by SVD Matrix Factorisation")
    print("=" * 60 + "\n")


def check_model() -> bool:
    """Return True if the saved model exists, else guide user to train first."""
    if not os.path.isfile(MODEL_PATH):
        print("!  No saved model found.")
        print(f"   Expected: {MODEL_PATH}")
        print("\n   Train the models first by running:")
        print("     python run_project.py\n")
        return False
    return True


def load_model() -> tuple:
    """Load the saved SVD model and training data from disk."""
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["train_df"]


def load_full_data() -> pd.DataFrame:
    """Load the full processed dataset."""
    with open(DATA_PATH, "rb") as f:
        return pickle.load(f)


def get_recommendations(df: pd.DataFrame, algo, user_id: str, top_n: int) -> pd.DataFrame:
    """
    Return top-N product recommendations for a user.
    Only considers products the user has NOT already rated.
    """
    rated_products = set(df[df["user_id"] == user_id]["prod_id"].unique())
    all_products   = set(df["prod_id"].unique())
    candidates     = list(all_products - rated_products)

    if not candidates:
        return pd.DataFrame(columns=["Rank", "Product ID", "Predicted Rating"])

    predictions = []
    for prod_id in candidates:
        pred = algo.predict(user_id, prod_id)
        predictions.append((prod_id, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)

    recs = pd.DataFrame(predictions[:top_n], columns=["Product ID", "Predicted Rating"])
    recs["Predicted Rating"] = recs["Predicted Rating"].round(2)
    recs.insert(0, "Rank", range(1, len(recs) + 1))
    return recs


def show_user_history(df: pd.DataFrame, user_id: str, n: int = 5) -> None:
    """Print the user's highest-rated products."""
    history = (
        df[df["user_id"] == user_id]
        .sort_values("rating", ascending=False)
        .head(n)
    )
    if history.empty:
        print("  (No rating history found for this user)")
        return

    print(f"  Recent ratings by {user_id} (top {n}):")
    print(f"  {'Product ID':<15} {'Stars':>7}")
    print("  " + "-" * 24)
    for _, row in history.iterrows():
        stars = "*" * int(row["rating"]) + "-" * (5 - int(row["rating"]))
        print(f"  {row['prod_id']:<15} {stars}")
    print()


# ---------------------------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------------------------

def show_charts_matplotlib() -> None:
    """
    Display all EDA and model charts in a single matplotlib figure window.
    Generates and opens an interactive popup with all 5 charts in a grid.
    """
    # Collect available charts
    available = [(fn, title) for fn, title in CHARTS
                 if os.path.isfile(os.path.join(FIGURES_DIR, fn))]

    if not available:
        print("  No charts found. Run 'python run_project.py' first to generate them.")
        return

    n = len(available)
    cols = 2
    rows = (n + 1) // cols

    # Switch to interactive backend for popup display
    import importlib
    try:
        matplotlib.use("TkAgg")
    except Exception:
        try:
            matplotlib.use("Qt5Agg")
        except Exception:
            matplotlib.use("WXAgg")

    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 6))
    fig.suptitle("Amazon Product Recommendation System - EDA & Model Results",
                 fontsize=16, fontweight="bold", y=1.01)
    axes = axes.flatten()

    for idx, (fn, title) in enumerate(available):
        path = os.path.join(FIGURES_DIR, fn)
        img  = mpimg.imread(path)
        axes[idx].imshow(img)
        axes[idx].set_title(title, fontsize=12, fontweight="bold", pad=8)
        axes[idx].axis("off")

    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    print(f"\n  [OK] Opening {n} charts in a window...")
    print("       Close the window to continue.\n")
    plt.show()


def open_charts_native() -> None:
    """
    Open each chart with the OS default image viewer (Photos / Preview / eog).
    Falls back silently if the viewer is unavailable.
    """
    available = [(fn, title) for fn, title in CHARTS
                 if os.path.isfile(os.path.join(FIGURES_DIR, fn))]

    if not available:
        print("  No charts found. Run 'python run_project.py' first.")
        return

    system = platform.system()
    print(f"\n  Opening {len(available)} chart(s) in your default image viewer...")

    for fn, title in available:
        path = os.path.join(FIGURES_DIR, fn)
        print(f"    -> {title}")
        try:
            if system == "Windows":
                os.startfile(path)
            elif system == "Darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
            time.sleep(0.4)   # Small delay so windows don't stack instantly
        except Exception as e:
            print(f"       Could not open {fn}: {e}")

    print()


def print_chart_paths() -> None:
    """Print chart file paths so the user can open them manually."""
    print("\n  Charts saved at:")
    for fn, title in CHARTS:
        path = os.path.join(FIGURES_DIR, fn)
        exists = "[OK]" if os.path.isfile(path) else "[--]"
        print(f"    {exists}  {title}")
        print(f"         {path}")
    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Instant demo - Amazon Product Recommendation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--user", type=str, default=None,
        help="User ID to generate recommendations for (default: random)",
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="Number of recommendations to show (default: 10)",
    )
    parser.add_argument(
        "--list-users", action="store_true",
        help="Print 20 sample user IDs and exit",
    )
    parser.add_argument(
        "--charts", action="store_true",
        help="Open all EDA and model performance charts",
    )
    parser.add_argument(
        "--charts-native", action="store_true",
        help="Open charts in the OS default image viewer (Photos/Preview)",
    )
    args = parser.parse_args()

    banner()

    # ── Charts-only mode ---------------------------------------------------
    if args.charts and not args.user and not args.list_users:
        print("  Loading charts...\n")
        show_charts_matplotlib()
        print_chart_paths()
        return

    if args.charts_native and not args.user and not args.list_users:
        open_charts_native()
        print_chart_paths()
        return

    # ── Need model for recommendations & list-users -------------------------
    if not check_model():
        sys.exit(1)

    print("  Loading model and data...")
    t0 = time.time()
    algo, train_df = load_model()
    df = load_full_data()
    print(f"  [OK] Loaded in {time.time() - t0:.2f}s\n")

    # ── --list-users --------------------------------------------------------
    if args.list_users:
        sample_users = random.sample(
            df["user_id"].unique().tolist(),
            min(20, df["user_id"].nunique())
        )
        print("  Sample User IDs (copy one and use with --user):\n")
        for uid in sample_users:
            n_r = len(df[df["user_id"] == uid])
            print(f"    {uid}  ({n_r} ratings)")
        print()
        return

    # ── Choose user ---------------------------------------------------------
    all_users = df["user_id"].unique().tolist()

    if args.user:
        if args.user not in all_users:
            print(f"  [ERROR] User '{args.user}' not found in dataset.")
            print("     Run  python demo.py --list-users  to see valid IDs.\n")
            sys.exit(1)
        user_id = args.user
        print(f"  [USER] User: {user_id}")
    else:
        active_users = (
            df.groupby("user_id")["rating"].count()
            .where(lambda x: x >= 5)
            .dropna()
            .index.tolist()
        )
        user_id = random.choice(active_users) if active_users else random.choice(all_users)
        print(f"  [USER] Random user selected: {user_id}")

    n_ratings = len(df[df["user_id"] == user_id])
    print(f"  [INFO] Products rated: {n_ratings}\n")

    # Show history
    show_user_history(df, user_id)

    # ── Generate recommendations --------------------------------------------
    print(f"  Generating top-{args.top} recommendations...")
    t0 = time.time()
    recs = get_recommendations(df, algo, user_id, args.top)
    elapsed = time.time() - t0

    if recs.empty:
        print("  This user has already rated all products - no new recommendations.\n")
        return

    print(f"  [OK] Generated in {elapsed:.2f}s\n")
    print("-" * 60)
    print(f"  Top-{args.top} Recommended Products for {user_id}")
    print("-" * 60)
    print(recs.to_string(index=False))
    print("-" * 60)

    # ── Show charts if requested alongside recommendations ------------------
    if args.charts:
        show_charts_matplotlib()
    elif args.charts_native:
        open_charts_native()

    print_chart_paths()
    print(f"\n  Tips:")
    print(f"    python demo.py --charts                    open all charts in a window")
    print(f"    python demo.py --charts-native             open charts in Photos/Preview")
    print(f"    python demo.py --user <id> --charts        recs + charts together")
    print(f"    python demo.py --list-users                see valid user IDs\n")


if __name__ == "__main__":
    main()
