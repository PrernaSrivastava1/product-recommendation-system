# 🛒 Building a Smart Product Recommendation Engine (From Scratch!)

Hey there! 👋 Welcome to my recommendation system project. 

I built this engine to pull back the curtain on how e-commerce giants like Amazon decide which products to recommend to you. Rather than just wrapping a library, I wanted to understand the math, the trade-offs, and the engineering challenges of building, evaluating, and serving recommendation models at scale.

To bring the machine learning models to life, I also built a **sleek, dark glassmorphic web dashboard** using Streamlit, so you can interactively explore recommendations, see Explainable AI (XAI) confidence scores, and query the catalog in plain English.

---

## 🚀 The Live Dashboard
If you want to skip the code and see it in action, here is a preview of the dashboard interface. It runs locally and supports:
*   **Personalized user recommendations** with SVD Matrix Factorization.
*   **Explainable AI (XAI)** breakdowns showing why a product was recommended.
*   **Natural Language Catalog Search** (e.g., typing "trending" or "best rated under 4 stars").
*   **Deep user/product analytics** with interactive charts.

---

## 🧠 Why This Project? (The Core Challenges)

Recommendation systems in the real world are notoriously difficult. When I dug into the Amazon reviews dataset, I hit two classic industry problems immediately:

1.  **The "Positivity Bias"**: Almost 60% of all ratings in the dataset are 5-star reviews. People generally only review things they bought and liked. I had to ensure my models didn't just recommend top-rated items to everyone.
2.  **Sparsity (99.27%)**: In a matrix of users and products, 99.27% of the cells are empty. Recommending products when you have almost zero interaction data is like finding a needle in a haystack.

Here is how I tackled these problems across different models:

*   **Bayesian Average vs. Simple Mean**: If a product has a single 5-star review, a simple average makes it a "perfect 5.0". I implemented a Bayesian average that pulls ratings toward the global mean when review counts are low, ensuring new or rarely reviewed items don't artificially skew the rankings.
*   **Matrix Factorization (SVD)**: I trained an SVD model to decompose the massive sparse user-product matrix into lower-dimensional latent features, capturing the hidden preferences of users.
*   **The Hybrid Solution**: Collaborative filtering fails when a user has very few ratings (the Cold-Start problem). To fix this, I designed a Hybrid recommender that blends SVD personalization (60%) with Bayesian popularity rankings (40%) to guarantee robust fallbacks.

---

## 🤖 The Models I Compared

I implemented and benchmarked five different strategies to find the best balance between speed, personalization, and accuracy:

| Model | Approach | Why I Used It |
| :--- | :--- | :--- |
| **Rank-Based** | Bayesian Average Rating | Perfect for new users where we have zero history (solves Cold Start). |
| **User-User CF** | Cosine Similarity KNN | Finds users who shop like you and recommends what they bought. |
| **Item-Item CF** | Cosine Similarity KNN | Recommends items similar to what you've highly rated in the past. |
| **SVD ⭐** | Matrix Factorization | Learns latent features. This was my best-performing model (RMSE: **0.898**). |
| **Hybrid** | Blended Ensemble | Blends SVD predictions and Bayesian popularity. The most practical for production. |

---

## 📊 How I Evaluated Them

To make sure my recommendations were actually good, I built a custom evaluation pipeline that calculates:
*   **Predictive Accuracy**: Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).
*   **Ranking Quality (Top-10)**: Mean Average Precision (MAP), Mean Reciprocal Rank (MRR), Precision@K, Recall@K, and Hit Rate (did we get at least one recommendation right?).

### Leaderboard Results (Test Set, K=10)
My SVD implementation outperformed the neighborhood-based KNN methods, yielding a **13.7% improvement in RMSE** over Item-Item collaborative filtering:

*   **Best RMSE**: **0.898** (SVD)
*   **Best Precision@10**: **85.1%** (User-User CF)
*   **Recall@10**: **93.0%** (Rank-Based)
*   **Hit Rate@10**: **99.4%** (All models)

---

## 📁 Repository Structure

Here's how I organized the codebase to keep it clean and modular:

```
product-recommendation-system/
│
├── .streamlit/                  # Dashboard configuration
├── data/
│   ├── raw/                     # Original 13-part reviews dataset (~330MB)
│   └── processed/               # Cleaned, filtered pickles ready for training
│
├── models/
│   └── final_model_svd.pkl      # Pre-trained SVD model for instant loading
│
├── notebooks/
│   └── product_recommendation_system.ipynb  # My scratchpad & exploratory analysis
│
├── reports/figures/             # Visualizations saved during evaluation
│
├── src/                         # Modular backend python scripts
│   ├── rank_recommender.py      # Bayesian ranking logic
│   ├── cf_recommender.py        # Collaborative filtering wrapper
│   ├── hybrid_recommender.py    # Hybrid blending math
│   └── model_eval_functions.py  # Precision@K, MAP, RMSE math
│
├── app.py                       # ← The Streamlit Dashboard
├── run_project.py               # ← Train all models from scratch
├── demo.py                      # ← Command Line interface demo
└── requirements.txt
```

---

## 🛠️ Running it Locally

### 1. Clone and Install
Make sure you have Python 3.10 or newer installed:
```bash
git clone <your-repo-url>
cd product-recommendation-system
pip install -r requirements.txt
```

### 2. Launch the Web Dashboard
```bash
python -m streamlit run app.py
```
This will open the dashboard in your browser at `http://localhost:8501`. 

### 3. Run the CLI Demo
If you prefer the terminal, you can get instant recommendations for a user (under 1 second using the saved SVD model):
```bash
# Get recommendations for a random user
python demo.py

# Get top-5 recommendations for a specific user ID
python demo.py --user A3BMUBUC1N77U8 --top 5
```

### 4. Retrain the Models
Want to run the whole training pipeline? Run this command:
```bash
python run_project.py
```
It will load the dataset, retrain all 5 models, plot EDA charts, and save the best-performing SVD weights.

---

## 💡 What I Learned
*   **Collaborative Filtering is heavy**: Neighborhood-based methods (User-User/Item-Item KNN) struggle to scale because they compute similarities on the fly. SVD is much lighter to serve because it pre-computes user/item embeddings.
*   **Evaluation is multi-dimensional**: A model with the lowest RMSE (rating error) isn't always the one users find most engaging. Precision@K and Hit Rate are often more representative of real-world success.
*   **Explainability matters**: Showing users *why* they received a recommendation (e.g. "90% match based on item similarity") drastically increases trust and click-through rates.

---

## 📄 License
This project is licensed under the MIT License — see [LICENSE.txt](LICENSE.txt) for details. Feel free to use the code or dashboard templates for your own projects!

*Thanks for checking out my work! If you have any questions or feedback, feel free to reach out.*
