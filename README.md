# 🛒 Amazon Product Recommendation System

> A multi-algorithm machine learning system that predicts what Amazon customers will love — built with real-world rating data, clean modular code, and production-ready design patterns.

---

## 📌 Project Overview

I built this recommendation system to explore how large e-commerce platforms like Amazon decide what to surface to each user. The dataset contains **~7.8 million raw ratings** across Amazon product categories, filtered down to a meaningful subset of **78,798 interactions** between **1,992 users** and **5,402 products**.

The project implements five recommendation strategies — from simple popularity ranking all the way to matrix factorisation and a hybrid ensemble — and evaluates all of them using both predictive accuracy (RMSE) and ranking quality (Precision@K, MAP, Hit Rate@K).

---

## 🎯 Key Highlights

- **5 recommendation algorithms** in a unified, extensible framework
- **Custom evaluation pipeline** with 7 metrics: RMSE, Precision@K, Recall@K, F1@K, MRR, MAP, Hit Rate@K
- **Hybrid recommender** blends collaborative filtering with Bayesian popularity scores to handle cold-start items
- **Works in both Jupyter notebooks and terminal** — no IPython rendering artifacts
- **Pre-trained SVD model** included for instant demo without retraining
- **CLI demo tool** — get recommendations in under 1 second

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   DATA PIPELINE                     │
│  Raw CSVs → Cleaning → Filtering → processed_data   │
└─────────────────────────┬───────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │     TRAIN / TEST SPLIT   │
              │    80% train / 20% test  │
              └────────────┬────────────┘
                           │
         ┌─────────────────┴──────────────────┐
         │                                    │
┌────────▼────────┐                  ┌────────▼────────┐
│  Rank-Based     │                  │  Surprise CF    │
│  (Bayesian Avg) │                  │  (kNN / SVD)    │
└────────┬────────┘                  └────────┬────────┘
         │                                    │
         └──────────────┬─────────────────────┘
                        │
               ┌────────▼────────┐
               │  Hybrid System  │
               │  CF×0.6 + Rank×0.4│
               └────────┬────────┘
                        │
               ┌────────▼────────┐
               │   EVALUATION    │
               │  RMSE, P@K,MAP  │
               └─────────────────┘
```

---

## 🤖 Models Implemented

| Model | Algorithm | Key Idea |
|-------|-----------|----------|
| **Rank-Based** | Bayesian Average | Surfaces globally popular, well-rated products |
| **User-User CF** | KNNBasic (cosine) | Finds users with similar taste and borrows their ratings |
| **Item-Item CF** | KNNBasic (cosine) | Recommends products similar to what a user already liked |
| **SVD** | Matrix Factorisation | Discovers hidden latent factors in user-item interactions |
| **Hybrid** | CF + Bayesian Blend | Combines personalisation with popularity for best-of-both |

---

## 📊 Dataset

- **Source**: Amazon Product Reviews (public dataset)
- **Raw size**: ~7.8 million ratings across 13 CSV files
- **Processed size**: 78,798 interactions (users with ≥ 5 ratings)
- **Sparsity**: 99.27% — a realistic, challenging recommendation scenario
- **Rating scale**: 1–5 stars (mean: 4.28 — shows positive rating bias typical of Amazon)

---

## 📁 Project Structure

```
product-recommendation-system/
│
├── data/
│   ├── raw/                     # Original 13-part CSV files (~330 MB total)
│   └── processed/
│       └── processed_data.pkl   # Cleaned, filtered dataset ready for modelling
│
├── models/
│   └── final_model_svd.pkl      # Pre-trained SVD model (skip retraining)
│
├── notebooks/
│   └── product_recommendation_system.ipynb  # Full EDA + modelling walkthrough
│
├── reports/
│   └── figures/                 # Auto-generated EDA charts (rating dist., etc.)
│
├── src/
│   ├── __init__.py
│   ├── utils.py                 # Smart display, CSV loader, metric formatters
│   ├── eda_functions.py         # EDA plotting utilities
│   ├── rank_recommender.py      # Bayesian/average rank-based recommender
│   ├── cf_recommender.py        # kNN + SVD collaborative filtering
│   ├── hybrid_recommender.py    # Weighted CF + rank hybrid system
│   └── model_eval_functions.py  # RMSE, Precision@K, MAP, MRR, Hit Rate@K
│
├── run_project.py               # ← Full pipeline: trains all 5 models end-to-end
├── demo.py                      # ← Instant CLI demo using saved SVD model
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone and navigate

```bash
git clone <your-repo-url>
cd product-recommendation-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Python ≥ 3.10** required. Tested on Python 3.11 and 3.14.

### 3. Run the full pipeline

```bash
python run_project.py
```

This will:
- Load the processed dataset (78K interactions)
- Train all 5 recommendation models
- Print evaluation metrics for each model
- Save EDA charts to `reports/figures/`
- Save the best SVD model to `models/`

**Expected runtime**: ~5–10 minutes (KNN models dominate training time)

### 4. Run the instant demo

```bash
# Random user — top 10 recommendations
python demo.py

# Specific user
python demo.py --user A3BMUBUC1N77U8

# Specific user, top 5
python demo.py --user A3BMUBUC1N77U8 --top 5

# See a list of valid user IDs
python demo.py --list-users
```

**Expected runtime**: < 1 second (uses pre-trained model)

### 5. Explore the notebook

```bash
jupyter notebook notebooks/product_recommendation_system.ipynb
```

The notebook contains the full EDA, step-by-step modelling, and visualisations.

---

## 🔧 Requirements

```
numpy>=1.26.0
pandas>=2.0.0
scikit-learn>=1.4.0
scikit-surprise>=1.1.3
scipy>=1.12.0
matplotlib>=3.8.0
seaborn>=0.13.0
statsmodels>=0.14.0
jupyter>=1.0.0
ipython>=8.0.0
joblib>=1.4.0
pillow>=10.0.0
```

---

## 🧠 Key Design Decisions

**Why Bayesian Average instead of plain average?**
Products with only 1–2 ratings can have a perfect 5.0 score, which artificially inflates their ranking. Bayesian average pulls ratings toward the global mean proportionally to how few ratings a product has — much fairer.

**Why SVD over kNN for the final model?**
kNN is memory-based and scales poorly to millions of users. SVD learns compressed latent representations that generalise better and can be deployed cheaply at scale.

**Why a Hybrid system?**
Pure collaborative filtering struggles when a user has very few interactions (cold-start). Adding a rank-based component ensures the system always has a reasonable fallback — blending 60% CF personalisation with 40% popularity signal.

---

## 📈 Evaluation Metrics

The project uses two families of metrics:

**Predictive Quality** — how accurate are the rating estimates?
- **RMSE** — root mean squared error between predicted and true ratings

**Ranking Quality** — are relevant items actually at the top of the list?
- **Precision@K** — fraction of top-K recommendations that are truly relevant
- **Recall@K** — fraction of relevant items that appear in top-K
- **F1@K** — harmonic mean of Precision and Recall
- **MRR** — Mean Reciprocal Rank of the first relevant item
- **MAP** — Mean Average Precision across relevant items
- **Hit Rate@K** — proportion of users who got at least one good recommendation

---

## 🔮 Future Improvements

- [ ] Add content-based filtering using product metadata (category, description)
- [ ] Implement Neural Collaborative Filtering (NCF) with PyTorch
- [ ] Build a Streamlit web dashboard for interactive exploration
- [ ] Add A/B testing framework to compare model variants online
- [ ] Deploy as a REST API with FastAPI + Docker

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE.txt](LICENSE.txt) for details.

---

*Built as part of a machine learning portfolio project exploring production-grade recommendation system design.*
