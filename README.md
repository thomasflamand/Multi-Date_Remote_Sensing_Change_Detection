# 🛰️ Multi-Date Remote Sensing Change Detection

> **Kaggle competition** held as part of the *2EL1730 Machine Learning* course at CentraleSupélec.  
> Team: **LMTPE** — Group project, February 2024. Code cleaned up and refactored post-submission.

---

## 📌 Overview

This project focuses on detecting changes in geographical areas using multi-date satellite imagery. The goal is to classify each area into one of **6 categories** based on urban and geographical features, using machine learning techniques.

<img width="499" height="640" alt="change detection classes" src="https://github.com/user-attachments/assets/d12e00c4-0e9f-470e-9a4a-db6168da942c" />

---

## 🏷️ Classes

| Label | Class |
|-------|-------|
| 0 | Demolition |
| 1 | Road |
| 2 | Residential |
| 3 | Commercial |
| 4 | Industrial |
| 5 | Mega Projects |

---

## 📂 Data

Each instance represents an **irregular polygon** with:
- **Categorical status** across 5 different dates (e.g., under construction → completed)
- **Neighbourhood urban features** (density, industrial zones, etc.)
- **Geographical features** (proximity to rivers, hills, etc.)

The dataset is derived from the [QFabric](https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/html/Verma_QFabric_Multi-Task_Change_Detection_Dataset_CVPRW_2021_paper.html) multi-task change detection dataset (CVPR 2021).

---

## ⚙️ Pipeline

### 1. Preprocessing
- Parse and flatten `.geojson` files
- One-hot encode multi-valued categorical columns (urban & geographic features)
- Extract geometry features: polygon area, perimeter, compactness ratio
- Engineer temporal features: status encoding across 5 dates, days between consecutive dates

### 2. Feature Engineering
- Generate features from polygon geometry (area, perimeter)
- Encode categorical variables (one-hot encoding)
- Compute temporal differences between dates
- Dimensionality reduction and feature selection explored

### 3. Model Selection & Training
We started from a simple **k-Nearest Neighbors (k-NN)** baseline (~40% F1) and progressively moved toward more expressive models.

Linear approaches such as **Logistic Regression** were quickly discarded. The decision boundaries between the six land-use classes are inherently non-linear, making linear models a poor fit for this task.

Tree-based ensemble methods proved significantly more effective:

- **Random Forest** provided a solid improvement thanks to its ability to capture complex feature interactions.
- **XGBoost** was ultimately selected as the final model.

XGBoost’s gradient boosting strategy — sequentially correcting residual errors — is particularly well-suited for tabular, heterogeneous data like ours (a mix of geometric features, one-hot encoded categorical variables, and temporal indicators).

Compared to bagging methods, it:
- Handles class imbalance more robustly  
- Is more resilient to noisy labels  
- Includes built-in regularization (L1/L2) that helped prevent overfitting on our relatively small dataset (Train: (296146, 104),  Test: (120526, 104))

### Hyperparameter Tuning

Hyperparameters were optimized via cross-validation, including:

- Learning rate  
- Maximum tree depth  
- Number of estimators  
- Subsample ratio  

### 4. Evaluation

**Metric:** Mean F1-Score (macro average across all 6 classes).

---

## 📊 Results

- Baseline k-NN: ~40% Mean F1-Score
- After feature engineering and model tuning: significant improvement
- Detailed experiments, feature combinations, and model comparisons are documented in the code

---

## 🗂️ Project Structure

```
.
├── data/
│   ├── raw/
│   │   ├── train.geojson
│   │   └── test.geojson
│   ├── processed/
│   │   ├── train_processed.csv
│   │   └── test_processed.csv
│   └── submission/
│       └── submission_LMTPE.csv
├── notebooks/
│   └── Driver.ipynb        # Main pipeline notebook
└── src/
    ├── preprocessing.py    # Feature engineering & data preprocessing
    └── modeling.py         # Model training, tuning & inference
```

---

## 🚀 How to Run

### On Google Colab (recommended)

1. Mount your Google Drive and make sure the project is at:
   ```
   /content/drive/MyDrive/Colab Notebooks/LMTPE_Kaggle/
   ```
2. Open `notebooks/Driver.ipynb` in Colab
3. Install dependencies if needed:
   ```python
   !pip -q install geopandas shapely pyproj fiona xgboost
   ```
4. Run all cells — the notebook handles preprocessing and modeling end to end

### Output

The final submission file is saved to:
```
data/submission/submission_LMTPE.csv
```

---

## 📦 Dependencies

- `geopandas`, `shapely`, `pyproj`, `fiona` — geospatial data processing
- `scikit-learn` — ML models and cross-validation
- `xgboost` — gradient boosting
- `pandas`, `numpy` — data manipulation

---

## 📚 Reference

```bibtex
@InProceedings{Verma_2021_CVPR,
  author    = {Verma, Sagar and Panigrahi, Akash and Gupta, Siddharth},
  title     = {QFabric: Multi-Task Change Detection Dataset},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month     = {June},
  year      = {2021},
  pages     = {1052--1061}
}
```

*Competition hosted by: nouzir — 2EL1730 Machine Learning Project, CentraleSupélec, Jan. 2024*
