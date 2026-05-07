# Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.9-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.11-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2-yellow)

## 📋 Project Overview

**Objective:** Predict fraudulent credit card transactions using machine learning with extreme class imbalance (0.52% fraud rate).

**Dataset:** 339,607 transactions from Kaggle  
**Context:** Academic project demonstrating practical ML implementation for imbalanced classification  
**Date:** April 2026

---

## 🎯 Business Problem

Credit card fraud detection is a critical challenge in the financial sector where:
- False negatives (missed frauds) are highly costly
- The fraud rate is extremely low (~0.5%)
- Models must balance precision and recall effectively

This project compares two approaches:
1. **XGBoost** – Gradient boosting optimized for tabular data
2. **MLP Neural Network** – Deep learning with dropout regularization

---

## 📊 Dataset

- **Source:** [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 339,607 transactions
- **Features:** 15 columns (temporal, geographic, transactional)
- **Target:** Binary (0 = Normal, 1 = Fraud)
- **Class Distribution:** 
  - Normal: 337,825 (99.48%)
  - Fraud: 1,782 (0.52%)

### Key Features:
- Transaction amount
- Geographic coordinates (customer & merchant)
- Temporal features (hour, day of week, month)
- Distance between customer and merchant (Haversine formula)
- Category, state (one-hot encoded)

---

## 🔧 Methodology

### 1. Data Preprocessing
- **Missing values:** None detected
- **Outlier handling:** Retained extreme values (informative for fraud detection)
- **Feature engineering:**
  - Haversine distance calculation
  - Temporal feature extraction (hour, day_of_week, month)
  - Age calculation from date of birth
  - One-hot encoding for categorical variables

### 2. Train/Test Split
- **Strategy:** 80/20 stratified split
- **Preservation:** Fraud/normal ratio maintained in both sets

### 3. Class Imbalance Handling
**Critical design decision:** Cost-sensitive learning instead of resampling
- **XGBoost:** `scale_pos_weight = 189.5` (ratio normal/fraud)
- **MLP:** `class_weight = {0: 1, 1: 189.5}`

**Rationale:**
- Preserves actual data distribution
- Avoids synthetic data generation (SMOTE)
- Forces model to prioritize minority class

### 4. Model Training

#### **Model 1: XGBoost**
```python
XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=189.5,
    eval_metric="aucpr"
)
```

#### **Model 2: Multi-Layer Perceptron (MLP)**
```python
Sequential([
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
```
- **Optimizer:** Adam
- **Loss:** Binary crossentropy
- **Epochs:** 20
- **Batch size:** 256

---

## 📈 Results

### Model Comparison

| Metric | XGBoost | MLP |
|--------|---------|-----|
| **AUC-ROC** | **0.9967** | 0.9871 |
| **Recall** | **92.13%** | 94.38% |
| **Precision** | **49.92%** | 7.15% |
| **F1-Score** | **64.76%** | 13.29% |
| **False Positives** | **329** | 4,364 |
| **False Negatives** | **28** | 20 |

### Key Insights

✅ **XGBoost is the recommended model:**
- **Operational efficiency:** 50% chance of correct fraud alert vs 7% for MLP
- **Customer experience:** 329 false positives vs 4,364 (MLP blocks 13x more legitimate transactions)
- **Business viability:** Balances fraud detection with customer satisfaction

⚠️ **MLP achieves higher recall but catastrophic precision:**
- Detects 8 more frauds than XGBoost
- Generates 4,364 false alarms (commercially unacceptable)


**Detection rate:** 92.13% of actual frauds caught  
**False alarm rate:** 0.49% of normal transactions flagged

---

## 🚀 Technical Implementation

### Installation

```bash
# Clone repository
git clone https://github.com/willisdongmo/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.10.0
tensorflow>=2.11.0
scikit-learn>=1.2.0
xgboost>=1.7.0
```

### Usage

```python
# Run complete analysis
python credit_card_fraud_detection.py

# Or use Jupyter notebook
jupyter notebook CCFD.ipynb
```

---
---

## 🔍 Key Findings

### Factor Analysis
1. **No linear separability:** PCA and t-SNE show fraud and normal transactions overlap significantly
2. **Non-linear patterns:** Tree-based and neural models required
3. **Critical features:**
   - Transaction amount (fraudsters keep amounts low to evade detection)
   - Time of day (higher fraud rate 10 PM – 4 AM)
   - Category (online shopping and shopping_pos more vulnerable)
   - Geographic distance

### EDA Insights
- **Class imbalance:** 99.48% normal vs 0.52% fraud
- **Temporal patterns:** Fraud peaks during late-night hours
- **Category risk:** Shopping categories show highest fraud rates (1.4%)
- **Amount overlap:** Fraudulent transactions span similar ranges as legitimate ones

---

## 💡 Limitations & Future Work

### Limitations
1. **Dataset age:** Data may not reflect current fraud patterns
2. **Feature availability:** Real-world deployment may lack some features
3. **Threshold optimization:** Default 0.5 threshold not optimized for business costs

### Future Improvements
1. **Threshold tuning:** Optimize decision threshold using Precision-Recall curve
2. **Ensemble methods:** Combine XGBoost + MLP predictions
3. **Feature engineering:** Add customer transaction history, velocity features
4. **Temporal validation:** Train on past data, test on future data
5. **Deep learning architectures:** TabNet, LSTM for sequential patterns

---

## 📚 References

- Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- XGBoost Documentation: [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)
- Imbalanced Learning: Chawla et al. (2002) SMOTE
- Cost-sensitive Learning: Elkan (2001)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Contributors & Acknowledgments

**Primary Author:** Willis Dongmo  
GitHub: [@willisdongmo](https://github.com/willisdongmo)  
LinkedIn: [Willis Dongmo](https://linkedin.com/in/willisdongmo)
- Complete data analysis, modeling, and evaluation

**Project Contributor:** Emily Chiu  
- Initial project idea and scope definition
- Dataset selection and sourcing
- Academic context coordination

**Academic Context:** DAT 402 - Data Science Project Course

---

---

**⭐ If you found this project useful, please consider giving it a star!**

