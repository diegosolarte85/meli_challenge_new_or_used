# New vs Used Item Classification - MercadoLibre Challenge

## Overview

This project implements a machine learning solution to predict whether items listed on MercadoLibre's marketplace are new or used. The solution achieves an accuracy of **86.6%** and an F1-score of **86.6%**, meeting the minimum requirement of 86% accuracy.

## Problem Statement

In the context of MercadoLibre's Marketplace, an algorithm is needed to predict if an item listed in the marketplace is new or used. The task involves:

- Data analysis and feature engineering
- Model design and training
- Evaluation using accuracy (primary metric) and F1-score (secondary metric)
- Achieving a minimum accuracy of 0.86

## Solution Architecture

### Data Processing Pipeline

1. **Feature Extraction**: Extracts 49 features from the raw JSON data including:
   - **Numerical features**: Price, quantities, discounts, counts, etc.
   - **Categorical features**: Category ID, listing type, buying mode, site ID, currency, seller country
   - **Text features**: Product titles processed with TF-IDF vectorization
   - **Boolean features**: Payment methods, shipping options, warranty, etc.

2. **Feature Engineering**: Creates domain-specific features such as:
   - Text-based indicators (has "usado", "nuevo", "original", etc.)
   - Price-related features (discounts, original price flags)
   - Temporal features (listing age)
   - Seller and location features

3. **Data Preprocessing**:
   - Handles missing values and empty strings
   - Standardizes numerical features
   - One-hot encodes categorical features
   - TF-IDF vectorization for text features

### Model

- **Algorithm**: Random Forest Classifier
- **Parameters**: 100 estimators, parallel processing enabled
- **Pipeline**: Combines preprocessing and classification in a single pipeline

## Performance Results

### Primary Metric: Accuracy
- **Achieved**: 87.36%
- **Requirement**: ≥86%
- **Status**: ✅ **REQUIREMENT MET**

### Secondary Metric: F1-Score (Weighted)
- **Achieved**: 87.36%
- **Rationale**: F1-score is chosen as the secondary metric because:
  1. **Balanced Performance**: It provides a harmonic mean of precision and recall, ensuring the model performs well on both classes
  2. **Class Imbalance Handling**: With weighted averaging, it accounts for the slight class imbalance in the dataset
  3. **Business Impact**: In e-commerce, both false positives (new items classified as used) and false negatives (used items classified as new) have business implications
  4. **Comprehensive Evaluation**: Unlike accuracy alone, F1-score penalizes models that perform poorly on either precision or recall

### Model Comparison Results

| Model | Accuracy | F1-Score | Overfitting Status |
|-------|----------|----------|-------------------|
| **XGBoost** | **87.35%** | **87.35%** | ✅ No overfitting |
| CatBoost | 87.17% | 87.17% | ✅ No overfitting |
| Neural Network | 86.63% | 86.65% | ✅ No overfitting |
| Random Forest | 85.42% | 85.40% | ✅ No overfitting |
| Logistic Regression | 84.94% | 84.87% | ✅ No overfitting |
| Decision Tree | 84.40% | 84.40% | ✅ No overfitting |
| K-Nearest Neighbors | 76.97% | 77.00% | ✅ No overfitting |

### Champion Model Performance (XGBoost)

#### Detailed Performance
```
              precision    recall  f1-score   support

         new       0.88      0.88      0.88      5406
        used       0.86      0.86      0.86      4594

    accuracy                           0.87     10000
   macro avg       0.87      0.87      0.87     10000
weighted avg       0.87      0.87      0.87     10000
```

#### Confusion Matrix
```
[[4769  637]  # True Negatives: 4769, False Positives: 637
 [ 628 3966]] # False Negatives: 628, True Positives: 3966
```

#### Overfitting Assessment
- **Training Accuracy**: 87.73%
- **Test Accuracy**: 87.36%
- **Accuracy Gap**: 0.37% (≤ 2% threshold)
- **Cross-validation F1**: 87.26% (± 0.38%)
- **Status**: ✅ **No significant overfitting detected**

### Detailed Model Comparison Results

#### Initial Training (30% of data)
| Model | Accuracy | F1-Score | Overfitting Status |
|-------|----------|----------|-------------------|
| **XGBoost** | **87.35%** | **87.35%** | ✅ No overfitting (gap: 0.98%) |
| CatBoost | 87.17% | 87.17% | ✅ No overfitting (gap: 0.14%) |
| Neural Network | 86.63% | 86.65% | ✅ No overfitting (gap: 0.04%) |
| Random Forest | 85.42% | 85.40% | ✅ No overfitting (gap: -0.36%) |
| Logistic Regression | 84.94% | 84.87% | ✅ No overfitting (gap: -0.19%) |
| Decision Tree | 84.40% | 84.40% | ✅ No overfitting (gap: -0.50%) |
| K-Nearest Neighbors | 76.97% | 77.00% | ✅ No overfitting (gap: -0.89%) |

#### Champion Model (Full dataset)
- **Model**: XGBoost
- **Training Samples**: 90,000
- **Test Samples**: 10,000
- **Final Accuracy**: 87.36%
- **Final F1-Score**: 87.36%
- **Cross-validation F1**: 87.26% (± 0.38%)
- **Overfitting Gap**: 0.37% (well below 2% threshold)

### Performance Insights
- **Best Algorithm**: XGBoost consistently outperformed all other models
- **Stable Performance**: Minimal performance degradation from sample to full dataset
- **Strong Generalization**: All models show excellent generalization with minimal overfitting
- **Balanced Classes**: Model performs well on both "new" and "used" classes
- **High Precision**: 88% precision for new items, 86% for used items

## Feature Importance Analysis

The top 20 most important features (based on XGBoost feature importance):

| Rank | Feature | Importance | Description |
|------|---------|------------|-------------|
| 1 | **text_vinilo** | 0.0136 | Text feature indicating "vinilo" (vinyl) |
| 2 | **text_antigua** | 0.0077 | Text feature indicating "antigua" (antique) |
| 3 | **text_lp** | 0.0069 | Text feature indicating "lp" (long play) |
| 4 | **text_mint** | 0.0067 | Text feature indicating "mint" (perfect condition) |
| 5 | **text_impecable** | 0.0066 | Text feature indicating "impecable" (flawless) |
| 6 | **text_consolador** | 0.0053 | Text feature indicating "consolador" (console) |
| 7 | **text_kit** | 0.0053 | Text feature indicating "kit" |
| 8 | **text_estado** | 0.0050 | Text feature indicating "estado" (condition) |
| 9 | **text_honda** | 0.0046 | Text feature indicating "honda" (brand) |
| 10 | **text_revista** | 0.0046 | Text feature indicating "revista" (magazine) |
| 11 | **text_faro** | 0.0045 | Text feature indicating "faro" (headlight) |
| 12 | **text_libro digital** | 0.0043 | Text feature indicating "digital book" |
| 13 | **text_trasero** | 0.0042 | Text feature indicating "trasero" (rear) |
| 14 | **text_marca** | 0.0040 | Text feature indicating "marca" (brand) |
| 15 | **text_delantero** | 0.0036 | Text feature indicating "delantero" (front) |
| 16 | **text_aros** | 0.0036 | Text feature indicating "aros" (rings/wheels) |
| 17 | **text_antiguo** | 0.0036 | Text feature indicating "antiguo" (old) |
| 18 | **text_renault** | 0.0035 | Text feature indicating "renault" (brand) |
| 19 | **text_vhs** | 0.0034 | Text feature indicating "vhs" (video format) |
| 20 | **text_filtro** | 0.0034 | Text feature indicating "filtro" (filter) |

### Key Insights:
- **Text features dominate**: The most important features are text-based, indicating that product titles contain strong signals about item condition
- **Condition indicators**: Words like "mint", "impecable", "antigua", "antiguo" are highly predictive of item condition
- **Product-specific terms**: Terms like "vinilo", "lp", "vhs" suggest vintage/collectible items are more likely to be used
- **Brand names**: Specific brands like "honda", "renault" appear as important features
- **Automotive parts**: Terms like "faro", "trasero", "delantero", "aros", "filtro" suggest automotive listings have strong condition indicators

### Final Results Summary

#### 🏆 Champion Model: XGBoost
- **Sample Performance**: Accuracy=87.35%, F1=87.35%
- **Champion Performance**: Accuracy=87.36%, F1=87.36%
- **Model File**: `champion_model_xgboost.pkl`
- **Visualizations**: `champion_model_performance_xgboost.png`

#### 📊 Performance Metrics
- **Accuracy**: 87.36% (exceeds 86% requirement)
- **F1-Score**: 87.36% (weighted average)
- **Precision (New)**: 88%
- **Recall (New)**: 88%
- **Precision (Used)**: 86%
- **Recall (Used)**: 86%

#### ✅ Quality Assurance
- **Overfitting Check**: ✅ No significant overfitting (gap: 0.37%)
- **Cross-validation**: 87.26% (± 0.38%)
- **Generalization**: Excellent model stability
- **Class Balance**: Good performance on both classes

## Technical Implementation

### Data Sampling Strategy
- **Initial training**: 30% of training data (27,000 samples) for model selection
- **Full training**: Complete dataset (90,000 samples) for champion model
- **Test set**: 10,000 samples for final evaluation
- **Rationale**: Efficient model comparison with full dataset for final model

### Model Configuration
- **XGBoost (Champion)**: 200 estimators, max_depth=5, learning_rate=0.1
- **All models**: Standardized to 200 iterations for fair comparison
- **TF-IDF**: Spanish stop words, n-gram range (1,2), max_features=1000
- **Categorical encoding**: One-hot encoding with unknown handling
- **Numerical scaling**: StandardScaler for normalization

### Performance Optimizations
- **Parallel processing**: Enabled for all compatible models
- **Memory management**: Excluded high-cardinality features (seller_id, seller_city, seller_state)
- **Feature selection**: Disabled to use all available features
- **Regularization**: Strong L2 regularization (C=0.1) for Logistic Regression
- **Early stopping**: Enabled for Neural Network to prevent overfitting

## Visualizations and Plots

The pipeline generates comprehensive visualizations for the champion model:

### Generated Files
- `champion_model_performance_xgboost.png`: Combined visualization containing:
  - **ROC Curve**: Shows model's ability to distinguish between classes
  - **Confusion Matrix**: Visual representation of predictions vs actual values
  - **Performance Metrics**: Accuracy, precision, recall, F1-score

### Plot Interpretation
- **ROC Curve**: Area under curve (AUC) indicates model discrimination ability
- **Confusion Matrix**: 
  - True Negatives (top-left): Correctly classified new items
  - False Positives (top-right): New items classified as used
  - False Negatives (bottom-left): Used items classified as new
  - True Positives (bottom-right): Correctly classified used items

### Model Performance Visualization
The plots provide visual confirmation of the model's strong performance:
- High accuracy in both classes (new: 88%, used: 86%)
- Balanced precision and recall across classes
- Clear separation in ROC curve indicating good discrimination

## Usage

To run the complete pipeline:

```bash
python new_or_used.py
```

The script will:
1. Load and sample the dataset
2. Extract and engineer features
3. Train the Random Forest model
4. Evaluate performance
5. Display results and feature importance

## Files

### Source Files
- `new_or_used.py`: Main implementation with complete ML pipeline
- `MLA_100k.jsonlines`: Dataset containing 100k MercadoLibre listings
- `README.md`: This documentation file

### Generated Files
- `champion_model_xgboost.pkl`: Saved champion model (XGBoost)
- `champion_model_performance_xgboost.png`: Performance visualizations (ROC curve + confusion matrix)

## Future Improvements

1. **Full Dataset Training**: Train on the complete dataset for potentially better performance
2. **Hyperparameter Tuning**: Use GridSearchCV or RandomizedSearchCV for optimal parameters
3. **Ensemble Methods**: Combine multiple models (XGBoost, LightGBM, etc.)
4. **Deep Learning**: Experiment with neural networks for text processing
5. **Feature Engineering**: Add more domain-specific features based on MercadoLibre's business logic

## Conclusion

The implemented solution successfully meets all requirements:
- ✅ Achieves accuracy ≥86% (87.36%)
- ✅ Includes comprehensive feature engineering and model comparison
- ✅ Provides detailed evaluation with secondary metric (F1-score: 87.36%)
- ✅ Demonstrates feature importance analysis with XGBoost
- ✅ Uses appropriate preprocessing and modeling techniques
- ✅ Includes comprehensive overfitting assessment
- ✅ Generates performance visualizations

### Key Achievements
- **Best Model**: XGBoost achieved 87.36% accuracy and F1-score
- **No Overfitting**: All models show minimal train-test performance gaps
- **Comprehensive Evaluation**: 7 different algorithms compared with standardized iterations
- **Strong Generalization**: Cross-validation confirms good model performance
- **Feature Insights**: Text features dominate, with condition indicators being most predictive

The model shows strong performance in distinguishing between new and used items, with text features being the most predictive, which aligns with the intuitive understanding that product titles often contain explicit condition indicators. The XGBoost model demonstrates excellent balance between performance and generalization.

