"""
Exercise description
--------------------

Description:
In the context of Mercadolibre's Marketplace an algorithm is needed to predict if an item listed in the markeplace is new or used.

Your tasks involve the data analysis, designing, processing and modeling of a machine learning solution 
to predict if an item is new or used and then evaluate the model over held-out test data.

To assist in that task a dataset is provided in `MLA_100k.jsonlines` and a function to read that dataset in `build_dataset`.

For the evaluation, you will use the accuracy metric in order to get a result of 0.86 as minimum. 
Additionally, you will have to choose an appropiate secondary metric and also elaborate an argument on why that metric was chosen.

The deliverables are:
--The file, including all the code needed to define and evaluate a model.
--A document with an explanation on the criteria applied to choose the features, 
  the proposed secondary metric and the performance achieved on that metrics. 
  Optionally, you can deliver an EDA analysis with other formart like .ipynb



"""

# Import shared utilities
from utils import (
    build_dataset, extract_features, create_feature_matrix, preprocess_data,
    build_models, evaluate_model, plot_roc_curve_and_confusion_matrix,
    save_model, load_model
)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier as RFForSelection
import xgboost as xgb
import catboost as cb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re
import warnings
warnings.filterwarnings('ignore')

def select_features(X_train, y_train, text_features, categorical_features, numerical_features):
    """
    Perform feature selection to identify the most important features
    """
    print("\n" + "="*50)
    print("FEATURE SELECTION")
    print("="*50)
    
    # Create a simple pipeline for feature selection
    from sklearn.preprocessing import OneHotEncoder
    
    # Text preprocessing with Spanish stop words
    spanish_stop_words = [
        'a', 'al', 'ante', 'bajo', 'con', 'contra', 'de', 'del', 'desde', 'durante',
        'en', 'entre', 'hacia', 'hasta', 'mediante', 'para', 'por', 'según', 'sin',
        'sobre', 'tras', 'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
        'y', 'o', 'pero', 'si', 'no', 'que', 'cual', 'quien', 'cuando', 'donde',
        'como', 'porque', 'pues', 'ya', 'también', 'más', 'menos', 'muy', 'poco',
        'mucho', 'todo', 'nada', 'algo', 'nadie', 'alguien', 'cada', 'cualquier',
        'este', 'esta', 'estos', 'estas', 'ese', 'esa', 'esos', 'esas', 'aquel',
        'aquella', 'aquellos', 'aquellas', 'mi', 'tu', 'su', 'nuestro', 'vuestro',
        'yo', 'tú', 'él', 'ella', 'nosotros', 'vosotros', 'ellos', 'ellas', 'me',
        'te', 'le', 'nos', 'os', 'les', 'se', 'lo', 'la', 'los', 'las', 'le',
        'les', 'me', 'te', 'se', 'nos', 'os', 'mi', 'tu', 'su', 'nuestro',
        'vuestro', 'mío', 'tuyo', 'suyo', 'mía', 'tuya', 'suya', 'míos', 'tuyos',
        'suyos', 'mías', 'tuyas', 'suyas', 'este', 'esta', 'estos', 'estas',
        'ese', 'esa', 'esos', 'esas', 'aquel', 'aquella', 'aquellos', 'aquellas'
    ]
    text_transformer = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=500, stop_words=spanish_stop_words, ngram_range=(1, 2)))
    ])
    
    # Categorical preprocessing
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Numerical preprocessing
    numerical_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    # Combine all transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_transformer, 'title'),
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)
        ],
        remainder='drop'
    )
    
    # Transform the data
    X_transformed = preprocessor.fit_transform(X_train)
    
    print(f"Original feature matrix shape: {X_transformed.shape}")
    
    # Select top features using faster methods
    n_features_to_select = max(50, int(X_transformed.shape[1] * 0.8))  # Reduce to 80% of features, minimum 50
    
    # Method 1: SelectKBest with f_classif (fast)
    print("Performing SelectKBest feature selection...")
    selector_kbest = SelectKBest(score_func=f_classif, k=n_features_to_select)
    X_selected_kbest = selector_kbest.fit_transform(X_transformed, y_train)
    
    # Method 2: Simple variance threshold for additional filtering
    from sklearn.feature_selection import VarianceThreshold
    print("Performing variance-based feature selection...")
    variance_selector = VarianceThreshold(threshold=0.01)
    X_variance_selected = variance_selector.fit_transform(X_transformed)
    
    print(f"Original features: {X_transformed.shape[1]}")
    print(f"Features selected by SelectKBest: {X_selected_kbest.shape[1]}")
    print(f"Features after variance filtering: {X_variance_selected.shape[1]}")
    print(f"Feature reduction: {((X_transformed.shape[1] - X_selected_kbest.shape[1]) / X_transformed.shape[1] * 100):.1f}%")
    
    # Analyze feature scores to understand why few features are needed
    feature_scores = selector_kbest.scores_
    feature_pvalues = selector_kbest.pvalues_
    
    # Get top 10 features with their scores
    top_features_idx = np.argsort(feature_scores)[-10:][::-1]
    print("\nTop 10 Most Discriminative Features:")
    for i, idx in enumerate(top_features_idx):
        print(f"{i+1:2d}. Feature {idx:4d}: Score={feature_scores[idx]:.4f}, p-value={feature_pvalues[idx]:.2e}")
    
    # Use SelectKBest for this implementation (faster than RFE)
    selected_features_mask = selector_kbest.get_support()
    
    return preprocessor, selected_features_mask, selector_kbest

def select_best_model(models, X_train, y_train, X_test, y_test):
    """
    Train and evaluate multiple models to select the best one
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON AND SELECTION")
    print("="*60)
    
    results = {}
    best_model = None
    best_score = 0
    best_model_name = ""
    
    for name, model in models.items():
        print(f"\n{'='*20} Training {name} {'='*20}")
        
        # Train the model
        print(f"Starting training for {name}...")
        model.fit(X_train, y_train)
        print(f"Training completed for {name}")
        
        # Make predictions
        print(f"Making predictions for {name}...")
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Cross-validation to assess overfitting
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1_weighted')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'model': model,
            'predictions': y_pred
        }
        
        print(f"✅ {name} - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        print(f"   Cross-validation F1: {cv_mean:.4f} (+/- {cv_std*2:.4f})")
        
        # Check for overfitting (CV score vs test score)
        overfitting_gap = cv_mean - f1
        if overfitting_gap > 0.05:
            print(f"   ⚠️  Potential overfitting detected (gap: {overfitting_gap:.4f})")
        elif overfitting_gap > 0.02:
            print(f"   ⚠️  Moderate overfitting detected (gap: {overfitting_gap:.4f})")
        else:
            print(f"   ✅ No significant overfitting (gap: {overfitting_gap:.4f})")
        
        print(f"{'='*50}")
        
        # Update best model (consider both performance and overfitting)
        if f1 > best_score and overfitting_gap < 0.1:  # Prefer models with less overfitting
            best_score = f1
            best_model = model
            best_model_name = name
    
    print(f"\n" + "="*50)
    print("MODEL COMPARISON SUMMARY")
    print("="*50)
    
    # Sort models by F1-score
    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
    
    print(f"{'Model':<20} {'Accuracy':<10} {'F1-Score':<10}")
    print("-" * 40)
    for name, metrics in sorted_results:
        print(f"{name:<20} {metrics['accuracy']:<10.4f} {metrics['f1_score']:<10.4f}")
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"F1-Score: {best_score:.4f}")
    print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
    
    return best_model, best_model_name, results

def train_champion_model(best_model_name, text_features, categorical_features, numerical_features, X_train_full, y_train_full):
    """
    Train the best model on the full dataset
    """
    print(f"\n{'='*60}")
    print(f"TRAINING CHAMPION MODEL: {best_model_name}")
    print(f"{'='*60}")
    
    # Optional feature selection for champion model (80% feature reduction)
    USE_FEATURE_SELECTION = False  # Set to False to skip feature selection
    
    if USE_FEATURE_SELECTION:
        print("Using feature selection for champion model training...")
        feature_preprocessor, selected_features_mask, feature_selector = select_features(
            X_train_full, y_train_full, text_features, categorical_features, numerical_features
        )
        models = build_models(text_features, categorical_features, numerical_features, feature_selector)
    else:
        print("Skipping feature selection for champion model training...")
        models = build_models(text_features, categorical_features, numerical_features, None)
    champion_model = models[best_model_name]
    
    print(f"Training {best_model_name} on full dataset ({len(X_train_full)} samples)...")
    
    # For Random Forest, we can add more regularization if needed
    if 'Random Forest' in best_model_name:
        # Get the current Random Forest parameters
        rf_classifier = champion_model.named_steps['classifier']
        print(f"   Current parameters: n_estimators={rf_classifier.n_estimators}, max_depth={rf_classifier.max_depth}")
        
        # If we detect overfitting, we can adjust parameters
        if hasattr(rf_classifier, 'max_depth') and rf_classifier.max_depth is None:
            print("   ⚠️  Adjusting max_depth to prevent overfitting...")
            rf_classifier.max_depth = 8
    
    champion_model.fit(X_train_full, y_train_full)
    print("✅ Champion model training completed!")
    
    # Assess overfitting on the full dataset
    print("\n" + "="*50)
    print("CHAMPION MODEL OVERFITTING ASSESSMENT")
    print("="*50)
    
    # Cross-validation on full dataset
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(champion_model, X_train_full, y_train_full, cv=3, scoring='f1_weighted')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"Cross-validation F1: {cv_mean:.4f} (+/- {cv_std*2:.4f})")
    
    # Make predictions on training set to assess overfitting
    y_train_pred = champion_model.predict(X_train_full)
    train_accuracy = accuracy_score(y_train_full, y_train_pred)
    train_f1 = f1_score(y_train_full, y_train_pred, average='weighted')
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Training F1-Score: {train_f1:.4f}")
    
    # Calculate overfitting gap (we'll compare with test performance later)
    print(f"\nOverfitting Assessment:")
    print(f"CV F1-Score: {cv_mean:.4f}")
    print(f"Training F1-Score: {train_f1:.4f}")
    
    # Save the champion model
    model_filename = save_model(champion_model, best_model_name)
    
    return champion_model

def main():
    """
    Main function to run the complete pipeline
    """
    print("Starting the New vs Used Item Classification Pipeline")
    print("="*60)
    
    # Use 30% of training data for initial training to determine champion
    print("Using 30% of training data for initial training to determine champion...")
    sample_size = int(len(X_train) * 0.3)  # 30% of training data
    print(f"Training set: {sample_size} samples (30% of {len(X_train)})")
    print(f"Test set: {len(X_test)} samples")
    
    # Sample training data
    train_indices = np.random.choice(len(X_train), sample_size, replace=False)
    X_train_sample = [X_train[i] for i in train_indices]
    y_train_sample = [y_train[i] for i in train_indices]
    
    # Preprocess data
    print("Preprocessing data...")
    train_df, test_df, y_train_encoded, y_test_encoded, text_features, categorical_features, numerical_features = preprocess_data(
        X_train_sample, X_test, y_train_sample, y_test
    )
    print(f"Feature extraction completed. Using {len(numerical_features)} numerical features and {len(categorical_features)} categorical features.")
    
    # Optional feature selection - reduce to 80% of features
    USE_FEATURE_SELECTION = False  # Set to False to skip feature selection
    
    if USE_FEATURE_SELECTION:
        print("\nUsing feature selection to reduce to 80% of features...")
        feature_preprocessor, selected_features_mask, feature_selector = select_features(
            train_df, y_train_encoded, text_features, categorical_features, numerical_features
        )
        models = build_models(text_features, categorical_features, numerical_features, feature_selector)
    else:
        print("\nSkipping feature selection - using all features...")
        print(f"Using all {len(text_features) + len(categorical_features) + len(numerical_features)} features")
        models = build_models(text_features, categorical_features, numerical_features, None)
    
    # Select best model
    best_model, best_model_name, all_results = select_best_model(
        models, train_df, y_train_encoded, test_df, y_test_encoded
    )
    
    # Evaluate best model in detail
    print(f"\nDetailed evaluation of best model: {best_model_name}")
    accuracy, f1_score_result = evaluate_model(best_model, train_df, y_train_encoded, test_df, y_test, y_test_encoded)
    
    # Check if accuracy meets the minimum requirement
    print("\n" + "="*50)
    print("REQUIREMENT CHECK")
    print("="*50)
    print(f"Minimum required accuracy: 0.86")
    print(f"Achieved accuracy: {accuracy:.4f}")
    print(f"Requirement met: {'✓ YES' if accuracy >= 0.86 else '✗ NO'}")
    
    # Train champion model on full dataset
    print("\n" + "="*60)
    print("CHAMPION MODEL TRAINING")
    print("="*60)
    
    # Preprocess full training data
    print("Preprocessing full training dataset...")
    train_df_full, _, y_train_encoded_full, _, _, _, _ = preprocess_data(
        X_train, X_test, y_train, y_test
    )
    
    # Train champion model on full dataset with optional feature selection
    champion_model = train_champion_model(
        best_model_name, text_features, categorical_features, numerical_features,
        train_df_full, y_train_encoded_full
    )
    
    # Load and evaluate champion model
    model_filename = f"models/champion_model_{best_model_name.lower().replace(' ', '_')}.pkl"
    loaded_champion_model = load_model(model_filename)
    
    # Create visualizations
    champion_accuracy, champion_f1 = plot_roc_curve_and_confusion_matrix(
        loaded_champion_model, test_df, y_test, y_test_encoded, best_model_name
    )
    
    # Final overfitting assessment on champion model
    print("\n" + "="*50)
    print("FINAL CHAMPION MODEL OVERFITTING ASSESSMENT")
    print("="*50)
    
    # Get training performance from the champion model
    y_train_pred_champion = loaded_champion_model.predict(train_df_full)
    train_accuracy_champion = accuracy_score(y_train_encoded_full, y_train_pred_champion)
    train_f1_champion = f1_score(y_train_encoded_full, y_train_pred_champion, average='weighted')
    
    # Cross-validation on full dataset for champion model
    cv_scores_champion = cross_val_score(loaded_champion_model, train_df_full, y_train_encoded_full, cv=3, scoring='f1_weighted')
    cv_mean_champion = cv_scores_champion.mean()
    cv_std_champion = cv_scores_champion.std()
    
    print(f"Champion Model Performance:")
    print(f"  Training Accuracy: {train_accuracy_champion:.4f}")
    print(f"  Training F1-Score: {train_f1_champion:.4f}")
    print(f"  Test Accuracy: {champion_accuracy:.4f}")
    print(f"  Test F1-Score: {champion_f1:.4f}")
    print(f"  Cross-validation F1: {cv_mean_champion:.4f} (+/- {cv_std_champion*2:.4f})")
    
    # Calculate overfitting gaps
    accuracy_gap = train_accuracy_champion - champion_accuracy
    f1_gap = train_f1_champion - champion_f1
    cv_test_gap = cv_mean_champion - champion_f1
    
    print(f"\nOverfitting Analysis:")
    print(f"  Accuracy Gap (Train-Test): {accuracy_gap:.4f}")
    print(f"  F1-Score Gap (Train-Test): {f1_gap:.4f}")
    print(f"  CV-Test F1 Gap: {cv_test_gap:.4f}")
    
    # Overfitting assessment
    if accuracy_gap > 0.05 or f1_gap > 0.05:
        print(f"  ⚠️  WARNING: Significant overfitting detected!")
        print(f"     - Accuracy gap: {accuracy_gap:.4f} (> 0.05)")
        print(f"     - F1-score gap: {f1_gap:.4f} (> 0.05)")
    elif accuracy_gap > 0.02 or f1_gap > 0.02:
        print(f"  ⚠️  MODERATE overfitting detected!")
        print(f"     - Accuracy gap: {accuracy_gap:.4f} (> 0.02)")
        print(f"     - F1-score gap: {f1_gap:.4f} (> 0.02)")
    else:
        print(f"  ✅ No significant overfitting detected!")
        print(f"     - Accuracy gap: {accuracy_gap:.4f} (≤ 0.02)")
        print(f"     - F1-score gap: {f1_gap:.4f} (≤ 0.02)")
    
    if cv_test_gap > 0.05:
        print(f"  ⚠️  Cross-validation suggests potential overfitting!")
    elif cv_test_gap > 0.02:
        print(f"  ⚠️  Cross-validation suggests moderate overfitting!")
    else:
        print(f"  ✅ Cross-validation confirms good generalization!")
    
    # Feature importance analysis
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*50)
    
    # Get feature names from the preprocessor
    feature_names = []
    
    # Add text features (TF-IDF features)
    text_vectorizer = best_model.named_steps['preprocessor'].named_transformers_['text']
    if hasattr(text_vectorizer, 'get_feature_names_out'):
        text_features_names = text_vectorizer.get_feature_names_out()
        feature_names.extend([f"text_{name}" for name in text_features_names])
    
    # Add categorical features
    feature_names.extend(categorical_features)
    
    # Add numerical features
    feature_names.extend(numerical_features)
    
    # Get feature importance (only for tree-based models)
    classifier = best_model.named_steps['classifier']
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        
        # Ensure feature_names and importances have the same length
        if len(feature_names) >= len(importances):
            feature_names_used = feature_names[:len(importances)]
        else:
            # If we have more importances than feature names, truncate importances
            importances = importances[:len(feature_names)]
            feature_names_used = feature_names
        
        # Create feature importance DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': feature_names_used,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("Top 20 Most Important Features:")
        print(feature_importance_df.head(20))
    else:
        print("Feature importance not available for this model type.")
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"🏆 Best Model: {best_model_name}")
    print(f"📊 Sample Performance: Accuracy={accuracy:.4f}, F1={f1_score_result:.4f}")
    print(f"🏆 Champion Performance: Accuracy={champion_accuracy:.4f}, F1={champion_f1:.4f}")
    print(f"💾 Model saved as: {model_filename}")
    print(f"📊 Visualizations saved as: model_results/champion_model_performance_{best_model_name.lower().replace(' ', '_')}.png")
    
    return loaded_champion_model, champion_accuracy, champion_f1

if __name__ == "__main__":
    print("Loading dataset...")
    # Train and test data following sklearn naming conventions
    # X_train (X_test too) is a list of dicts with information about each item.
    # y_train (y_test too) contains the labels to be predicted (new or used).
    # The label of X_train[i] is y_train[i].
    # The label of X_test[i] is y_test[i].
    X_train, y_train, X_test, y_test = build_dataset()
    
    # Run the complete pipeline
    model, accuracy, f1_score_result = main()


