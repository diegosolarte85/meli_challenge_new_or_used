"""
Advanced Modeling for New vs Used Item Classification
===================================================

This module extends the base implementation with hyperparameter tuning and deep learning.
Reuses the working data processing from utils.py.

Author: Diego
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
import time
import re
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

# Scikit-learn imports
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, auc, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# XGBoost and CatBoost
import xgboost as xgb
import catboost as cb

# Deep Learning imports (optional)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional, GlobalMaxPooling1D
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️  TensorFlow not available. Deep learning features will be disabled.")

# Suppress warnings
warnings.filterwarnings('ignore')
if TENSORFLOW_AVAILABLE:
    tf.get_logger().setLevel('ERROR')

# Import the working functions from utils.py
from utils import (
    build_dataset, extract_features, create_feature_matrix, preprocess_data, save_model,
    prepare_text_data_for_deep_learning, build_deep_learning_models, analyze_text_patterns,
    compare_model_performance, clean_text
)

class AdvancedModeling:
    """
    Advanced modeling with hyperparameter tuning and deep learning
    """
    
    def __init__(self, tuning_level='simple', enable_deep_learning=True):
        """
        Initialize the advanced modeling class
        
        Args:
            tuning_level: 'simple', 'comprehensive', or 'full'
            enable_deep_learning: Whether to enable deep learning models
        """
        self.tuning_level = tuning_level
        self.enable_deep_learning = enable_deep_learning and TENSORFLOW_AVAILABLE
        self.best_xgb_params = None
        self.best_xgb_score = 0
        self.deep_learning_model = None
        self.tokenizer = None
        self.label_encoder = LabelEncoder()
        
        print(f"🔧 Configuration:")
        print(f"   Tuning Level: {tuning_level}")
        print(f"   Deep Learning: {'✅ Enabled' if self.enable_deep_learning else '❌ Disabled'}")
        
    def load_and_prepare_data(self):
        """
        Load and prepare the dataset using the working functions from utils.py
        """
        print("\n📊 Loading and preparing data...")
        
        # Use the working functions from utils.py
        X_train, y_train, X_test, y_test = build_dataset()
        
        print("Extracting features...")
        train_features = extract_features(X_train)
        test_features = extract_features(X_test)
        
        print("Creating feature matrices...")
        train_df, text_features, categorical_features, numerical_features = create_feature_matrix(train_features)
        test_df, _, _, _ = create_feature_matrix(test_features)
        
        # Add title column to DataFrames
        train_df['title'] = [item.get('title', '') for item in X_train]
        test_df['title'] = [item.get('title', '') for item in X_test]
        
        # Encode target variable
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)
        
        print(f"Training set shape: {train_df.shape}")
        print(f"Test set shape: {test_df.shape}")
        print(f"Target classes: {le.classes_}")
        print(f"Class distribution in training set: {np.bincount(y_train_encoded)}")
        print(f"Class distribution in test set: {np.bincount(y_test_encoded)}")
        
        # Preprocess data for XGBoost - convert categorical to numerical
        print("Preprocessing data for XGBoost...")
        
        # Create a copy to avoid modifying original data
        train_df_processed = train_df.copy()
        test_df_processed = test_df.copy()
        
        # Encode categorical features
        categorical_encoders = {}
        for col in categorical_features:
            if col in train_df_processed.columns:
                le_cat = LabelEncoder()
                # Combine unique values from both train and test to handle unseen values
                all_values = pd.concat([train_df_processed[col], test_df_processed[col]]).unique()
                le_cat.fit(all_values)
                
                # Transform both datasets
                train_df_processed[col] = le_cat.transform(train_df_processed[col])
                test_df_processed[col] = le_cat.transform(test_df_processed[col])
                
                categorical_encoders[col] = le_cat
        
        # Handle text features - convert to numerical features
        # For simplicity, we'll use basic text features instead of TF-IDF for XGBoost
        if 'title' in train_df_processed.columns:
            # Remove title column as XGBoost can't handle text directly
            train_df_processed = train_df_processed.drop('title', axis=1)
            test_df_processed = test_df_processed.drop('title', axis=1)
        
        # Convert all columns to numeric
        for col in train_df_processed.columns:
            train_df_processed[col] = pd.to_numeric(train_df_processed[col], errors='coerce').fillna(0)
            test_df_processed[col] = pd.to_numeric(test_df_processed[col], errors='coerce').fillna(0)
        
        # Convert to numpy arrays
        X_train_processed = train_df_processed.values
        X_test_processed = test_df_processed.values
        y_train_processed = np.array(y_train_encoded)
        y_test_processed = np.array(y_test_encoded)
        
        print(f"✓ Final processed training set shape: {X_train_processed.shape}")
        print(f"✓ Final processed test set shape: {X_test_processed.shape}")
        
        return X_train_processed, X_test_processed, y_train_processed, y_test_processed, train_df, test_df, y_train_encoded, y_test_encoded

    def define_parameter_spaces(self):
        """
        Define parameter spaces for hyperparameter tuning based on tuning level
        """
        if self.tuning_level == 'simple':
            # Simple parameter space for quick tuning
            param_space = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'reg_alpha': [0, 0.1],
                'reg_lambda': [0.1, 1.0]
            }
        elif self.tuning_level == 'comprehensive':
            # Comprehensive parameter space
            param_space = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'reg_alpha': [0, 0.1, 0.2],
                'reg_lambda': [0.1, 0.5, 1.0, 2.0]
            }
        else:  # full
            # Full parameter space for exhaustive tuning
            param_space = {
                'n_estimators': [50, 100, 200, 300, 500],
                'max_depth': [3, 5, 7, 9, 11],
                'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.2, 0.5],
                'reg_lambda': [0.1, 0.5, 1.0, 2.0, 5.0]
            }
        
        return param_space

    def perform_grid_search(self, X_train, y_train):
        """
        Perform grid search for hyperparameter tuning
        """
        print(f"\n🔍 Performing Grid Search (Level: {self.tuning_level})")
        
        param_space = self.define_parameter_spaces()
        
        # Create base XGBoost model
        base_model = xgb.XGBClassifier(
            random_state=42,
            verbosity=0,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_space,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        print("Starting grid search...")
        grid_search.fit(X_train, y_train)
        
        print(f"✓ Grid search completed!")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

    def perform_randomized_search(self, X_train, y_train):
        """
        Perform randomized search for hyperparameter tuning
        """
        print(f"\n🔍 Performing Randomized Search (Level: {self.tuning_level})")
        
        param_space = self.define_parameter_spaces()
        
        # Create base XGBoost model
        base_model = xgb.XGBClassifier(
            random_state=42,
            verbosity=0,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        # Determine number of iterations based on tuning level
        if self.tuning_level == 'simple':
            n_iter = 20
        elif self.tuning_level == 'comprehensive':
            n_iter = 50
        else:  # full
            n_iter = 100
        
        # Perform randomized search
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_space,
            n_iter=n_iter,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
        print(f"Starting randomized search with {n_iter} iterations...")
        random_search.fit(X_train, y_train)
        
        print(f"✓ Randomized search completed!")
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best score: {random_search.best_score_:.4f}")
        
        return random_search.best_estimator_, random_search.best_params_, random_search.best_score_

    def hyperparameter_tuning_xgboost(self, X_train, y_train, X_test, y_test):
        """
        Perform hyperparameter tuning for XGBoost
        """
        print("\n" + "="*60)
        print("XGBOOST HYPERPARAMETER TUNING")
        print("="*60)
        
        # Choose tuning method based on tuning level
        if self.tuning_level == 'simple':
            # Use grid search for simple tuning
            best_model, best_params, best_score = self.perform_grid_search(X_train, y_train)
        else:
            # Use randomized search for comprehensive and full tuning
            best_model, best_params, best_score = self.perform_randomized_search(X_train, y_train)
        
        # Evaluate the tuned model
        print("\n📊 Evaluating tuned XGBoost model...")
        y_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Tuned XGBoost Performance:")
        print(f"  Training F1 (CV): {best_score:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(f"  Test F1-Score: {test_f1:.4f}")
        
        # Store results
        self.best_xgb_params = best_params
        self.best_xgb_score = best_score
        
        return best_model, best_params, best_score

    def prepare_text_data_for_deep_learning(self, X_train, X_test, y_train, y_test):
        """
        Prepare text data for deep learning models using utils function
        """
        print("\n📝 Preparing text data for deep learning...")
        
        # Use the function from utils.py
        X_train_padded, X_test_padded, y_train_dl, y_test_dl, tokenizer = prepare_text_data_for_deep_learning(
            X_train, X_test, y_train, y_test
        )
        
        # Store tokenizer for later use
        self.tokenizer = tokenizer
        
        return X_train_padded, X_test_padded, y_train_dl, y_test_dl

    def build_deep_learning_models(self, X_train_padded, X_test_padded, y_train_encoded, y_test_encoded):
        """
        Build and train deep learning models using utils function
        """
        print("\n🧠 Building Deep Learning Models")
        print("="*60)
        
        # Use the function from utils.py
        models, best_model_name, best_score = build_deep_learning_models(
            X_train_padded, X_test_padded, y_train_encoded, y_test_encoded
        )
        
        return models, best_model_name, best_score

    def plot_training_history(self, models):
        """
        Plot training history for deep learning models
        """
        print("\n📈 Plotting Training History")
        
        fig, axes = plt.subplots(len(models), 2, figsize=(15, 5*len(models)))
        if len(models) == 1:
            axes = axes.reshape(1, -1)
        
        for i, (name, model_info) in enumerate(models.items()):
            history = model_info['history']
            
            # Plot accuracy
            axes[i, 0].plot(history.history['accuracy'], label='Training Accuracy')
            axes[i, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
            axes[i, 0].set_title(f'{name} - Accuracy')
            axes[i, 0].set_xlabel('Epoch')
            axes[i, 0].set_ylabel('Accuracy')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            
            # Plot loss
            axes[i, 1].plot(history.history['loss'], label='Training Loss')
            axes[i, 1].plot(history.history['val_loss'], label='Validation Loss')
            axes[i, 1].set_title(f'{name} - Loss')
            axes[i, 1].set_xlabel('Epoch')
            axes[i, 1].set_ylabel('Loss')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_results/deep_learning_training_history.png', dpi=300, bbox_inches='tight')
        print("📊 Training history plots saved as: model_results/deep_learning_training_history.png")
        plt.show()

    def compare_all_models(self, best_xgb_model, best_xgb_score, dl_models, best_dl_name, best_dl_score):
        """
        Compare all models and select the best overall
        """
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        # Create comparison DataFrame
        comparison_data = []
        
        # Add XGBoost results
        comparison_data.append({
            'Model': 'XGBoost (Tuned)',
            'F1_Score': best_xgb_score,
            'Type': 'Traditional ML'
        })
        
        # Add deep learning results
        if dl_models:
            for name, model_info in dl_models.items():
                comparison_data.append({
                    'Model': f'Deep Learning - {name}',
                    'F1_Score': model_info['score'],
                    'Type': 'Deep Learning'
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1_Score', ascending=False)
        
        print("Model Performance Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Select best overall model
        best_overall = comparison_df.iloc[0]['Model']
        best_score = comparison_df.iloc[0]['F1_Score']
        
        print(f"\n🏆 Best Overall Model: {best_overall}")
        print(f"Best F1-Score: {best_score:.4f}")
        
        return comparison_df, best_overall, best_score

    def detailed_comparison_analysis(self, best_xgb_model, best_xgb_score, dl_models, best_dl_name, best_dl_score):
        """
        Perform detailed analysis comparing traditional ML vs Deep Learning
        """
        print("\n" + "="*60)
        print("DETAILED COMPARISON ANALYSIS")
        print("="*60)
        
        # Traditional ML Analysis
        print("\n📊 Traditional Machine Learning (XGBoost):")
        print(f"  F1-Score: {best_xgb_score:.4f}")
        print(f"  Advantages:")
        print(f"    - Fast training and inference")
        print(f"    - Interpretable feature importance")
        print(f"    - Works well with tabular data")
        print(f"    - Robust to overfitting")
        print(f"  Disadvantages:")
        print(f"    - Limited text understanding")
        print(f"    - Requires feature engineering")
        
        # Deep Learning Analysis
        if dl_models:
            print(f"\n🧠 Deep Learning Models:")
            print(f"  Best Model: {best_dl_name}")
            print(f"  Best F1-Score: {best_dl_score:.4f}")
            print(f"  Advantages:")
            print(f"    - Better text understanding")
            print(f"    - Automatic feature learning")
            print(f"    - Can capture complex patterns")
            print(f"  Disadvantages:")
            print(f"    - Slower training and inference")
            print(f"    - Requires more data")
            print(f"    - Less interpretable")
            print(f"    - Risk of overfitting")
        
        # Performance comparison
        print(f"\n📈 Performance Comparison:")
        if best_xgb_score > best_dl_score:
            print(f"  Traditional ML (XGBoost) performs better by {best_xgb_score - best_dl_score:.4f}")
            print(f"  Recommendation: Use XGBoost for production")
        elif best_dl_score > best_xgb_score:
            print(f"  Deep Learning performs better by {best_dl_score - best_xgb_score:.4f}")
            print(f"  Recommendation: Consider deep learning for production")
        else:
            print(f"  Both approaches perform similarly")
            print(f"  Recommendation: Use XGBoost for simplicity and speed")
        
        # Detailed deep learning analysis if available
        if dl_models:
            self.detailed_deep_learning_analysis(dl_models, best_dl_name, best_dl_score)
            
        # Analyze text patterns using utils function
        if self.tokenizer:
            analyze_text_patterns(self.tokenizer, X_train, X_test)

    def detailed_deep_learning_analysis(self, dl_models, best_dl_name, best_dl_score):
        """
        Perform detailed analysis of deep learning models
        """
        print(f"\n🔍 Detailed Deep Learning Analysis")
        print("="*40)
        
        for name, model_info in dl_models.items():
            print(f"\n📊 {name}:")
            print(f"  F1-Score: {model_info['score']:.4f}")
            
            # Analyze training history
            history = model_info['history']
            final_train_acc = history.history['accuracy'][-1]
            final_val_acc = history.history['val_accuracy'][-1]
            final_train_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            
            print(f"  Final Training Accuracy: {final_train_acc:.4f}")
            print(f"  Final Validation Accuracy: {final_val_acc:.4f}")
            print(f"  Final Training Loss: {final_train_loss:.4f}")
            print(f"  Final Validation Loss: {final_val_loss:.4f}")
            
            # Overfitting analysis
            acc_gap = final_train_acc - final_val_acc
            loss_gap = final_val_loss - final_train_loss
            
            if acc_gap > 0.1 or loss_gap > 0.1:
                print(f"  ⚠️  Significant overfitting detected")
            elif acc_gap > 0.05 or loss_gap > 0.05:
                print(f"  ⚠️  Moderate overfitting detected")
            else:
                print(f"  ✅ No significant overfitting")
            
            # Model complexity
            total_params = model_info['model'].count_params()
            print(f"  Model Parameters: {total_params:,}")
            
            if name == best_dl_name:
                print(f"  🏆 BEST DEEP LEARNING MODEL")
        
        # Text analysis insights
        print(f"\n📝 Text Analysis Insights:")
        print(f"  Vocabulary Size: {len(self.tokenizer.word_index) + 1:,}")
        print(f"  Most Common Words:")
        
        # Get most common words
        word_counts = self.tokenizer.word_counts
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        for i, (word, count) in enumerate(sorted_words[:10]):
            print(f"    {i+1:2d}. '{word}': {count:,} occurrences")
        
        # Analyze text patterns
        print(f"\n🔍 Text Pattern Analysis:")
        
        # Use clean_text function from utils.py
        
        # Analyze text patterns
        from utils import build_dataset
        X_train, y_train, X_test, y_test = build_dataset()
        
        train_texts = [item.get('title', '') for item in X_train]
        test_texts = [item.get('title', '') for item in X_test]
        
        # Count words related to condition
        condition_words = {
            'new': ['nuevo', 'original', 'originales', 'nueva', 'nuevos'],
            'used': ['usado', 'usada', 'usados', 'usadas', 'segunda', 'segundo', 'antiguo', 'antigua']
        }
        
        print(f"  Condition-related words in titles:")
        for condition, words in condition_words.items():
            count = 0
            for text in train_texts + test_texts:
                text_clean = clean_text(text)
                if any(word in text_clean for word in words):
                    count += 1
            print(f"    {condition.capitalize()}: {count:,} titles")

    def save_models(self, best_xgb_model, best_xgb_params, dl_models, best_dl_name):
        """
        Save all trained models
        """
        print("\n" + "="*60)
        print("SAVING MODELS AND PARAMETERS")
        print("="*60)
        
        # Save tuned XGBoost model
        save_model(best_xgb_model, "tuned_xgboost", "models/tuned_xgboost_model.pkl")
        
        # Save XGBoost parameters
        with open('models/tuned_xgboost_params.pkl', 'wb') as f:
            pickle.dump(best_xgb_params, f)
        
        # Save best deep learning model if available
        if dl_models and best_dl_name:
            best_dl_model = dl_models[best_dl_name]['model']
            best_dl_model.save(f'models/best_deep_learning_model_{best_dl_name.lower().replace(" ", "_")}.h5')
        
        # Save tokenizer if available
        if self.tokenizer:
            with open('models/text_tokenizer.pkl', 'wb') as f:
                pickle.dump(self.tokenizer, f)
        
        # Save label encoder
        with open('models/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print("✅ Models saved successfully!")
        print("- models/tuned_xgboost_model.pkl")
        print("- models/tuned_xgboost_params.pkl")
        if dl_models and best_dl_name:
            print(f"- models/best_deep_learning_model_{best_dl_name.lower().replace(' ', '_')}.h5")
        if self.tokenizer:
            print("- models/text_tokenizer.pkl")
        print("- models/label_encoder.pkl")
    
    def run_complete_analysis(self):
        """
        Run the complete advanced modeling analysis
        """
        print("🚀 STARTING ADVANCED MODELING ANALYSIS")
        print("="*60)
        
        # Load and prepare data
        X_train, X_test, y_train, y_test, train_df, test_df, y_train_encoded, y_test_encoded = self.load_and_prepare_data()
        
        # Hyperparameter tuning for XGBoost
        best_xgb_model, best_xgb_params, best_xgb_score = self.hyperparameter_tuning_xgboost(
            X_train, y_train, X_test, y_test
        )
        
        # Deep learning models
        dl_models = {}
        best_dl_name = None
        best_dl_score = 0
        
        if self.enable_deep_learning:
            print("\n" + "="*60)
            print("DEEP LEARNING NEURAL NETWORK APPROACH")
            print("="*60)
            
            # Prepare text data for deep learning
            X_train_padded, X_test_padded, y_train_dl, y_test_dl = self.prepare_text_data_for_deep_learning(
                X_train, X_test, y_train, y_test
            )
            
            if X_train_padded is not None:
                # Build and train deep learning models
                dl_models, best_dl_name, best_dl_score = self.build_deep_learning_models(
                    X_train_padded, X_test_padded, y_train_dl, y_test_dl
                )
                
                # Plot training history
                self.plot_training_history(dl_models)
        else:
            print("\n⚠️  Deep learning is disabled. Skipping neural network approach.")
            print("   To enable deep learning, install TensorFlow: pip install tensorflow")
        
        # Compare all models
        comparison_df, best_overall, best_score = self.compare_all_models(
            best_xgb_model, best_xgb_score, dl_models, best_dl_name, best_dl_score
        )
        
        # Save models
        self.save_models(best_xgb_model, best_xgb_params, dl_models, best_dl_name)
        
        # Detailed comparison analysis
        self.detailed_comparison_analysis(best_xgb_model, best_xgb_score, dl_models, best_dl_name, best_dl_score)
        
        print("\n" + "="*60)
        print("🎉 ADVANCED MODELING ANALYSIS COMPLETED!")
        print("="*60)
        
        return {
            'best_xgb_model': best_xgb_model,
            'best_xgb_params': best_xgb_params,
            'best_xgb_score': best_xgb_score,
            'dl_models': dl_models,
            'best_dl_name': best_dl_name,
            'best_dl_score': best_dl_score,
            'comparison_df': comparison_df,
            'best_overall': best_overall,
            'best_score': best_score
        }

    def run_deep_learning_only(self):
        """
        Run only deep learning analysis (skip XGBoost tuning)
        """
        print("🚀 STARTING DEEP LEARNING ONLY ANALYSIS")
        print("="*60)
        
        # Load and prepare data
        X_train, X_test, y_train, y_test, train_df, test_df, y_train_encoded, y_test_encoded = self.load_and_prepare_data()
        
        if not self.enable_deep_learning:
            print("❌ Deep learning is disabled. Cannot run deep learning only analysis.")
            return None
        
        # Prepare text data for deep learning
        X_train_padded, X_test_padded, y_train_dl, y_test_dl = self.prepare_text_data_for_deep_learning(
            X_train, X_test, y_train, y_test
        )
        
        if X_train_padded is not None:
            # Build and train deep learning models
            dl_models, best_dl_name, best_dl_score = self.build_deep_learning_models(
                X_train_padded, X_test_padded, y_train_dl, y_test_dl
            )
            
            # Plot training history
            self.plot_training_history(dl_models)
            
            # Save models
            self.save_models(None, None, dl_models, best_dl_name)
            
            print("\n" + "="*60)
            print("🎉 DEEP LEARNING ANALYSIS COMPLETED!")
            print("="*60)
            
            return {
                'dl_models': dl_models,
                'best_dl_name': best_dl_name,
                'best_dl_score': best_dl_score,
                'best_overall': f'Deep Learning - {best_dl_name}',
                'best_score': best_dl_score
            }
        else:
            print("❌ Failed to prepare text data for deep learning.")
            return None

def main():
    """
    Main function to run the advanced modeling analysis
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Modeling for New vs Used Classification')
    parser.add_argument('--tuning-level', choices=['simple', 'comprehensive', 'full'], 
                       default='simple', help='Hyperparameter tuning level')
    parser.add_argument('--no-deep-learning', action='store_true', 
                       help='Disable deep learning models')
    parser.add_argument('--deep-learning-only', action='store_true',
                       help='Run only deep learning models (skip XGBoost tuning)')
    
    args = parser.parse_args()
    
    # Check if TensorFlow is available
    if not TENSORFLOW_AVAILABLE and not args.no_deep_learning:
        print("⚠️  TensorFlow not found. Deep learning will be disabled.")
        print("   Install with: pip install tensorflow")
        args.no_deep_learning = True
    
    # Initialize and run analysis
    analyzer = AdvancedModeling(
        tuning_level=args.tuning_level,
        enable_deep_learning=not args.no_deep_learning
    )
    
    if args.deep_learning_only:
        results = analyzer.run_deep_learning_only()
    else:
        results = analyzer.run_complete_analysis()
    
    print("\n📊 Final Results Summary:")
    print(f"Best Overall Model: {results['best_overall']}")
    print(f"Best Score: {results['best_score']:.4f}")
    if not args.deep_learning_only:
        print(f"XGBoost Tuned Score: {results['best_xgb_score']:.4f}")
    if results['best_dl_score'] > 0:
        print(f"Best Deep Learning Score: {results['best_dl_score']:.4f}")

if __name__ == "__main__":
    main() 