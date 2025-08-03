"""
Common utilities for New vs Used Item Classification
==================================================

This module contains shared functions used by both new_or_used.py and advanced_modeling.py
to reduce code duplication and improve maintainability.

Author: Diego
Date: 2025
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re
import warnings
from typing import Dict, List, Tuple, Any

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
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
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional, GlobalMaxPooling1D
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')

# Spanish stop words for text processing
SPANISH_STOP_WORDS = [
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

def build_dataset():
    """
    Build the dataset from the JSON lines file
    """
    data = [json.loads(x) for x in open("MLA_100k.jsonlines")]
    target = lambda x: x.get("condition")
    N = -10000
    X_train = data[:N]
    X_test = data[N:]
    y_train = [target(x) for x in X_train]
    y_test = [target(x) for x in X_test]
    for x in X_test:
        del x["condition"]
    return X_train, y_train, X_test, y_test

def clean_text(text):
    """
    Clean and preprocess text for NLP tasks
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep Spanish characters
    text = re.sub(r'[^a-záéíóúñü\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def extract_features(data):
    """
    Extract and engineer features from the raw data
    """
    features = []
    
    for item in data:
        feature_dict = {}
        
        # Basic numerical features
        feature_dict['price'] = item.get('price', 0)
        feature_dict['base_price'] = item.get('base_price', 0)
        feature_dict['initial_quantity'] = item.get('initial_quantity', 1)
        feature_dict['available_quantity'] = item.get('available_quantity', 1)
        feature_dict['sold_quantity'] = item.get('sold_quantity', 0)
        
        # Price-related features
        if item.get('original_price'):
            feature_dict['has_original_price'] = 1
            feature_dict['price_discount'] = item.get('original_price', 0) - item.get('price', 0)
        else:
            feature_dict['has_original_price'] = 0
            feature_dict['price_discount'] = 0
            
        # Categorical features - ensure no empty strings
        category_id = item.get('category_id')
        feature_dict['category_id'] = str(category_id) if category_id is not None else 'unknown'
        
        listing_type_id = item.get('listing_type_id')
        feature_dict['listing_type_id'] = str(listing_type_id) if listing_type_id is not None else 'unknown'
        
        buying_mode = item.get('buying_mode')
        feature_dict['buying_mode'] = str(buying_mode) if buying_mode is not None else 'unknown'
        
        site_id = item.get('site_id')
        feature_dict['site_id'] = str(site_id) if site_id is not None else 'unknown'
        
        currency_id = item.get('currency_id')
        feature_dict['currency_id'] = str(currency_id) if currency_id is not None else 'unknown'
        
        # Boolean features
        accepts_mercadopago = item.get('accepts_mercadopago')
        feature_dict['accepts_mercadopago'] = 1 if accepts_mercadopago is True else 0
        
        automatic_relist = item.get('automatic_relist')
        feature_dict['automatic_relist'] = 1 if automatic_relist is True else 0
        
        # Shipping features
        shipping = item.get('shipping', {})
        free_shipping = shipping.get('free_shipping')
        feature_dict['free_shipping'] = 1 if free_shipping is True else 0
        
        local_pick_up = shipping.get('local_pick_up')
        feature_dict['local_pick_up'] = 1 if local_pick_up is True else 0
        
        # Text features
        title = item.get('title', '')
        feature_dict['title_length'] = len(title)
        feature_dict['title_word_count'] = len(title.split())
        
        # Text preprocessing for title
        title_lower = title.lower()
        feature_dict['has_usado'] = 1 if 'usado' in title_lower else 0
        feature_dict['has_nuevo'] = 1 if 'nuevo' in title_lower else 0
        feature_dict['has_original'] = 1 if 'original' in title_lower else 0
        feature_dict['has_antiguo'] = 1 if 'antiguo' in title_lower else 0
        feature_dict['has_vintage'] = 1 if 'vintage' in title_lower else 0
        feature_dict['has_segunda'] = 1 if 'segunda' in title_lower else 0
        
        # Seller features
        seller_id = item.get('seller_id')
        feature_dict['seller_id'] = str(seller_id) if seller_id is not None else 'unknown'
        
        # Date features
        if item.get('date_created'):
            try:
                date_created = pd.to_datetime(item['date_created'])
                feature_dict['listing_age_days'] = (pd.Timestamp.now() - date_created).days
            except:
                feature_dict['listing_age_days'] = 0
        else:
            feature_dict['listing_age_days'] = 0
            
        # Pictures features
        pictures = item.get('pictures', [])
        feature_dict['num_pictures'] = len(pictures)
        
        # Attributes features
        attributes = item.get('attributes', [])
        feature_dict['num_attributes'] = len(attributes)
        
        # Variations features
        variations = item.get('variations', [])
        feature_dict['num_variations'] = len(variations)
        
        # Tags features
        tags = item.get('tags', [])
        feature_dict['num_tags'] = len(tags)
        feature_dict['has_dragged_bids'] = 1 if 'dragged_bids_and_visits' in tags else 0
        
        # Warranty features
        warranty = item.get('warranty')
        feature_dict['has_warranty'] = 1 if warranty is not None else 0
        
        # Official store features
        official_store_id = item.get('official_store_id')
        feature_dict['has_official_store'] = 1 if official_store_id is not None else 0
        
        # Catalog product features
        catalog_product_id = item.get('catalog_product_id')
        feature_dict['has_catalog_product'] = 1 if catalog_product_id is not None else 0
        
        # Video features
        video_id = item.get('video_id')
        feature_dict['has_video'] = 1 if video_id is not None else 0
        
        # Subtitle features
        subtitle = item.get('subtitle')
        feature_dict['has_subtitle'] = 1 if subtitle is not None else 0
        
        # Differential pricing features
        differential_pricing = item.get('differential_pricing')
        feature_dict['has_differential_pricing'] = 1 if differential_pricing is not None else 0
        
        # International delivery features
        international_delivery_mode = item.get('international_delivery_mode')
        feature_dict['international_delivery'] = 1 if international_delivery_mode != 'none' else 0
        
        # Coverage areas features
        coverage_areas = item.get('coverage_areas', [])
        feature_dict['num_coverage_areas'] = len(coverage_areas)
        
        # Deal IDs features
        deal_ids = item.get('deal_ids', [])
        feature_dict['num_deals'] = len(deal_ids)
        
        # Sub status features
        sub_status = item.get('sub_status', [])
        feature_dict['num_sub_status'] = len(sub_status)
        
        # Descriptions features
        descriptions = item.get('descriptions', [])
        feature_dict['num_descriptions'] = len(descriptions)
        
        # Location features
        location = item.get('location', {})
        feature_dict['has_location'] = 1 if location is not None and location else 0
        
        # Geolocation features
        geolocation = item.get('geolocation', {})
        if geolocation:
            feature_dict['latitude'] = geolocation.get('latitude', 0)
            feature_dict['longitude'] = geolocation.get('longitude', 0)
        else:
            feature_dict['latitude'] = 0
            feature_dict['longitude'] = 0
            
        # Seller address features
        seller_address = item.get('seller_address', {})
        if seller_address:
            country_name = seller_address.get('country', {}).get('name', 'unknown')
            state_name = seller_address.get('state', {}).get('name', 'unknown')
            city_name = seller_address.get('city', {}).get('name', 'unknown')
            
            feature_dict['seller_country'] = country_name or 'unknown'
            feature_dict['seller_state'] = state_name or 'unknown'
            feature_dict['seller_city'] = city_name or 'unknown'
        else:
            feature_dict['seller_country'] = 'unknown'
            feature_dict['seller_state'] = 'unknown'
            feature_dict['seller_city'] = 'unknown'
            
        features.append(feature_dict)
    
    return features

def create_feature_matrix(features):
    """
    Convert features list to pandas DataFrame and prepare for modeling
    """
    df = pd.DataFrame(features)
    
    # Separate text and categorical features
    text_features = ['title']
    # Exclude high-cardinality columns from categorical_features
    categorical_features = ['category_id', 'listing_type_id', 'buying_mode', 'site_id', 'currency_id', 'seller_country']
    # Ensure all categorical columns are string and have no empty or NaN values
    for col in categorical_features:
        df[col] = df[col].fillna('unknown').replace('', 'unknown').astype(str)
    # Only select numeric columns for numerical_features
    numerical_features = [col for col in df.columns if col not in text_features + categorical_features and pd.api.types.is_numeric_dtype(df[col])]
    
    return df, text_features, categorical_features, numerical_features

def preprocess_data(X_train, X_test, y_train, y_test):
    """
    Preprocess the data for modeling
    """
    print("Extracting features from training data...")
    train_features = extract_features(X_train)
    print("Extracting features from test data...")
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
    
    return train_df, test_df, y_train_encoded, y_test_encoded, text_features, categorical_features, numerical_features

def build_models(text_features, categorical_features, numerical_features, feature_selector=None):
    """
    Build multiple machine learning models for comparison
    """
    # Text preprocessing with Spanish stop words
    text_transformer = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, stop_words=SPANISH_STOP_WORDS, ngram_range=(1, 2)))
    ])
    
    # Categorical preprocessing - use OneHotEncoder for multiple columns
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
    
    # Define multiple models with feature selection if provided
    if feature_selector is not None:
        # Add feature selection to the pipeline
        models = {
            'Random Forest': Pipeline([
                ('preprocessor', preprocessor),
                ('feature_selector', feature_selector),
                ('classifier', RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', bootstrap=True, oob_score=True, random_state=42, n_jobs=-1, verbose=1))
            ]),
            'XGBoost': Pipeline([
                ('preprocessor', preprocessor),
                ('feature_selector', feature_selector),
                ('classifier', xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=1, n_jobs=-1))
            ]),
            'CatBoost': Pipeline([
                ('preprocessor', preprocessor),
                ('feature_selector', feature_selector),
                ('classifier', cb.CatBoostClassifier(iterations=200, depth=5, learning_rate=0.1, l2_leaf_reg=3.0, random_strength=1.0, random_state=42, verbose=50, task_type='CPU'))
            ]),
            'Neural Network': Pipeline([
                ('preprocessor', preprocessor),
                ('feature_selector', feature_selector),
                ('classifier', MLPClassifier(hidden_layer_sizes=(80, 40), max_iter=200, alpha=0.01, learning_rate_init=0.001, random_state=42, verbose=True, early_stopping=True, validation_fraction=0.2))
            ]),
            'Logistic Regression': Pipeline([
                ('preprocessor', preprocessor),
                ('feature_selector', feature_selector),
                ('classifier', LogisticRegression(C=0.1, penalty='l2', random_state=42, max_iter=200, n_jobs=-1, verbose=1))
            ]),
            'Decision Tree': Pipeline([
                ('preprocessor', preprocessor),
                ('feature_selector', feature_selector),
                ('classifier', DecisionTreeClassifier(random_state=42))
            ]),
            'K-Nearest Neighbors': Pipeline([
                ('preprocessor', preprocessor),
                ('feature_selector', feature_selector),
                ('classifier', KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
            ])
        }
    else:
        # Models without feature selection
        models = {
            'Random Forest': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', bootstrap=True, oob_score=True, random_state=42, n_jobs=-1, verbose=1))
            ]),
            'XGBoost': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=1, n_jobs=-1))
            ]),
            'CatBoost': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', cb.CatBoostClassifier(iterations=200, depth=5, learning_rate=0.1, l2_leaf_reg=3.0, random_strength=1.0, random_state=42, verbose=50, task_type='CPU'))
            ]),
            'Neural Network': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', MLPClassifier(hidden_layer_sizes=(80, 40), max_iter=200, alpha=0.01, learning_rate_init=0.001, random_state=42, verbose=True, early_stopping=True, validation_fraction=0.2))
            ]),
            'Logistic Regression': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(C=0.1, penalty='l2', random_state=42, max_iter=200, n_jobs=-1, verbose=1))
            ]),
            'Decision Tree': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', DecisionTreeClassifier(random_state=42))
            ]),
            'K-Nearest Neighbors': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
            ])
        }
    
    return models

def evaluate_model(model, X_train, y_train_encoded, X_test, y_test, y_test_encoded):
    """
    Evaluate the model and print detailed metrics with overfitting detection
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Make predictions on both training and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics for both sets
    train_accuracy = accuracy_score(y_train_encoded, y_train_pred)
    test_accuracy = accuracy_score(y_test_encoded, y_test_pred)
    train_f1 = f1_score(y_train_encoded, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test_encoded, y_test_pred, average='weighted')
    
    # Overfitting detection
    accuracy_gap = train_accuracy - test_accuracy
    f1_gap = train_f1 - test_f1
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Accuracy Gap (Train-Test): {accuracy_gap:.4f}")
    print(f"Training F1-Score: {train_f1:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    print(f"F1-Score Gap (Train-Test): {f1_gap:.4f}")
    
    # Overfitting assessment
    print(f"\n{'='*30}")
    print("OVERFITTING ASSESSMENT")
    print(f"{'='*30}")
    if accuracy_gap > 0.05 or f1_gap > 0.05:
        print("⚠️  WARNING: Potential overfitting detected!")
        print(f"   - Accuracy gap: {accuracy_gap:.4f} (> 0.05)")
        print(f"   - F1-score gap: {f1_gap:.4f} (> 0.05)")
    elif accuracy_gap > 0.02 or f1_gap > 0.02:
        print("⚠️  CAUTION: Moderate overfitting detected!")
        print(f"   - Accuracy gap: {accuracy_gap:.4f} (> 0.02)")
        print(f"   - F1-score gap: {f1_gap:.4f} (> 0.02)")
    else:
        print("✅ No significant overfitting detected!")
        print(f"   - Accuracy gap: {accuracy_gap:.4f} (≤ 0.02)")
        print(f"   - F1-score gap: {f1_gap:.4f} (≤ 0.02)")
    
    # Detailed classification report for test set
    print(f"\nDetailed Classification Report (Test Set):")
    print(classification_report(y_test_encoded, y_test_pred, target_names=['new', 'used']))
    
    # Confusion matrix for test set
    cm = confusion_matrix(y_test_encoded, y_test_pred)
    print("\nConfusion Matrix (Test Set):")
    print(cm)
    
    return test_accuracy, test_f1

def plot_roc_curve_and_confusion_matrix(model, X_test, y_test, y_test_encoded, model_name):
    """
    Plot ROC curve and confusion matrix for the champion model
    """
    print(f"\n{'='*60}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_encoded, y_pred)
    f1 = f1_score(y_test_encoded, y_pred, average='weighted')
    
    print(f"Champion Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test_encoded, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'ROC Curve - {model_name}')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test_encoded, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, 
                xticklabels=['New', 'Used'], yticklabels=['New', 'Used'])
    ax2.set_title(f'Confusion Matrix - {model_name}')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    # Add performance metrics to confusion matrix
    ax2.text(0.5, -0.15, f'Accuracy: {accuracy:.3f}\nF1-Score: {f1:.3f}', 
             ha='center', va='center', transform=ax2.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'model_results/champion_model_performance_{model_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    print(f"📊 Visualizations saved as: model_results/champion_model_performance_{model_name.lower().replace(' ', '_')}.png")
    plt.show()
    
    return accuracy, f1

def save_model(model, model_name, filepath=None):
    """
    Save a model to a pickle file
    """
    if filepath is None:
        filepath = f"models/champion_model_{model_name.lower().replace(' ', '_')}.pkl"
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"💾 Model saved as: {filepath}")
    return filepath

def load_model(filepath):
    """
    Load a model from a pickle file
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"📂 Model loaded from: {filepath}")
    return model

# Deep Learning Utilities
def prepare_text_data_for_deep_learning(X_train, X_test, y_train, y_test, max_words=10000, max_len=100):
    """
    Prepare text data for deep learning models
    
    Args:
        X_train: Training data
        X_test: Test data
        y_train: Training labels
        y_test: Test labels
        max_words: Maximum vocabulary size
        max_len: Maximum sequence length
        
    Returns:
        Tuple of (X_train_padded, X_test_padded, y_train_dl, y_test_dl, tokenizer)
    """
    if not TENSORFLOW_AVAILABLE:
        print("⚠️  TensorFlow not available. Cannot prepare text data for deep learning.")
        return None, None, None, None, None
    
    print("\n📝 Preparing text data for deep learning...")
    
    # Extract text data
    train_texts = [item.get('title', '') for item in X_train]
    test_texts = [item.get('title', '') for item in X_test]
    
    # Clean texts
    train_texts_clean = [clean_text(text) for text in train_texts]
    test_texts_clean = [clean_text(text) for text in test_texts]
    
    # Tokenize texts
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_texts_clean)
    
    # Convert texts to sequences
    train_sequences = tokenizer.texts_to_sequences(train_texts_clean)
    test_sequences = tokenizer.texts_to_sequences(test_texts_clean)
    
    # Pad sequences
    X_train_padded = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')
    X_test_padded = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')
    
    # Prepare labels
    y_train_dl = np.array([1 if label == 'used' else 0 for label in y_train])
    y_test_dl = np.array([1 if label == 'used' else 0 for label in y_test])
    
    print(f"✓ Text data prepared:")
    print(f"  Vocabulary size: {len(tokenizer.word_index) + 1}")
    print(f"  Training sequences shape: {X_train_padded.shape}")
    print(f"  Test sequences shape: {X_test_padded.shape}")
    
    return X_train_padded, X_test_padded, y_train_dl, y_test_dl, tokenizer

def build_deep_learning_models(X_train_padded, X_test_padded, y_train_encoded, y_test_encoded):
    """
    Build and train deep learning models
    
    Args:
        X_train_padded: Padded training sequences
        X_test_padded: Padded test sequences
        y_train_encoded: Encoded training labels
        y_test_encoded: Encoded test labels
        
    Returns:
        Dictionary of trained models with their scores
    """
    if not TENSORFLOW_AVAILABLE:
        print("⚠️  TensorFlow not available. Cannot build deep learning models.")
        return {}, None, 0
    
    print("\n🧠 Building Deep Learning Models")
    print("="*60)
    
    models = {}
    best_model_name = None
    best_score = 0
    
    # Model 1: Simple LSTM
    print("\n📊 Training Simple LSTM...")
    model1 = Sequential([
        Embedding(10000, 128, input_length=100),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(16, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model1.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Train model
    history1 = model1.fit(
        X_train_padded, y_train_encoded,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate
    y_pred1 = (model1.predict(X_test_padded) > 0.5).astype(int)
    score1 = f1_score(y_test_encoded, y_pred1, average='weighted')
    
    models['Simple LSTM'] = {
        'model': model1,
        'history': history1,
        'score': score1
    }
    
    print(f"Simple LSTM F1-Score: {score1:.4f}")
    
    # Model 2: Bidirectional LSTM
    print("\n📊 Training Bidirectional LSTM...")
    model2 = Sequential([
        Embedding(10000, 128, input_length=100),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),
        Dense(16, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model2.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history2 = model2.fit(
        X_train_padded, y_train_encoded,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate
    y_pred2 = (model2.predict(X_test_padded) > 0.5).astype(int)
    score2 = f1_score(y_test_encoded, y_pred2, average='weighted')
    
    models['Bidirectional LSTM'] = {
        'model': model2,
        'history': history2,
        'score': score2
    }
    
    print(f"Bidirectional LSTM F1-Score: {score2:.4f}")
    
    # Model 3: CNN + LSTM
    print("\n📊 Training CNN + LSTM...")
    model3 = Sequential([
        Embedding(10000, 128, input_length=100),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model3.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history3 = model3.fit(
        X_train_padded, y_train_encoded,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate
    y_pred3 = (model3.predict(X_test_padded) > 0.5).astype(int)
    score3 = f1_score(y_test_encoded, y_pred3, average='weighted')
    
    models['CNN + LSTM'] = {
        'model': model3,
        'history': history3,
        'score': score3
    }
    
    print(f"CNN + LSTM F1-Score: {score3:.4f}")
    
    # Find best model
    for name, model_info in models.items():
        if model_info['score'] > best_score:
            best_score = model_info['score']
            best_model_name = name
    
    print(f"\n🏆 Best Deep Learning Model: {best_model_name}")
    print(f"Best F1-Score: {best_score:.4f}")
    
    return models, best_model_name, best_score

def analyze_text_patterns(tokenizer, X_train, X_test):
    """
    Analyze text patterns in the dataset
    
    Args:
        tokenizer: Fitted tokenizer
        X_train: Training data
        X_test: Test data
        
    Returns:
        Dictionary with analysis results
    """
    print(f"\n📝 Text Analysis Insights:")
    print(f"  Vocabulary Size: {len(tokenizer.word_index) + 1:,}")
    print(f"  Most Common Words:")
    
    # Get most common words
    word_counts = tokenizer.word_counts
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    for i, (word, count) in enumerate(sorted_words[:10]):
        print(f"    {i+1:2d}. '{word}': {count:,} occurrences")
    
    # Analyze text patterns
    print(f"\n🔍 Text Pattern Analysis:")
    
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
    
    return {
        'vocabulary_size': len(tokenizer.word_index) + 1,
        'most_common_words': sorted_words[:10],
        'condition_word_counts': condition_words
    }

def compare_model_performance(model_results):
    """
    Compare performance of multiple models
    
    Args:
        model_results: Dictionary with model names and their scores
        
    Returns:
        DataFrame with comparison results
    """
    comparison_data = []
    
    for name, results in model_results.items():
        comparison_data.append({
            'Model': name,
            'F1_Score': results.get('score', 0),
            'Type': results.get('type', 'Unknown')
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('F1_Score', ascending=False)
    
    print("Model Performance Comparison:")
    print(comparison_df.to_string(index=False))
    
    return comparison_df 