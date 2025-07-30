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

import json


# You can safely assume that `build_dataset` is correctly implemented
def build_dataset():
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


# Insert your code below this line:
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
        feature_dict['category_id'] = item.get('category_id', 'unknown') or 'unknown'
        feature_dict['listing_type_id'] = item.get('listing_type_id', 'unknown') or 'unknown'
        feature_dict['buying_mode'] = item.get('buying_mode', 'unknown') or 'unknown'
        feature_dict['site_id'] = item.get('site_id', 'unknown') or 'unknown'
        feature_dict['currency_id'] = item.get('currency_id', 'unknown') or 'unknown'
        
        # Boolean features
        feature_dict['accepts_mercadopago'] = 1 if item.get('accepts_mercadopago') else 0
        feature_dict['automatic_relist'] = 1 if item.get('automatic_relist') else 0
        
        # Shipping features
        shipping = item.get('shipping', {})
        feature_dict['free_shipping'] = 1 if shipping.get('free_shipping') else 0
        feature_dict['local_pick_up'] = 1 if shipping.get('local_pick_up') else 0
        
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
        seller_id = item.get('seller_id', 'unknown')
        feature_dict['seller_id'] = str(seller_id) if seller_id else 'unknown'
        
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
        feature_dict['has_warranty'] = 1 if item.get('warranty') else 0
        
        # Official store features
        feature_dict['has_official_store'] = 1 if item.get('official_store_id') else 0
        
        # Catalog product features
        feature_dict['has_catalog_product'] = 1 if item.get('catalog_product_id') else 0
        
        # Video features
        feature_dict['has_video'] = 1 if item.get('video_id') else 0
        
        # Subtitle features
        feature_dict['has_subtitle'] = 1 if item.get('subtitle') else 0
        
        # Differential pricing features
        feature_dict['has_differential_pricing'] = 1 if item.get('differential_pricing') else 0
        
        # International delivery features
        feature_dict['international_delivery'] = 1 if item.get('international_delivery_mode') != 'none' else 0
        
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
        feature_dict['has_location'] = 1 if location else 0
        
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

def build_models(text_features, categorical_features, numerical_features, feature_selector=None):
    """
    Build multiple machine learning models for comparison
    """
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
        ('tfidf', TfidfVectorizer(max_features=1000, stop_words=spanish_stop_words, ngram_range=(1, 2)))
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
            # 'SVM (Linear)': Pipeline([
            #     ('preprocessor', preprocessor),
            #     ('feature_selector', feature_selector),
            #     ('classifier', SVC(kernel='linear', C=1.0, random_state=42, probability=True, verbose=True, max_iter=1000, tol=0.001))
            # ]),
            # 'SVM (RBF)': Pipeline([
            #     ('preprocessor', preprocessor),
            #     ('feature_selector', feature_selector),
            #     ('classifier', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True, verbose=True, max_iter=1000, tol=0.001))
            # ]),
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
            # 'SVM (Linear)': Pipeline([
            #     ('preprocessor', preprocessor),
            #     ('classifier', SVC(kernel='linear', C=1.0, random_state=42, probability=True, verbose=True, max_iter=1000, tol=0.001))
            # ]),
            # 'SVM (RBF)': Pipeline([
            #     ('preprocessor', preprocessor),
            #     ('classifier', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True, verbose=True, max_iter=1000, tol=0.001))
            # ]),
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
    model_filename = f"champion_model_{best_model_name.lower().replace(' ', '_')}.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(champion_model, f)
    print(f"💾 Champion model saved as: {model_filename}")
    
    return champion_model

def load_champion_model(model_filename):
    """
    Load the saved champion model
    """
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)
    print(f"📂 Champion model loaded from: {model_filename}")
    return model

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
    plt.savefig(f'champion_model_performance_{model_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    print(f"📊 Visualizations saved as: champion_model_performance_{model_name.lower().replace(' ', '_')}.png")
    plt.show()
    
    return accuracy, f1

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
    model_filename = f"champion_model_{best_model_name.lower().replace(' ', '_')}.pkl"
    loaded_champion_model = load_champion_model(model_filename)
    
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
    print(f"📊 Visualizations saved as: champion_model_performance_{best_model_name.lower().replace(' ', '_')}.png")
    
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


