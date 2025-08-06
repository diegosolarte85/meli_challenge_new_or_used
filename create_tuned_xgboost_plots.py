"""
Create plots for tuned XGBoost model
====================================

This script loads the tuned XGBoost model and creates ROC curve and confusion matrix plots.

Author: Diego
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import traceback
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

# Import shared utilities
from utils import build_dataset, extract_features, create_feature_matrix, preprocess_data

def load_tuned_model():
    """Load the tuned XGBoost model and parameters."""
    try:
        print("Attempting to load tuned model files...")
        with open('models/tuned_xgboost_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✓ Model loaded successfully")
        
        with open('models/tuned_xgboost_params.pkl', 'rb') as f:
            params = pickle.load(f)
        print("✓ Parameters loaded successfully")
        
        return model, params
    except FileNotFoundError as e:
        print(f"Error: Tuned model files not found: {e}")
        print("Please run advanced_modeling.py first to generate the tuned model.")
        return None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return None, None

def prepare_data():
    """Prepare the data using the same pipeline as new_or_used.py."""
    try:
        print("Importing functions from utils.py...")
        
        print("Loading and processing data...")
        
        # Build dataset
        print("Building dataset...")
        train_data, train_labels, test_data, test_labels = build_dataset()
        print(f"✓ Dataset built: {len(train_data)} train, {len(test_data)} test samples")
        
        # Extract features
        print("Extracting features from train data...")
        train_features = extract_features(train_data)
        print("✓ Train features extracted")
        
        print("Extracting features from test data...")
        test_features = extract_features(test_data)
        print("✓ Test features extracted")
        
        # Create feature matrices
        print("Creating feature matrices...")
        train_result = create_feature_matrix(train_features)
        test_result = create_feature_matrix(test_features)
        
        # Unpack the results
        train_df, _, _, _ = train_result
        test_df, _, _, _ = test_result
        
        print(f"✓ Feature matrices created: train shape {train_df.shape}, test shape {test_df.shape}")
        
        # Preprocess data for XGBoost (convert categorical to numeric)
        print("Preprocessing data for XGBoost...")
        
        # Identify categorical columns
        categorical_columns = ['category_id', 'listing_type_id', 'buying_mode', 'site_id', 'currency_id', 'seller_country']
        
        # Convert categorical features to numeric using LabelEncoder
        for col in categorical_columns:
            if col in train_df.columns:
                print(f"Converting {col} to numeric...")
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                # Combine unique values from both train and test
                all_values = np.concatenate([train_df[col].values, test_df[col].values])
                le.fit(all_values)
                train_df[col] = le.transform(train_df[col])
                test_df[col] = le.transform(test_df[col])
        
        # Convert all columns to numeric
        for col in train_df.columns:
            if col != 'title':  # Skip text column
                train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
                test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0)
        
        # Remove text column for XGBoost
        if 'title' in train_df.columns:
            train_df = train_df.drop('title', axis=1)
            test_df = test_df.drop('title', axis=1)
        
        print(f"✓ Final processed training set shape: {train_df.shape}")
        print(f"✓ Final processed test set shape: {test_df.shape}")
        
        # Prepare labels
        print("Preparing labels...")
        y_train = [1 if label == 'used' else 0 for label in train_labels]
        y_test = [1 if label == 'used' else 0 for label in test_labels]
        print(f"✓ Labels prepared: train {sum(y_train)}/{len(y_train)} used, test {sum(y_test)}/{len(y_test)} used")
        
        # Convert to numpy arrays
        X_train = train_df.values
        X_test = test_df.values
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"Error preparing data: {e}")
        traceback.print_exc()
        return None, None, None, None

def create_roc_curve(model, X_test, y_test, save_path='model_results/tuned_xgboost_roc_curve.png'):
    """
    Create and save ROC curve plot for the tuned XGBoost model
    """
    print(f"\n📊 Creating ROC curve...")
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Tuned XGBoost Model')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Add AUC text
    plt.text(0.6, 0.2, f'AUC = {roc_auc:.3f}', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
             fontsize=12)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROC curve saved as: {save_path}")
    plt.show()
    
    return roc_auc

def create_confusion_matrix(model, X_test, y_test, save_path='model_results/tuned_xgboost_confusion_matrix.png'):
    """
    Create and save confusion matrix plot for the tuned XGBoost model
    """
    print(f"\n📊 Creating confusion matrix...")
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Create confusion matrix heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['New', 'Used'], yticklabels=['New', 'Used'])
    plt.title('Confusion Matrix - Tuned XGBoost Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Add metrics text
    metrics_text = f'Accuracy: {accuracy:.3f}\nF1-Score: {f1:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}'
    plt.text(0.5, -0.15, metrics_text, 
             ha='center', va='center', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
             fontsize=10)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved as: {save_path}")
    plt.show()
    
    return accuracy, f1, precision, recall

def print_classification_report(model, X_test, y_test):
    """
    Print detailed classification report
    """
    print(f"\n📊 Classification Report:")
    print("="*50)
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Print classification report
    from sklearn.metrics import classification_report
    report = classification_report(y_test, y_pred, target_names=['New', 'Used'])
    print(report)
    
    # Print additional metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    print(f"\n📈 Summary Metrics:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")

def main():
    """
    Main function to create plots for the tuned XGBoost model
    """
    print("🎨 Creating Plots for Tuned XGBoost Model")
    print("="*60)
    
    # Load model
    model, params = load_tuned_model()
    if model is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    # Print model parameters
    print(f"\n📋 Model Parameters:")
    print("="*30)
    for param, value in params.items():
        print(f"   {param}: {value}")
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data()
    if X_train is None:
        print("❌ Failed to prepare data. Exiting.")
        return
    
    # Create plots
    print(f"\n🎨 Creating Visualizations...")
    
    # ROC Curve
    roc_auc = create_roc_curve(model, X_test, y_test)
    
    # Confusion Matrix
    accuracy, f1, precision, recall = create_confusion_matrix(model, X_test, y_test)
    
    # Classification Report
    print_classification_report(model, X_test, y_test)
    
    # Summary
    print(f"\n✅ Plot Creation Complete!")
    print("="*30)
    print(f"   ROC Curve: model_results/tuned_xgboost_roc_curve.png")
    print(f"   Confusion Matrix: model_results/tuned_xgboost_confusion_matrix.png")
    print(f"   Model Performance:")
    print(f"     - Accuracy:  {accuracy:.4f}")
    print(f"     - F1-Score:  {f1:.4f}")
    print(f"     - Precision: {precision:.4f}")
    print(f"     - Recall:    {recall:.4f}")
    print(f"     - AUC:       {roc_auc:.4f}")

if __name__ == "__main__":
    import pandas as pd
    main() 
