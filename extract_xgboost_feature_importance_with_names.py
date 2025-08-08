"""
XGBoost Feature Importance Extractor with Proper Feature Names
============================================================

This script loads the XGBoost model and extracts feature importance with proper feature names
by reconstructing the preprocessing pipeline to get accurate feature names.

Author: Diego
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
from pathlib import Path
import xgboost as xgb
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Import utilities from the project
from utils import build_dataset, extract_features, create_feature_matrix

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Spanish stop words (from utils.py)
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

class XGBoostFeatureImportanceWithNames:
    """
    Extract XGBoost feature importance with proper feature names
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the analyzer
        
        Args:
            model_path: Path to the XGBoost model file
        """
        self.model_path = model_path
        self.model = None
        self.pipeline = None
        self.feature_names = None
        self.feature_importance_df = None
        self.categorical_features = None
        self.numerical_features = None
        
        if not model_path:
            self.model_path = self._find_best_model()
        
        print(f"🔍 XGBoost Feature Importance Analyzer (with names) initialized")
        print(f"📁 Model path: {self.model_path}")
    
    def _find_best_model(self) -> str:
        """Find the best XGBoost model from available models"""
        model_candidates = [
            "models/tuned_xgboost_model.pkl",
            "models/champion_model_xgboost.pkl"
        ]
        
        for candidate in model_candidates:
            if Path(candidate).exists():
                print(f"✅ Found model: {candidate}")
                return candidate
        
        raise FileNotFoundError("No XGBoost models found.")
    
    def load_model_and_pipeline(self) -> bool:
        """
        Load the XGBoost model and extract the pipeline
        """
        try:
            print(f"📂 Loading model from: {self.model_path}")
            
            with open(self.model_path, 'rb') as f:
                loaded_object = pickle.load(f)
            
            # Handle pipeline object
            if hasattr(loaded_object, 'named_steps'):
                self.pipeline = loaded_object
                if 'classifier' in loaded_object.named_steps:
                    self.model = loaded_object.named_steps['classifier']
                    print("✅ Extracted XGBoost classifier from pipeline")
                else:
                    raise ValueError("No classifier found in pipeline")
            else:
                raise ValueError("Expected pipeline object")
            
            # Verify it's an XGBoost model
            if not hasattr(self.model, 'feature_importances_'):
                raise ValueError("Model doesn't have feature_importances_ attribute")
            
            print(f"✅ Model and pipeline loaded successfully!")
            print(f"📊 Model type: {type(self.model).__name__}")
            print(f"🔧 Pipeline steps: {list(self.pipeline.named_steps.keys())}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def reconstruct_feature_names(self) -> List[str]:
        """
        Reconstruct feature names by analyzing the preprocessing pipeline
        """
        print("🔧 Reconstructing feature names from preprocessing pipeline...")
        
        # Load sample data to understand the preprocessing
        print("Loading sample data...")
        X_train, _, _, _ = build_dataset()
        
        # Extract features
        print("Extracting features...")
        train_features = extract_features(X_train[:1000])  # Use sample for speed
        
        # Create feature matrix
        print("Creating feature matrix...")
        train_df, text_features, categorical_features, numerical_features = create_feature_matrix(train_features)
        
        # Add title column
        train_df['title'] = [item.get('title', '') for item in X_train[:1000]]
        
        # Store for later use
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        
        print(f"📊 Feature types:")
        print(f"  - Text features: {len(text_features)} (titles)")
        print(f"  - Categorical features: {len(categorical_features)}")
        print(f"  - Numerical features: {len(numerical_features)}")
        
        # Get the preprocessor from the pipeline
        if 'preprocessor' not in self.pipeline.named_steps:
            raise ValueError("No preprocessor found in pipeline")
        
        preprocessor = self.pipeline.named_steps['preprocessor']
        
        # Fit the preprocessor to get feature names
        print("Fitting preprocessor to extract feature names...")
        preprocessor.fit(train_df)
        
        # Extract feature names from each transformer
        feature_names = []
        
        # 1. Text features (TF-IDF)
        text_transformer = preprocessor.named_transformers_['text']
        if hasattr(text_transformer, 'named_steps') and 'tfidf' in text_transformer.named_steps:
            tfidf = text_transformer.named_steps['tfidf']
            if hasattr(tfidf, 'get_feature_names_out'):
                text_feature_names = tfidf.get_feature_names_out()
                feature_names.extend([f"text_{name}" for name in text_feature_names])
                print(f"  ✅ Extracted {len(text_feature_names)} text features")
            else:
                # Fallback for older sklearn versions
                if hasattr(tfidf, 'vocabulary_'):
                    vocab = tfidf.vocabulary_
                    text_feature_names = [word for word, _ in sorted(vocab.items(), key=lambda x: x[1])]
                    feature_names.extend([f"text_{name}" for name in text_feature_names])
                    print(f"  ✅ Extracted {len(text_feature_names)} text features (fallback method)")
        
        # 2. Categorical features (One-hot encoded)
        cat_transformer = preprocessor.named_transformers_['cat']
        if hasattr(cat_transformer, 'named_steps') and 'onehot' in cat_transformer.named_steps:
            onehot = cat_transformer.named_steps['onehot']
            if hasattr(onehot, 'get_feature_names_out'):
                cat_feature_names = onehot.get_feature_names_out(categorical_features)
                feature_names.extend(cat_feature_names)
                print(f"  ✅ Extracted {len(cat_feature_names)} categorical features")
            else:
                # Fallback: generate names based on categories
                cat_feature_names = []
                for i, feature in enumerate(categorical_features):
                    if hasattr(onehot, 'categories_') and i < len(onehot.categories_):
                        categories = onehot.categories_[i]
                        for category in categories:
                            cat_feature_names.append(f"{feature}_{category}")
                feature_names.extend(cat_feature_names)
                print(f"  ✅ Generated {len(cat_feature_names)} categorical feature names")
        
        # 3. Numerical features (scaled)
        feature_names.extend(numerical_features)
        print(f"  ✅ Added {len(numerical_features)} numerical features")
        
        print(f"🎉 Total feature names reconstructed: {len(feature_names)}")
        
        return feature_names
    
    def extract_feature_importance_with_names(self) -> pd.DataFrame:
        """
        Extract feature importance with proper feature names
        """
        if not self.model or not self.pipeline:
            raise ValueError("Model and pipeline not loaded. Call load_model_and_pipeline() first.")
        
        print("📊 Extracting feature importance with names...")
        
        # Get feature importance values
        importances = self.model.feature_importances_
        print(f"📊 Found {len(importances)} feature importances")
        
        # Reconstruct feature names
        feature_names = self.reconstruct_feature_names()
        
        # Handle length mismatch
        if len(feature_names) != len(importances):
            print(f"⚠️  Feature name count ({len(feature_names)}) doesn't match importance count ({len(importances)})")
            
            if len(feature_names) > len(importances):
                print("  Truncating feature names to match importances")
                feature_names = feature_names[:len(importances)]
            else:
                print("  Padding feature names with generic names")
                for i in range(len(importances) - len(feature_names)):
                    feature_names.append(f"feature_{len(feature_names) + i}")
        
        # Create DataFrame
        self.feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Add percentage and cumulative importance
        total_importance = self.feature_importance_df['importance'].sum()
        if total_importance > 0:
            self.feature_importance_df['importance_pct'] = (
                self.feature_importance_df['importance'] / total_importance * 100
            )
        else:
            self.feature_importance_df['importance_pct'] = 0
        
        self.feature_importance_df['cumulative_pct'] = self.feature_importance_df['importance_pct'].cumsum()
        
        print(f"✅ Feature importance extracted with names!")
        print(f"📊 Non-zero importance features: {(self.feature_importance_df['importance'] > 0).sum()}")
        
        return self.feature_importance_df
    
    def display_top_features(self, n_features: int = 20) -> pd.DataFrame:
        """
        Display top N most important features with proper names
        """
        if self.feature_importance_df is None:
            raise ValueError("Feature importance not extracted. Call extract_feature_importance_with_names() first.")
        
        top_features = self.feature_importance_df.head(n_features)
        
        print(f"\n📋 Top {n_features} Most Important Features (with names):")
        print("=" * 100)
        print(f"{'Rank':<4} {'Feature Name':<50} {'Importance':<12} {'%':<8} {'Cumulative %':<12}")
        print("-" * 100)
        
        for idx, (_, row) in enumerate(top_features.iterrows(), 1):
            feature_name = row['feature'][:47] + "..." if len(row['feature']) > 50 else row['feature']
            print(f"{idx:<4} {feature_name:<50} {row['importance']:<12.6f} {row['importance_pct']:<8.2f} {row['cumulative_pct']:<12.1f}")
        
        print("=" * 100)
        
        return top_features
    
    def analyze_feature_types(self) -> Dict:
        """
        Analyze feature importance by feature type
        """
        if self.feature_importance_df is None:
            raise ValueError("Feature importance not extracted.")
        
        # Categorize features by type
        text_features = self.feature_importance_df[self.feature_importance_df['feature'].str.startswith('text_')]
        categorical_features = self.feature_importance_df[
            ~self.feature_importance_df['feature'].str.startswith('text_') &
            ~self.feature_importance_df['feature'].isin(self.numerical_features)
        ]
        numerical_features = self.feature_importance_df[
            self.feature_importance_df['feature'].isin(self.numerical_features)
        ]
        
        analysis = {
            'text': {
                'count': len(text_features),
                'total_importance': text_features['importance_pct'].sum(),
                'top_feature': text_features.iloc[0] if len(text_features) > 0 else None
            },
            'categorical': {
                'count': len(categorical_features),
                'total_importance': categorical_features['importance_pct'].sum(),
                'top_feature': categorical_features.iloc[0] if len(categorical_features) > 0 else None
            },
            'numerical': {
                'count': len(numerical_features),
                'total_importance': numerical_features['importance_pct'].sum(),
                'top_feature': numerical_features.iloc[0] if len(numerical_features) > 0 else None
            }
        }
        
        print(f"\n📈 Feature Importance Analysis by Type:")
        print("=" * 60)
        
        for feature_type, stats in analysis.items():
            print(f"\n{feature_type.upper()} FEATURES:")
            print(f"  Count: {stats['count']:,}")
            print(f"  Total Importance: {stats['total_importance']:.2f}%")
            if stats['top_feature'] is not None:
                top = stats['top_feature']
                print(f"  Top Feature: {top['feature']} ({top['importance']:.6f})")
        
        print("=" * 60)
        
        return analysis
    
    def create_visualizations(self, n_top_features: int = 20, save_plots: bool = True):
        """
        Create comprehensive visualizations with feature names
        """
        if self.feature_importance_df is None:
            raise ValueError("Feature importance not extracted.")
        
        print(f"\n📊 Creating feature importance visualizations...")
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Top N features horizontal bar plot
        ax1 = plt.subplot(2, 2, 1)
        top_features = self.feature_importance_df.head(n_top_features)
        
        # Truncate long feature names for display
        display_names = []
        for name in top_features['feature']:
            if len(name) > 40:
                display_names.append(name[:37] + "...")
            else:
                display_names.append(name)
        
        y_pos = np.arange(len(top_features))
        bars = ax1.barh(y_pos, top_features['importance'], alpha=0.8)
        
        # Color bars by feature type
        colors = []
        for name in top_features['feature']:
            if name.startswith('text_'):
                colors.append('skyblue')
            elif name in self.numerical_features:
                colors.append('lightgreen')
            else:
                colors.append('salmon')
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(display_names, fontsize=9)
        ax1.set_xlabel('Feature Importance', fontsize=12)
        ax1.set_title(f'Top {n_top_features} Most Important Features', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        ax1.invert_yaxis()
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
            ax1.text(importance + max(top_features['importance']) * 0.01, i, 
                    f'{importance:.4f}', va='center', fontsize=8, fontweight='bold')
        
        # 2. Feature type distribution pie chart
        ax2 = plt.subplot(2, 2, 2)
        type_analysis = self.analyze_feature_types()
        
        type_names = []
        type_importances = []
        colors_pie = ['skyblue', 'salmon', 'lightgreen']
        
        for feature_type, stats in type_analysis.items():
            if stats['total_importance'] > 0:
                type_names.append(f"{feature_type.title()}\n({stats['count']} features)")
                type_importances.append(stats['total_importance'])
        
        if type_importances:
            wedges, texts, autotexts = ax2.pie(type_importances, labels=type_names, 
                                              autopct='%1.1f%%', colors=colors_pie[:len(type_importances)],
                                              startangle=90)
            ax2.set_title('Feature Importance by Type', fontsize=14, fontweight='bold')
        
        # 3. Cumulative importance plot
        ax3 = plt.subplot(2, 2, 3)
        cumulative_data = self.feature_importance_df.head(min(100, len(self.feature_importance_df)))
        ax3.plot(range(1, len(cumulative_data) + 1), cumulative_data['cumulative_pct'], 
                marker='o', linewidth=2, markersize=3)
        
        # Add threshold lines
        ax3.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% threshold')
        ax3.axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='90% threshold')
        
        ax3.set_xlabel('Number of Features', fontsize=12)
        ax3.set_ylabel('Cumulative Importance (%)', fontsize=12)
        ax3.set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Top text features word cloud style
        ax4 = plt.subplot(2, 2, 4)
        
        # Get top text features
        text_features = self.feature_importance_df[
            self.feature_importance_df['feature'].str.startswith('text_')
        ].head(20)
        
        if len(text_features) > 0:
            # Extract words from text features (remove 'text_' prefix)
            words = [feat.replace('text_', '') for feat in text_features['feature']]
            importances = text_features['importance'].values
            
            # Create a simple word importance plot
            y_pos = np.arange(len(words))
            bars = ax4.barh(y_pos, importances, color='skyblue', alpha=0.7)
            
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(words, fontsize=9)
            ax4.set_xlabel('Importance', fontsize=12)
            ax4.set_title('Top Text Features (Words/Phrases)', fontsize=14, fontweight='bold')
            ax4.grid(axis='x', alpha=0.3)
            ax4.invert_yaxis()
        else:
            ax4.text(0.5, 0.5, 'No text features found', transform=ax4.transAxes, 
                    ha='center', va='center', fontsize=12)
            ax4.set_title('Top Text Features', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_plots:
            output_dir = Path("model_results")
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / "xgboost_feature_importance_with_names.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"📊 Visualizations saved to: {output_file}")
        
        plt.show()
    
    def save_results(self, output_dir: str = "model_results"):
        """
        Save feature importance results to files
        """
        if self.feature_importance_df is None:
            raise ValueError("Feature importance not extracted.")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save CSV
        csv_file = output_path / "xgboost_feature_importance_with_names.csv"
        self.feature_importance_df.to_csv(csv_file, index=False)
        print(f"📄 Feature importance CSV saved to: {csv_file}")
        
        # Save detailed report
        report = self.generate_detailed_report()
        report_file = output_path / "xgboost_feature_importance_report_with_names.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"📄 Detailed report saved to: {report_file}")
        
        return csv_file, report_file
    
    def generate_detailed_report(self, n_top_features: int = 30) -> str:
        """
        Generate a comprehensive report with feature names
        """
        if self.feature_importance_df is None:
            raise ValueError("Feature importance not extracted.")
        
        # Analyze by types
        type_analysis = self.analyze_feature_types()
        
        report = []
        report.append("XGBoost Feature Importance Analysis Report (with Feature Names)")
        report.append("=" * 70)
        report.append(f"Model: {self.model_path}")
        report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        total_features = len(self.feature_importance_df)
        non_zero_features = (self.feature_importance_df['importance'] > 0).sum()
        
        report.append("SUMMARY")
        report.append("-" * 20)
        report.append(f"Total Features: {total_features:,}")
        report.append(f"Non-zero Importance: {non_zero_features:,} ({non_zero_features/total_features*100:.1f}%)")
        report.append("")
        
        # Feature type analysis
        report.append("FEATURE TYPE ANALYSIS")
        report.append("-" * 30)
        for feature_type, stats in type_analysis.items():
            report.append(f"{feature_type.upper()}:")
            report.append(f"  Features: {stats['count']:,}")
            report.append(f"  Total Importance: {stats['total_importance']:.2f}%")
            if stats['top_feature'] is not None:
                top = stats['top_feature']
                report.append(f"  Top Feature: {top['feature']} ({top['importance']:.6f})")
            report.append("")
        
        # Top features
        report.append(f"TOP {n_top_features} FEATURES")
        report.append("-" * 50)
        report.append(f"{'Rank':<4} {'Feature Name':<45} {'Importance':<12} {'%':<8}")
        report.append("-" * 75)
        
        for idx, (_, row) in enumerate(self.feature_importance_df.head(n_top_features).iterrows(), 1):
            feature_name = row['feature'][:42] + "..." if len(row['feature']) > 45 else row['feature']
            report.append(f"{idx:<4} {feature_name:<45} {row['importance']:<12.6f} {row['importance_pct']:<8.2f}")
        
        report.append("")
        
        # Key insights
        report.append("KEY INSIGHTS")
        report.append("-" * 20)
        
        # Most important feature type
        max_type = max(type_analysis.items(), key=lambda x: x[1]['total_importance'])
        report.append(f"• Most important feature type: {max_type[0].title()} ({max_type[1]['total_importance']:.1f}%)")
        
        # Feature concentration
        top_10_pct = self.feature_importance_df.head(10)['importance_pct'].sum()
        top_20_pct = self.feature_importance_df.head(20)['importance_pct'].sum()
        
        report.append(f"• Top 10 features explain {top_10_pct:.1f}% of total importance")
        report.append(f"• Top 20 features explain {top_20_pct:.1f}% of total importance")
        
        # Text feature insights
        text_features = self.feature_importance_df[
            self.feature_importance_df['feature'].str.startswith('text_')
        ]
        if len(text_features) > 0:
            report.append(f"• Text features: {len(text_features)} features contribute {text_features['importance_pct'].sum():.1f}% importance")
            top_text = text_features.iloc[0]
            report.append(f"• Most important text feature: '{top_text['feature'].replace('text_', '')}' ({top_text['importance']:.6f})")
        
        report.append("")
        report.append("=" * 70)
        report.append("End of Report")
        
        return "\n".join(report)
    
    def run_complete_analysis(self, n_top_features: int = 20, save_outputs: bool = True):
        """
        Run the complete analysis with feature names
        """
        print("🚀 Starting Complete XGBoost Feature Importance Analysis (with names)")
        print("=" * 70)
        
        # Load model and pipeline
        if not self.load_model_and_pipeline():
            raise RuntimeError("Failed to load model and pipeline")
        
        # Extract feature importance with names
        feature_df = self.extract_feature_importance_with_names()
        
        # Display top features
        top_features = self.display_top_features(n_top_features)
        
        # Analyze by feature types
        type_analysis = self.analyze_feature_types()
        
        # Create visualizations
        self.create_visualizations(n_top_features, save_plots=save_outputs)
        
        # Save results
        if save_outputs:
            csv_file, report_file = self.save_results()
        
        # Generate and display report
        report = self.generate_detailed_report(n_top_features)
        print("\n" + report)
        
        print("\n🎉 Complete Feature Importance Analysis (with names) Finished!")
        print("=" * 70)
        
        return {
            'feature_importance_df': feature_df,
            'top_features': top_features,
            'type_analysis': type_analysis,
            'report': report
        }


def main():
    """
    Main function to run the feature importance analysis with names
    """
    try:
        # Initialize analyzer
        analyzer = XGBoostFeatureImportanceWithNames()
        
        # Run complete analysis
        results = analyzer.run_complete_analysis(
            n_top_features=20,
            save_outputs=True
        )
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📊 Analyzed {len(results['feature_importance_df'])} features")
        
        # Display key results
        top_feature = results['top_features'].iloc[0]
        print(f"🏆 Most important feature: {top_feature['feature']}")
        print(f"📈 Importance value: {top_feature['importance']:.6f} ({top_feature['importance_pct']:.2f}%)")
        
        # Show feature type summary
        type_analysis = results['type_analysis']
        print(f"\n📊 Feature Type Summary:")
        for feature_type, stats in type_analysis.items():
            print(f"  {feature_type.title()}: {stats['count']} features, {stats['total_importance']:.1f}% importance")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
