"""
Extract Feature Importance from Champion Model with Proper Names
===============================================================

This script specifically handles the champion model pipeline structure
and extracts feature importance with proper feature names.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

def load_champion_model():
    """Load the champion model pipeline"""
    model_path = "models/champion_model_xgboost.pkl"
    
    print(f"Loading champion model from: {model_path}")
    
    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)
    
    print(f"Pipeline type: {type(pipeline).__name__}")
    print(f"Pipeline steps: {list(pipeline.named_steps.keys())}")
    
    # Extract the classifier
    classifier = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']
    
    print(f"Classifier: {type(classifier).__name__}")
    print(f"Preprocessor: {type(preprocessor).__name__}")
    print(f"Features: {len(classifier.feature_importances_)}")
    
    return pipeline, classifier, preprocessor

def extract_feature_names_from_pipeline(pipeline, classifier):
    """Extract proper feature names from the fitted pipeline"""
    
    preprocessor = pipeline.named_steps['preprocessor']
    
    try:
        # Get feature names from the fitted preprocessor
        feature_names = preprocessor.get_feature_names_out()
        print(f"✅ Successfully extracted {len(feature_names)} feature names from preprocessor")
        
        # Verify the length matches
        n_features = len(classifier.feature_importances_)
        if len(feature_names) != n_features:
            print(f"⚠️  Length mismatch: {len(feature_names)} names vs {n_features} importances")
            if len(feature_names) > n_features:
                feature_names = feature_names[:n_features]
                print(f"✂️  Truncated to {len(feature_names)} feature names")
        
        return list(feature_names)
        
    except Exception as e:
        print(f"❌ Error extracting feature names: {e}")
        
        # Fallback: generate generic names
        n_features = len(classifier.feature_importances_)
        feature_names = [f"feature_{i:04d}" for i in range(n_features)]
        print(f"🔄 Generated {n_features} generic feature names as fallback")
        return feature_names

def analyze_feature_importance(classifier, feature_names):
    """Analyze feature importance with proper names"""
    
    importances = classifier.feature_importances_
    
    # Create DataFrame
    feature_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Add percentages
    total_importance = feature_df['importance'].sum()
    feature_df['importance_pct'] = (feature_df['importance'] / total_importance * 100)
    feature_df['cumulative_pct'] = feature_df['importance_pct'].cumsum()
    
    return feature_df

def categorize_features(feature_df):
    """Categorize features by type"""
    
    categories = {
        'Text (TF-IDF)': feature_df[feature_df['feature'].str.contains('text__', case=False, na=False)],
        'Categorical (One-Hot)': feature_df[feature_df['feature'].str.contains('cat__', case=False, na=False)],
        'Numerical (Scaled)': feature_df[feature_df['feature'].str.contains('num__', case=False, na=False)],
        'Price Features': feature_df[feature_df['feature'].str.contains('price', case=False, na=False)],
        'Quantity Features': feature_df[feature_df['feature'].str.contains('quantity', case=False, na=False)],
        'Boolean Features': feature_df[feature_df['feature'].str.contains('has_|is_', case=False, na=False)],
        'Other': feature_df[~feature_df['feature'].str.contains('text__|cat__|num__|price|quantity|has_|is_', case=False, na=False)]
    }
    
    # Remove empty categories
    categories = {k: v for k, v in categories.items() if len(v) > 0}
    
    return categories

def display_results(feature_df, categories, n_top=20):
    """Display comprehensive results"""
    
    print(f"\n" + "="*100)
    print("CHAMPION MODEL FEATURE IMPORTANCE ANALYSIS")
    print(f"="*100)
    
    total_features = len(feature_df)
    non_zero_features = (feature_df['importance'] > 0).sum()
    
    print(f"Total Features: {total_features:,}")
    print(f"Non-zero Importance: {non_zero_features:,} ({non_zero_features/total_features*100:.1f}%)")
    
    # Feature categories
    print(f"\nFEATURE CATEGORIES:")
    print("-" * 50)
    for category, features in categories.items():
        if len(features) > 0:
            total_imp = features['importance_pct'].sum()
            print(f"{category:<20}: {len(features):>3} features ({total_imp:>5.1f}% importance)")
            if len(features) > 0:
                top_feature = features.iloc[0]
                print(f"  └─ Top: {top_feature['feature'][:60]} ({top_feature['importance']:.6f})")
    
    # Top features
    print(f"\nTOP {n_top} MOST IMPORTANT FEATURES:")
    print("-" * 100)
    print(f"{'Rank':<4} {'Feature Name':<65} {'Importance':<12} {'%':<8} {'Cumulative':<10}")
    print("-" * 100)
    
    for idx, (_, row) in enumerate(feature_df.head(n_top).iterrows(), 1):
        feature_name = row['feature'][:62] + "..." if len(row['feature']) > 65 else row['feature']
        print(f"{idx:<4} {feature_name:<65} {row['importance']:<12.6f} {row['importance_pct']:<8.2f} {row['cumulative_pct']:<10.1f}")
    
    print("-" * 100)
    
    # Key insights
    print(f"\nKEY INSIGHTS:")
    top_10_pct = feature_df.head(10)['importance_pct'].sum()
    top_20_pct = feature_df.head(20)['importance_pct'].sum()
    
    print(f"• Top 10 features explain {top_10_pct:.1f}% of total importance")
    print(f"• Top 20 features explain {top_20_pct:.1f}% of total importance")
    print(f"• Most important feature: {feature_df.iloc[0]['feature']}")
    print(f"• Top feature contributes {feature_df.iloc[0]['importance_pct']:.1f}% of total importance")
    
    # Text feature analysis
    text_features = categories.get('Text (TF-IDF)', pd.DataFrame())
    if len(text_features) > 0:
        print(f"• Text features: {len(text_features)} features, {text_features['importance_pct'].sum():.1f}% total importance")
        print(f"• Top text feature: {text_features.iloc[0]['feature'].replace('text__tfidf__', '')}")

def create_visualizations(feature_df, categories):
    """Create visualizations with formal, professional colors"""
    
    # Set professional color scheme
    plt.style.use('default')
    
    # Define diverse professional color palette
    FORMAL_COLORS = {
        'text': '#3498DB',        # Bright blue for text features
        'categorical': '#E74C3C', # Red for categorical features
        'numerical': '#27AE60',   # Green for numerical features
        'price': '#F39C12',       # Orange for price features
        'quantity': '#9B59B6',    # Purple for quantity features
        'boolean': '#1ABC9C',     # Teal for boolean features
        'other': '#95A5A6',       # Gray for other features
        'primary': '#2C3E50',     # Dark blue-gray for primary elements
        'secondary': '#34495E',   # Medium blue-gray for secondary elements
        'accent': '#7F8C8D'       # Light gray for accents
    }
    
    # Set up the plot with white background
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.patch.set_facecolor('white')
    
    # 1. Top 15 features bar plot
    top_features = feature_df.head(15)
    
    # Shorten feature names for display and assign diverse colors
    display_names = []
    colors = []
    for name in top_features['feature']:
        if 'text__' in name:
            display_name = name.replace('text__', '')[:50]
            colors.append(FORMAL_COLORS['text'])
        elif 'cat__' in name:
            display_name = name.replace('cat__', '').replace('onehot__', '')[:50]
            colors.append(FORMAL_COLORS['categorical'])
        elif 'num__' in name:
            display_name = name.replace('num__', '').replace('scaler__', '')[:50]
            # Assign specific colors based on feature content
            if 'price' in name.lower():
                colors.append(FORMAL_COLORS['price'])
            elif 'quantity' in name.lower():
                colors.append(FORMAL_COLORS['quantity'])
            elif any(x in name.lower() for x in ['has_', 'is_', 'free_', 'automatic_']):
                colors.append(FORMAL_COLORS['boolean'])
            else:
                colors.append(FORMAL_COLORS['numerical'])
        else:
            display_name = name[:50]
            colors.append(FORMAL_COLORS['other'])
        
        if len(display_name) > 50:
            display_name = display_name[:47] + "..."
        display_names.append(display_name)
    
    y_pos = np.arange(len(top_features))
    bars = ax1.barh(y_pos, top_features['importance'], color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(display_names, fontsize=10, color='#333333')
    ax1.set_xlabel('Feature Importance', fontsize=12, color='#333333', fontweight='bold')
    ax1.set_title('Top 15 Most Important Features', fontsize=13, color='#2E5984', fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.2, color='#CCCCCC')
    ax1.set_facecolor('#FAFAFA')
    ax1.invert_yaxis()
    
    # Add value labels
    for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
        ax1.text(importance + max(top_features['importance']) * 0.01, i, 
                 f'{importance:.4f}', va='center', fontsize=8, color='#333333', fontweight='bold')
    
    # Add legend for feature types
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor=FORMAL_COLORS['text'], label='Text Features'),
        plt.Rectangle((0,0),1,1, facecolor=FORMAL_COLORS['categorical'], label='Categorical'),
        plt.Rectangle((0,0),1,1, facecolor=FORMAL_COLORS['numerical'], label='Numerical'),
        plt.Rectangle((0,0),1,1, facecolor=FORMAL_COLORS['quantity'], label='Quantity'),
        plt.Rectangle((0,0),1,1, facecolor=FORMAL_COLORS['boolean'], label='Boolean'),
        plt.Rectangle((0,0),1,1, facecolor=FORMAL_COLORS['price'], label='Price')
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    # 2. Feature category pie chart with formal colors
    if len(categories) > 1:
        category_names = []
        category_importances = []
        formal_pie_colors = [
            FORMAL_COLORS['text'],        # Text features - Blue
            FORMAL_COLORS['categorical'], # Categorical - Red
            FORMAL_COLORS['numerical'],   # Numerical - Green
            FORMAL_COLORS['price'],       # Price - Orange
            FORMAL_COLORS['quantity'],    # Quantity - Purple
            FORMAL_COLORS['boolean'],     # Boolean - Teal
            FORMAL_COLORS['other']        # Other - Gray
        ]
        
        for i, (category, features) in enumerate(categories.items()):
            if features['importance_pct'].sum() > 1:  # Only show categories with >1% importance
                category_names.append(f"{category}\n({len(features)} features)")
                category_importances.append(features['importance_pct'].sum())
        
        if category_importances:
            wedges, texts, autotexts = ax2.pie(category_importances, labels=category_names, autopct='%1.1f%%',
                   colors=formal_pie_colors[:len(category_importances)], startangle=90,
                   textprops={'fontsize': 10, 'color': '#333333'})
            
            # Style the percentage text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(9)
            
            ax2.set_title('Feature Importance by Category', fontsize=13, color='#2E5984', fontweight='bold', pad=15)
    
    # 3. Cumulative importance with formal styling
    n_plot = min(50, len(feature_df))
    cumulative_data = feature_df.head(n_plot)
    
    ax3.plot(range(1, len(cumulative_data) + 1), cumulative_data['cumulative_pct'], 
             marker='o', linewidth=3, markersize=4, color=FORMAL_COLORS['quantity'], alpha=0.9)
    ax3.axhline(y=80, color=FORMAL_COLORS['price'], linestyle='--', alpha=0.8, linewidth=2, label='80% threshold')
    ax3.axhline(y=90, color=FORMAL_COLORS['categorical'], linestyle='--', alpha=0.8, linewidth=2, label='90% threshold')
    ax3.set_xlabel('Number of Features', fontsize=12, color='#333333', fontweight='bold')
    ax3.set_ylabel('Cumulative Importance (%)', fontsize=12, color='#333333', fontweight='bold')
    ax3.set_title('Cumulative Feature Importance', fontsize=13, color='#2E5984', fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.2, color='#CCCCCC')
    ax3.set_facecolor('#FAFAFA')
    ax3.legend(fontsize=10)
    
    # 4. Text features with formal styling
    text_features = categories.get('Text (TF-IDF)', pd.DataFrame())
    if len(text_features) > 0:
        top_text = text_features.head(15)
        words = [name.replace('text__', '') for name in top_text['feature']]
        
        y_pos = np.arange(len(words))
        
        # Create a gradient of colors for text features
        text_colors = [
            '#3498DB', '#5DADE2', '#85C1E9', '#AED6F1', '#D6EAF8',
            '#1ABC9C', '#58D68D', '#82E0AA', '#A9DFBF', '#D1F2EB',
            '#E74C3C', '#EC7063', '#F1948A', '#F5B7B1', '#FADBD8'
        ]
        
        # Assign colors cyclically
        bar_colors = [text_colors[i % len(text_colors)] for i in range(len(words))]
        
        bars = ax4.barh(y_pos, top_text['importance'], color=bar_colors, alpha=0.8, 
                       edgecolor='white', linewidth=0.5)
        
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(words, fontsize=9, color='#333333')
        ax4.set_xlabel('Importance', fontsize=12, color='#333333', fontweight='bold')
        ax4.set_title('Top Text Features (Words/Phrases)', fontsize=13, color='#2E5984', fontweight='bold', pad=15)
        ax4.grid(axis='x', alpha=0.2, color='#CCCCCC')
        ax4.set_facecolor('#FAFAFA')
        ax4.invert_yaxis()
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, top_text['importance'])):
            ax4.text(importance + max(top_text['importance']) * 0.02, i, 
                     f'{importance:.4f}', va='center', fontsize=8, color='#333333', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No text features found', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=12, color='#666666')
        ax4.set_title('Top Text Features', fontsize=13, color='#2E5984', fontweight='bold', pad=15)
        ax4.set_facecolor('#FAFAFA')
    
    # Add overall title with proper spacing
    fig.suptitle('XGBoost Champion Model: Feature Importance Analysis', 
                 fontsize=16, color='#2E5984', fontweight='bold', y=0.96)
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.25)  # More space for titles
    
    # Save plot with high quality
    output_dir = Path("model_results")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "champion_model_feature_importance.png", 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"📊 Professional visualization saved: {output_dir / 'champion_model_feature_importance.png'}")
    
    plt.show()

def save_results(feature_df, categories):
    """Save results to files"""
    
    output_dir = Path("model_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed CSV
    csv_file = output_dir / "champion_model_feature_importance.csv"
    feature_df.to_csv(csv_file, index=False)
    
    # Save report
    report_file = output_dir / "champion_model_feature_importance_report.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("Champion Model Feature Importance Analysis\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")
        
        f.write("FEATURE CATEGORIES:\n")
        f.write("-" * 30 + "\n")
        for category, features in categories.items():
            if len(features) > 0:
                f.write(f"{category}: {len(features)} features ({features['importance_pct'].sum():.1f}%)\n")
                if len(features) > 0:
                    top = features.iloc[0]
                    f.write(f"  Top: {top['feature']} ({top['importance']:.6f})\n")
                f.write("\n")
        
        f.write("TOP 30 FEATURES:\n")
        f.write("-" * 50 + "\n")
        for idx, (_, row) in enumerate(feature_df.head(30).iterrows(), 1):
            f.write(f"{idx:2d}. {row['feature']:<60} {row['importance']:>10.6f} ({row['importance_pct']:>5.2f}%)\n")
    
    print(f"📁 Results saved:")
    print(f"  - CSV: {csv_file}")
    print(f"  - Report: {report_file}")

def main():
    """Main function"""
    
    try:
        print("🚀 Champion Model Feature Importance Analysis")
        print("=" * 60)
        
        # Load model
        pipeline, classifier, preprocessor = load_champion_model()
        
        # Extract feature names
        print("\n📋 Extracting feature names...")
        feature_names = extract_feature_names_from_pipeline(pipeline, classifier)
        
        # Analyze importance
        print("\n📊 Analyzing feature importance...")
        feature_df = analyze_feature_importance(classifier, feature_names)
        
        # Categorize features
        print("\n🏷️  Categorizing features...")
        categories = categorize_features(feature_df)
        
        # Display results
        display_results(feature_df, categories, n_top=20)
        
        # Create visualizations
        print("\n📈 Creating visualizations...")
        create_visualizations(feature_df, categories)
        
        # Save results
        print("\n💾 Saving results...")
        save_results(feature_df, categories)
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"🏆 Most important feature: {feature_df.iloc[0]['feature']}")
        print(f"📈 Importance: {feature_df.iloc[0]['importance']:.6f} ({feature_df.iloc[0]['importance_pct']:.2f}%)")
        
        return feature_df, categories
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    feature_df, categories = main()
