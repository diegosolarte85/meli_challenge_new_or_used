# Advanced Modeling Usage Guide

## Overview

The `advanced_modeling.py` script provides comprehensive hyperparameter tuning and deep learning models with progress bars and multiple configuration options.

## Quick Start

### 1. Simple Hyperparameter Tuning (Recommended)
```bash
python advanced_modeling.py
```
- Fast execution (~5-10 minutes)
- 729 parameter combinations
- Grid Search only
- No deep learning

### 2. Comprehensive Tuning with Deep Learning
```bash
python advanced_modeling.py --tuning-level comprehensive
```
- Medium execution (~15-30 minutes)
- 3,888 parameter combinations
- Grid Search + Randomized Search
- Deep learning models included

### 3. Full Tuning with Deep Learning
```bash
python advanced_modeling.py --tuning-level full
```
- Long execution (~30-60 minutes)
- 15,552 parameter combinations
- Grid Search + Randomized Search
- Deep learning models included

### 4. XGBoost Only (No Deep Learning)
```bash
python advanced_modeling.py --no-deep-learning
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--tuning-level` | `simple`, `comprehensive`, or `full` | `simple` |
| `--no-deep-learning` | Disable deep learning models | `False` |

## Tuning Levels

### Simple Level
- **Parameter Combinations**: 729
- **Search Methods**: Grid Search only
- **Expected Time**: 5-10 minutes
- **Best for**: Quick testing and development

### Comprehensive Level
- **Parameter Combinations**: 3,888 (Grid) + 50 (Random)
- **Search Methods**: Grid Search + Randomized Search
- **Expected Time**: 15-30 minutes
- **Best for**: Production optimization

### Full Level
- **Parameter Combinations**: 15,552 (Grid) + 50 (Random)
- **Search Methods**: Grid Search + Randomized Search
- **Expected Time**: 30-60 minutes
- **Best for**: Maximum optimization

## Expected Results

### Hyperparameter Tuning
- **Target Improvement**: 0.5-2% over original 87.35%
- **Expected CV Score**: 87.5-89.0%
- **Test Accuracy**: 87.5-89.0%

### Deep Learning Models
- **LSTM**: 85-87% accuracy
- **BiLSTM**: 86-88% accuracy
- **CNN+LSTM**: 87-89% accuracy

## Output Files

After running the script, you'll get:

### XGBoost Files
- `tuned_xgboost_model.pkl` - Best tuned model
- `tuned_xgboost_params.pkl` - Best parameters

### Deep Learning Files (if enabled)
- `best_deep_learning_model_lstm.h5` - Best DL model
- `text_tokenizer.pkl` - Text tokenizer
- `label_encoder.pkl` - Label encoder

### Visualization Files
- `model_results/deep_learning_training_history.png` - Training plots

## Progress Tracking

The script includes progress bars for:
- ✅ Grid Search progress
- ✅ Randomized Search progress
- ✅ Text data cleaning
- ✅ Deep learning training epochs

## Troubleshooting

### TensorFlow Not Available
```bash
# Install TensorFlow
pip install tensorflow

# Or run without deep learning
python advanced_modeling.py --no-deep-learning
```

### Memory Issues
- Use `--tuning-level simple` for faster execution
- Reduce batch size in deep learning (modify code)
- Use fewer CV folds (modify code)

### Performance Optimization
- Use GPU if available for deep learning
- Increase `n_jobs` parameter for parallel processing
- Reduce parameter grid size for faster execution

## Examples

### Quick Test
```bash
python advanced_modeling.py
```

### Production Optimization
```bash
python advanced_modeling.py --tuning-level comprehensive
```

### Maximum Optimization
```bash
python advanced_modeling.py --tuning-level full
```

### XGBoost Only
```bash
python advanced_modeling.py --no-deep-learning
```

## Monitoring Progress

The script provides real-time progress tracking:
- 🔍 Grid Search progress
- 🎲 Randomized Search progress
- 🧠 Deep learning training epochs
- 📊 Performance metrics

## Results Interpretation

### Best Model Selection
- **XGBoost vs Deep Learning**: Compare F1-scores
- **Grid vs Randomized**: Compare CV scores
- **Overall Best**: Highest F1-score across all models

### Performance Metrics
- **Accuracy**: Primary metric (target ≥86%)
- **F1-Score**: Secondary metric (balanced performance)
- **CV Score**: Cross-validation performance
- **Test Score**: Final evaluation performance

## Next Steps

1. **Run the script** with your preferred configuration
2. **Monitor progress** using the progress bars
3. **Analyze results** in the final summary
4. **Load saved models** for predictions
5. **Compare with original** champion model

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify TensorFlow installation
3. Try simpler tuning levels first
4. Monitor system resources during execution 