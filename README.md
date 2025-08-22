# News Category Classification Project

A comprehensive text classification system that automatically categorizes news articles into four categories: World, Sports, Business, and Sci/Tech using multiple machine learning approaches.

## Project Overview

This project implements a complete news classification pipeline using the AG News dataset from Kaggle. The system processes raw news text through advanced NLP techniques and trains multiple machine learning models to achieve high-accuracy classification.

### Dataset
- **Source**: AG News Dataset (Kaggle)
- **Training samples**: 120,000 news articles
- **Test samples**: 7,600 news articles
- **Categories**: 4 classes
  - 1: World News
  - 2: Sports
  - 3: Business
  - 4: Science & Technology

## Features Implemented

### Core Requirements
- ✅ **Text Preprocessing**: Tokenization, stopword removal, lemmatization
- ✅ **Feature Engineering**: TF-IDF vectorization with n-grams
- ✅ **Multiple Classifiers**: Logistic Regression, Random Forest, Multinomial Naive Bayes
- ✅ **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, confusion matrices
- ✅ **Visualizations**: Word clouds, feature importance, performance comparisons

### Bonus Features
- ✅ **Neural Network**: Deep learning implementation with TensorFlow/Keras
- ✅ **Advanced Visualizations**: Training history, confidence analysis
- ✅ **Memory Optimization**: Efficient handling of large sparse matrices
- ✅ **Model Persistence**: Complete saving and loading system

## Technology Stack

### Libraries Used
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Deep Learning**: TensorFlow/Keras
- **NLP**: NLTK, WordCloud
- **Visualization**: matplotlib, seaborn
- **Model Persistence**: pickle, joblib

### Models Implemented
1. **Logistic Regression** - Linear classifier with regularization
2. **Random Forest** - Ensemble method with decision trees
3. **Multinomial Naive Bayes** - Probabilistic text classifier
4. **Neural Network** - 3-layer feedforward network with dropout

## Project Structure

```
news-classification/
│
├── Dataset/
│   ├── train.csv              # Training data (120K samples)
│   └── test.csv               # Test data (7.6K samples)
│
├── saved_models/
│   ├── tfidf_vectorizer.pkl   # TF-IDF vectorizer
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── multinomial_naive_bayes_model.pkl
│   ├── neural_network_model.h5
│   ├── neural_network_scaler.pkl
│   ├── neural_network_tfidf.pkl
│   ├── model_results.pkl      # Performance metadata
│   └── model_loader.py        # Loading utility
│
├── main_notebook.ipynb        # Complete analysis notebook
├── predict_news.py           # Interactive prediction script
├── requirements.txt          # Dependencies
└── README.md                # This file
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- 4GB+ RAM (for neural network training)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd news-classification
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (run in Python)
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## Usage

### Training Models
Run the complete analysis notebook:
```bash
jupyter notebook main_notebook.ipynb
```

### Making Predictions
Use the interactive prediction script:
```bash
python predict_news.py
```

### Programmatic Usage
```python
from saved_models.model_loader import NewsClassifierLoader

# Load trained models
loader = NewsClassifierLoader()
loader.load_components()

# Make prediction
text = "Apple releases new iPhone with advanced AI features"
result = loader.predict_single(text)
print(f"Category: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.3f}")
```

## Model Performance

| Model | Test Accuracy | Training Time | Key Strengths |
|-------|---------------|---------------|---------------|
| Logistic Regression | ~94% | ~15s | Fast, interpretable, strong baseline |
| Random Forest | ~93% | ~45s | Feature importance, robust |
| Multinomial Naive Bayes | ~92% | ~2s | Extremely fast, good for text |
| Neural Network | ~93% | ~120s | Deep learning, handles complexity |

*Best Model: Logistic Regression with 94% accuracy*

## Key Technical Decisions

### Text Preprocessing
- **Combined text features**: Title weighted 2x, Description 1x
- **Advanced cleaning**: URL removal, news agency filtering
- **Smart tokenization**: Lemmatization over stemming
- **Custom stopwords**: Added domain-specific terms

### Feature Engineering
- **TF-IDF Configuration**: 
  - Max features: 10,000
  - N-grams: (1,2)
  - Min/Max document frequency: 5 / 80%
  - Sublinear TF scaling

### Model Optimization
- **Memory efficiency**: Sparse matrix handling
- **Neural network**: Reduced dimensions for memory constraints
- **Cross-validation**: Stratified splits maintaining class balance
- **Early stopping**: Prevent overfitting in neural network

## Insights & Analysis

### Top Performing Features by Category

**Sports**: game, team, season, player, match, win, league
**Business**: company, market, stock, profit, sales, revenue
**World**: government, country, president, war, election, policy  
**Sci/Tech**: technology, software, computer, internet, system, data

### Classification Challenges
- **Business vs World**: Overlap in economic/political news
- **Sci/Tech vs Business**: Technology company coverage
- **High confidence**: 85% of predictions above 80% confidence

## Future Improvements

### Potential Enhancements
- **Advanced NLP**: BERT/GPT embeddings
- **Ensemble methods**: Model stacking/voting
- **Real-time pipeline**: Streaming classification
- **Multi-language support**: International news sources
- **Category expansion**: Sub-categories within main classes

### Deployment Considerations
- **API development**: REST endpoints for predictions
- **Containerization**: Docker deployment
- **Monitoring**: Performance tracking in production
- **A/B testing**: Model comparison in live environment

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- **AG News Dataset**: Provided by Kaggle
- **Research inspiration**: Text classification best practices
- **Libraries**: Scikit-learn, TensorFlow, NLTK communities

## Contact

For questions or suggestions, please open an issue on GitHub.

---

*Project completed as part of machine learning text classification study.*
