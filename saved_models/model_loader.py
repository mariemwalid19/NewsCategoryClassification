
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report

class NewsClassifierLoader:
    """Utility class to load and use trained news classification models"""
    
    def __init__(self, models_dir='saved_models'):
        self.models_dir = models_dir
        self.models = {}
        self.tfidf_vectorizer = None
        self.class_mapping = None
        self.results_data = None
        
    def load_components(self):
        """Load all saved components"""
        
        # Load TF-IDF vectorizer
        with open(f'{self.models_dir}/tfidf_vectorizer.pkl', 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        
        # Load results metadata
        with open(f'{self.models_dir}/model_results.pkl', 'rb') as f:
            self.results_data = pickle.load(f)
            self.class_mapping = self.results_data['class_mapping']
        
        # Load scikit-learn models
        sklearn_models = ['logistic_regression', 'random_forest', 'multinomial_naive_bayes']
        for model_name in sklearn_models:
            try:
                with open(f'{self.models_dir}/{model_name}_model.pkl', 'rb') as f:
                    self.models[model_name] = pickle.load(f)
            except FileNotFoundError:
                print(f"Model {model_name} not found")
        
        # Load Neural Network model if exists
        try:
            self.models['neural_network'] = load_model(f'{self.models_dir}/neural_network_model.h5')
            # Load NN preprocessing components
            with open(f'{self.models_dir}/neural_network_scaler.pkl', 'rb') as f:
                self.nn_scaler = pickle.load(f)
            with open(f'{self.models_dir}/neural_network_tfidf.pkl', 'rb') as f:
                self.nn_tfidf = pickle.load(f)
        except:
            print("Neural Network model not found")
    
    def predict_single(self, text, model_name='best'):
        """Predict single text sample"""
        if model_name == 'best':
            model_name = self.results_data['best_model'].replace(' ', '_').lower()
        
        if model_name == 'neural_network':
            # Special handling for neural network
            processed_text = self.preprocess_for_nn([text])
            prediction_proba = self.models[model_name].predict(processed_text, verbose=0)
            prediction = np.argmax(prediction_proba, axis=1)[0]
            confidence = np.max(prediction_proba)
        else:
            # Standard scikit-learn models
            text_vectorized = self.tfidf_vectorizer.transform([text])
            prediction = self.models[model_name].predict(text_vectorized)[0]
            confidence = np.max(self.models[model_name].predict_proba(text_vectorized))
        
        predicted_class = self.class_mapping[prediction + 1]
        
        return {
            'predicted_class': predicted_class,
            'class_id': prediction,
            'confidence': confidence
        }
    
    def preprocess_for_nn(self, texts):
        """Preprocess texts for neural network"""
        X = self.nn_tfidf.transform(texts).toarray()
        X_scaled = self.nn_scaler.transform(X)
        return X_scaled
    
    def get_model_info(self):
        """Get information about loaded models"""
        return self.results_data

# Example usage:
# loader = NewsClassifierLoader()
# loader.load_components()
# result = loader.predict_single("Apple releases new iPhone with advanced AI features")
# print(f"Predicted: {result['predicted_class']} (confidence: {result['confidence']:.3f})")
