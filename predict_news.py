
#!/usr/bin/env python3
"""
Simple script to make predictions using trained news classification models
"""

import sys
import os
sys.path.append('saved_models')

from model_loader import NewsClassifierLoader

def main():
    # Load the trained models
    print("Loading trained models...")
    loader = NewsClassifierLoader()
    loader.load_components()
    
    # Get model info
    info = loader.get_model_info()
    print(f"Best model: {info['best_model']} (accuracy: {info['best_accuracy']:.3f})")
    print(f"Training date: {info['training_date']}")
    
    # Interactive prediction
    print("\nEnter news text for classification (or 'quit' to exit):")
    
    while True:
        text = input("\n> ")
        if text.lower() in ['quit', 'exit', 'q']:
            break
        
        if text.strip():
            try:
                result = loader.predict_single(text)
                print(f"Predicted Category: {result['predicted_class']}")
                print(f"Confidence: {result['confidence']:.3f}")
            except Exception as e:
                print(f"Error making prediction: {e}")
        else:
            print("Please enter some text")
    
    print("Goodbye!")

if __name__ == "__main__":
    main()
