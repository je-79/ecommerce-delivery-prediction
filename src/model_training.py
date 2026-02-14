
"""
Model training and evaluation module
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, classification_report, 
                            confusion_matrix, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Train and evaluate machine learning models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def initialize_models(self):
        """Initialize ML models"""
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=42, max_iter=1000, solver='lbfgs'
            ),
            'Random Forest': RandomForestClassifier(
                random_state=42, n_estimators=100, max_depth=10
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=42, n_estimators=100, max_depth=5
            )
        }
        print(f"✓ Initialized {len(self.models)} models")
        
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train all models"""
        print("\n" + "="*80)
        print("TRAINING MODELS")
        print("="*80)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'y_test': y_test
            }
            
            self.results[name] = metrics
            
            # Print metrics
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            
        # Find best model based on F1-score
        best_name = max(self.results, key=lambda x: self.results[x]['f1_score'])
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        print(f"\n Best Model: {best_name}")
        
    def get_model_comparison(self) -> pd.DataFrame:
        """Get comparison of all models"""
        comparison = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [self.results[m]['accuracy'] for m in self.results],
            'Precision': [self.results[m]['precision'] for m in self.results],
            'Recall': [self.results[m]['recall'] for m in self.results],
            'F1-Score': [self.results[m]['f1_score'] for m in self.results],
            'ROC-AUC': [self.results[m]['roc_auc'] for m in self.results]
        })
        return comparison
    
    def print_classification_report(self, model_name: str = None):
        """Print detailed classification report"""
        if model_name is None:
            model_name = self.best_model_name
            
        metrics = self.results[model_name]
        y_test = metrics['y_test']
        y_pred = metrics['y_pred']
        
        print(f"\n" + "="*80)
        print(f"CLASSIFICATION REPORT - {model_name}")
        print("="*80)
        print(classification_report(y_test, y_pred, 
                                   target_names=['On Time', 'Delayed']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"           On Time  Delayed")
        print(f"On Time       {cm[0,0]:>4}    {cm[0,1]:>4}")
        print(f"Delayed       {cm[1,0]:>4}    {cm[1,1]:>4}")
        
    def plot_model_comparison(self, output_path: str = 'outputs/model_comparison.png'):
        """Plot model comparison"""
        comparison = self.get_model_comparison()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar plot
        ax1 = axes[0]
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        x = np.arange(len(comparison))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            offset = width * (i - 2)
            ax1.bar(x + offset, comparison[metric], width, label=metric)
            
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Comparison', fontweight='bold', pad=20)
        ax1.set_xticks(x)
        ax1.set_xticklabels(comparison['Model'], rotation=15)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # ROC Curves
        ax2 = axes[1]
        for name in self.results:
            y_test = self.results[name]['y_test']
            y_pred_proba = self.results[name]['y_pred_proba']
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = self.results[name]['roc_auc']
            ax2.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
            
        ax2.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curves', fontweight='bold', pad=20)
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Model comparison plot saved to {output_path}")
        plt.close()
        
    def save_best_model(self, filepath: str = 'models/best_model.pkl'):
        """Save the best model"""
        joblib.dump(self.best_model, filepath)
        print(f"✓ Best model ({self.best_model_name}) saved to {filepath}")
        
    def load_model(self, filepath: str):
        """Load a saved model"""
        self.best_model = joblib.load(filepath)
        print(f"✓ Model loaded from {filepath}")
        return self.best_model


if __name__ == "__main__":
    print("Model training module loaded successfully!")
