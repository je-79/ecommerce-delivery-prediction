
"""
Main analysis pipeline
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from data_loader import DataLoader
from eda import EDA
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer


def main():
    """Run complete analysis pipeline"""
    print("="*80)
    print(" E-COMMERCE DELIVERY PREDICTION - COMPLETE ANALYSIS")
    print("="*80)
    
    # Step 1: Load Data
    print("\n[STEP 1/5] Loading Data...")
    loader = DataLoader('data/e_commerce.csv')
    df = loader.load_data()
    loader.validate_data()
    loader.get_target_distribution()
    
    # Step 2: Exploratory Data Analysis
    print("\n[STEP 2/5] Performing EDA...")
    eda = EDA(df)
    eda.create_eda_visualizations()
    eda.print_key_insights()
    
    # Step 3: Feature Engineering
    print("\n[STEP 3/5] Feature Engineering...")
    fe = FeatureEngineer(df)
    fe.encode_categorical_features()
    fe.create_features()
    X_train, X_test, y_train, y_test = fe.prepare_train_test_split()
    
    # Step 4: Model Training
    print("\n[STEP 4/5] Training Models...")
    trainer = ModelTrainer()
    trainer.initialize_models()
    trainer.train_models(X_train, y_train, X_test, y_test)
    
    # Step 5: Evaluation & Results
    print("\n[STEP 5/5] Generating Results...")
    print("\nModel Comparison:")
    print(trainer.get_model_comparison().to_string(index=False))
    
    trainer.print_classification_report()
    trainer.plot_model_comparison()
    trainer.save_best_model()
    
    print("\n" + "="*80)
    print(" ANALYSIS COMPLETE!")
    print("="*80)
    print("\n Outputs saved to:")
    print("   - outputs/eda_plots.png")
    print("   - outputs/model_comparison.png")
    print("   - models/best_model.pkl")
    print("\n✓ All done! Check the outputs folder for visualizations.")


if __name__ == "__main__":
    main()

