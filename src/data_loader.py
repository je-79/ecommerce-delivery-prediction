
"""
Data loading & initial preprocessing module
"""

import pandas as pd
import numpy as np
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Load and perform initial data validation"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load CSV data"""
        print("Loading data...")
        self.df = pd.read_csv(self.filepath)
        print(f"✓ Data loaded: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        return self.df
    
    def validate_data(self) -> dict:
        """Validate data quality"""
        validation_report = {
            'shape': self.df.shape,
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicates': self.df.duplicated().sum(),
            'data_types': self.df.dtypes.to_dict()
        }
        
        print("\n" + "="*80)
        print("DATA VALIDATION REPORT")
        print("="*80)
        print(f"Shape: {validation_report['shape']}")
        print(f"Duplicates: {validation_report['duplicates']}")
        print(f"Missing values: {sum(validation_report['missing_values'].values())}")
        
        return validation_report
    
    def get_basic_stats(self) -> pd.DataFrame:
        """Get basic statistics"""
        return self.df.describe()
    
    def get_target_distribution(self, target_col: str = 'Reached.on.Time_Y.N') -> dict:
        """Get target variable distribution"""
        distribution = self.df[target_col].value_counts().to_dict()
        percentages = self.df[target_col].value_counts(normalize=True) * 100
        
        print("\n" + "="*80)
        print("TARGET VARIABLE DISTRIBUTION")
        print("="*80)
        print(f"Class 0 (On Time): {distribution.get(0, 0)} ({percentages.get(0, 0):.2f}%)")
        print(f"Class 1 (Delayed): {distribution.get(1, 0)} ({percentages.get(1, 0):.2f}%)")
        
        return {
            'counts': distribution,
            'percentages': percentages.to_dict()
        }


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader('data/e_commerce.csv')
    df = loader.load_data()
    loader.validate_data()
    loader.get_target_distribution()
    print("\n✓ Data loader working correctly!")

