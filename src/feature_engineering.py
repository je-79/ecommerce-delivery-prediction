
"""
Feature engineering and preprocessing module
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Handle feature engineering and preprocessing"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def encode_categorical_features(self) -> pd.DataFrame:
        """Encode categorical variables"""
        categorical_cols = ['Warehouse_block', 'Mode_of_Shipment', 
                          'Product_importance', 'Gender']
        
        print("Encoding categorical features...")
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
            
        print(f"✓ Encoded {len(categorical_cols)} categorical features")
        return self.df
    
    def create_features(self) -> pd.DataFrame:
        """Create additional features"""
        print("Creating additional features...")
        
        # Interaction features
        self.df['discount_weight_ratio'] = self.df['Discount_offered'] / (self.df['Weight_in_gms'] + 1)
        self.df['cost_per_gram'] = self.df['Cost_of_the_Product'] / (self.df['Weight_in_gms'] + 1)
        self.df['high_discount_flag'] = (self.df['Discount_offered'] > 15).astype(int)
        self.df['light_item_flag'] = (self.df['Weight_in_gms'] < 3000).astype(int)
        
        print("✓ Created 4 additional features")
        return self.df
    
    def prepare_train_test_split(self, 
                                 test_size: float = 0.2,
                                 random_state: int = 42) -> Tuple:
        """Prepare train-test split"""
        # Drop ID column
        df_model = self.df.drop('ID', axis=1)
        
        # Separate features and target
        X = df_model.drop('Reached.on.Time_Y.N', axis=1)
        y = df_model['Reached.on.Time_Y.N']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\n✓ Train set: {X_train.shape[0]} samples")
        print(f"✓ Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple:
        """Scale numerical features"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrames
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        print("✓ Features scaled")
        return X_train_scaled, X_test_scaled
    
    def get_feature_names(self) -> list:
        """Get list of feature names"""
        df_model = self.df.drop(['ID', 'Reached.on.Time_Y.N'], axis=1, errors='ignore')
        return list(df_model.columns)


if __name__ == "__main__":
    # Test feature engineering
    from data_loader import DataLoader
    
    loader = DataLoader('data/e_commerce.csv')
    df = loader.load_data()
    
    fe = FeatureEngineer(df)
    fe.encode_categorical_features()
    fe.create_features()
    X_train, X_test, y_train, y_test = fe.prepare_train_test_split()
    
    print(f"\n✓ Feature engineering working correctly!")
    print(f"✓ Total features: {len(fe.get_feature_names())}")

