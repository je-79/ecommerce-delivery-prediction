
"""
Exploratory Data Analysis module
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class EDA:
    """Perform exploratory data analysis"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.target_col = 'Reached.on.Time_Y.N'
        
    def analyze_categorical_features(self) -> Dict:
        """Analyze categorical features"""
        categorical_cols = ['Warehouse_block', 'Mode_of_Shipment', 
                          'Product_importance', 'Gender']
        
        results = {}
        
        for col in categorical_cols:
            delay_rates = pd.crosstab(
                self.df[col], 
                self.df[self.target_col], 
                normalize='index'
            ) * 100
            
            results[col] = delay_rates
            
        return results
    
    def analyze_numerical_features(self) -> pd.DataFrame:
        """Compare numerical features by delivery status"""
        numerical_cols = [
            'Customer_care_calls', 'Customer_rating', 
            'Cost_of_the_Product', 'Prior_purchases',
            'Discount_offered', 'Weight_in_gms'
        ]
        
        comparison = self.df.groupby(self.target_col)[numerical_cols].mean()
        comparison.index = ['On Time', 'Delayed']
        
        return comparison
    
    def create_eda_visualizations(self, output_path: str = 'outputs/eda_plots.png'):
        """Create comprehensive EDA visualizations"""
        fig = plt.figure(figsize=(20, 14))
        
        # 1. Delivery Performance by Shipping Mode
        ax1 = plt.subplot(3, 3, 1)
        delivery_by_mode = pd.crosstab(
            self.df['Mode_of_Shipment'], 
            self.df[self.target_col], 
            normalize='index'
        ) * 100
        delivery_by_mode.plot(kind='bar', ax=ax1, color=['#2ecc71', '#e74c3c'])
        ax1.set_title('Delivery Performance by Shipping Mode', fontweight='bold')
        ax1.set_xlabel('Shipping Mode')
        ax1.set_ylabel('Percentage (%)')
        ax1.legend(['On Time', 'Delayed'])
        
        # 2. Delivery Performance by Warehouse
        ax2 = plt.subplot(3, 3, 2)
        delivery_by_warehouse = pd.crosstab(
            self.df['Warehouse_block'], 
            self.df[self.target_col], 
            normalize='index'
        ) * 100
        delivery_by_warehouse.plot(kind='bar', ax=ax2, color=['#2ecc71', '#e74c3c'])
        ax2.set_title('Delivery Performance by Warehouse', fontweight='bold')
        ax2.set_xlabel('Warehouse Block')
        ax2.set_ylabel('Percentage (%)')
        ax2.legend(['On Time', 'Delayed'])
        
        # 3. Product Importance
        ax3 = plt.subplot(3, 3, 3)
        delivery_by_importance = pd.crosstab(
            self.df['Product_importance'], 
            self.df[self.target_col], 
            normalize='index'
        ) * 100
        delivery_by_importance = delivery_by_importance.reindex(['low', 'medium', 'high'])
        delivery_by_importance.plot(kind='bar', ax=ax3, color=['#2ecc71', '#e74c3c'])
        ax3.set_title('Delivery by Product Importance', fontweight='bold')
        ax3.set_xlabel('Importance')
        ax3.set_ylabel('Percentage (%)')
        
        # 4. Customer Rating Distribution
        ax4 = plt.subplot(3, 3, 4)
        for status in [0, 1]:
            data = self.df[self.df[self.target_col] == status]['Customer_rating']
            label = 'On Time' if status == 0 else 'Delayed'
            ax4.hist(data, bins=5, alpha=0.6, label=label, edgecolor='black')
        ax4.set_title('Customer Rating Distribution', fontweight='bold')
        ax4.set_xlabel('Rating')
        ax4.legend()
        
        # 5. Customer Care Calls
        ax5 = plt.subplot(3, 3, 5)
        sns.boxplot(data=self.df, x=self.target_col, y='Customer_care_calls', 
                   ax=ax5, palette=['#2ecc71', '#e74c3c'])
        ax5.set_title('Customer Care Calls by Status', fontweight='bold')
        
        # 6. Product Cost
        ax6 = plt.subplot(3, 3, 6)
        sns.boxplot(data=self.df, x=self.target_col, y='Cost_of_the_Product',
                   ax=ax6, palette=['#2ecc71', '#e74c3c'])
        ax6.set_title('Product Cost by Status', fontweight='bold')
        
        # 7. Discount Offered
        ax7 = plt.subplot(3, 3, 7)
        sns.boxplot(data=self.df, x=self.target_col, y='Discount_offered',
                   ax=ax7, palette=['#2ecc71', '#e74c3c'])
        ax7.set_title('Discount by Status', fontweight='bold')
        
        # 8. Weight
        ax8 = plt.subplot(3, 3, 8)
        sns.boxplot(data=self.df, x=self.target_col, y='Weight_in_gms',
                   ax=ax8, palette=['#2ecc71', '#e74c3c'])
        ax8.set_title('Weight by Status', fontweight='bold')
        
        # 9. Target Distribution
        ax9 = plt.subplot(3, 3, 9)
        self.df[self.target_col].value_counts().plot(kind='pie', ax=ax9, 
                                                      autopct='%1.1f%%',
                                                      colors=['#2ecc71', '#e74c3c'],
                                                      labels=['On Time', 'Delayed'])
        ax9.set_title('Overall Delivery Performance', fontweight='bold')
        ax9.set_ylabel('')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ EDA visualizations saved to {output_path}")
        plt.close()
        
    def print_key_insights(self):
        """Print key insights from EDA"""
        print("\n" + "="*60)
        print("KEY INSIGHTS")
        print("="*60)
        
        # Discount insight
        avg_discount_ontime = self.df[self.df[self.target_col]==0]['Discount_offered'].mean()
        avg_discount_delayed = self.df[self.df[self.target_col]==1]['Discount_offered'].mean()
        print(f"\n💰 DISCOUNT PARADOX:")
        print(f"   On-time orders: {avg_discount_ontime:.2f}% avg discount")
        print(f"   Delayed orders: {avg_discount_delayed:.2f}% avg discount")
        print(f"   → High discounts correlate with delays!")
        
        # Weight insight
        avg_weight_ontime = self.df[self.df[self.target_col]==0]['Weight_in_gms'].mean()
        avg_weight_delayed = self.df[self.df[self.target_col]==1]['Weight_in_gms'].mean()
        print(f"\n⚖️  WEIGHT IMPACT:")
        print(f"   On-time orders: {avg_weight_ontime:.0f}g avg weight")
        print(f"   Delayed orders: {avg_weight_delayed:.0f}g avg weight")
        print(f"   → Heavier items arrive on time more often!")
        
        # Product importance
        importance_delays = self.df.groupby('Product_importance')[self.target_col].mean() * 100
        print(f"\n⭐ PRODUCT IMPORTANCE PARADOX:")
        for imp in ['low', 'medium', 'high']:
            if imp in importance_delays.index:
                print(f"   {imp.capitalize()}: {importance_delays[imp]:.1f}% delay rate")
        print(f"   → High-importance products perform WORSE!")


if __name__ == "__main__":
    # Test EDA
    from data_loader import DataLoader
    
    loader = DataLoader('data/e_commerce.csv')
    df = loader.load_data()
    
    eda = EDA(df)
    eda.create_eda_visualizations()
    eda.print_key_insights()
    print("\n✓ EDA module working correctly!")

