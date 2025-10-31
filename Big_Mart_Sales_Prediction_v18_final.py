"""
XGBoost Sales Prediction Pipeline

This script implements a complete machine learning pipeline for the Big Mart Sales
prediction problem. It includes:
- Data loading from S3.
- Extensive, leakage-safe feature engineering (including outlet efficiency & rank features).
- Hyperparameter tuning using Optuna.
- Optional Sequential Forward Selection (SFS) for feature selection.
- Bias correction factor calculation using Out-of-Fold (OOF) predictions.
- Final model training and submission file generation.

The pipeline is encapsulated in the `SalesPredictor` class.
To run, configure the constants in the 'CONFIGURATION' section and
execute the script.
"""

import time
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# --- 1. CONFIGURATION ---
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# -- File Paths and S3
S3_BUCKET_PATH = 's3://useast1-nlsn-dscihdamrpd-zoo-archive/users/vipulverma/Prediction_Tests_CSVs/'
TRAIN_FILE = 'train_v9rqX0R.csv'
TEST_FILE = 'test_AbJTz2l.csv'
SUBMISSION_FILE = 'submission_sfs_optimized_corrected_efficient_ranked.csv' # Changed file name

# -- Model & Data Constants
TARGET_VARIABLE = 'Item_Outlet_Sales'
ID_COLS = ['Item_Identifier', 'Outlet_Identifier']
CURRENT_YEAR = 2013
RANDOM_STATE = 4

# -- Pipeline Control
N_FOLDS = 5
N_OPTUNA_TRIALS = 200  # 55 for speed, 100+ for thoroughness

# -- Base XGBoost Parameters (for SFS and final training)
BASE_XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'n_jobs': -1,
    'seed': RANDOM_STATE
}

# --- 2. FEATURE SET DEFINITIONS ---

# All possible features created by the pipeline.
ALL_POSSIBLE_BASE_FEATURES = [
    'Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type',
    'Item_MRP', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type',
    'Item_Category', 'Outlet_Age'
]

ALL_POSSIBLE_ENGINEERED_FEATURES = [
    'Item_Popularity', 'Outlet_Mean_Sales', 'Outlet_Item_Count', 'Item_Age_Proxy',
    'Outlet_Food_Ratio', 'Outlet_Drinks_Ratio', 'Outlet_NC_Ratio',
    'Relative_Visibility', 'Visibility_to_Item_Mean_Ratio', 'Loc_Cat_Sales_Mean',
    'Price_per_Unit_Weight', 'Item_MRP_to_Outlet_Mean_MRP_Ratio',
    'Item_Saleability_Ratio', 'Outlet_to_Item_Age_Ratio', 'MRP_Bin',
    'Outlet_Low_MRP_Ratio', 'Outlet_Mid_MRP_Ratio', 'Outlet_High_MRP_Ratio',
    'Outlet_VHigh_MRP_Ratio', 'Outlet_Type_Food_Ratio', 'Outlet_Type_Drinks_Ratio',
    'Outlet_Type_NC_Ratio', 'is_Grocery_Store', 'is_Supermarket_Type1',
    'High_Visibility_Flag', 'Visibility_per_MRP', 'Price_per_Category_Mean',
    'Outlet_Era', 'Location_x_Type', 'Vis_per_Category_Mean', 'Item_MRP_sq',
    'Outlet_Sales_Efficiency',
    'Price_Rank_in_Category', 'Visibility_Rank_in_Category'  # <-- NEW FEATURES
]

# --- Manually selected features (used if SFS is skipped) ---
MANUAL_BASE_FEATURES = [
    'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP',
    'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Outlet_Age'
]

MANUAL_ENGINEERED_FEATURES = [
    'Outlet_Food_Ratio', 'Outlet_Drinks_Ratio', 'Outlet_NC_Ratio',
    'is_Grocery_Store', 'High_Visibility_Flag', 'MRP_Bin',
    'Outlet_Sales_Efficiency'
]


# --- 3. PIPELINE CLASS ---

class SalesPredictor:
    """
    Encapsulates the entire sales prediction pipeline.
    """
    def __init__(self, config):
        """
        Initializes the pipeline with configuration settings.
        """
        self.bucket_path = config['S3_BUCKET_PATH']
        self.train_file = config['TRAIN_FILE']
        self.test_file = config['TEST_FILE']
        self.submission_file = config['SUBMISSION_FILE']
        self.target = config['TARGET_VARIABLE']
        self.id_cols = config['ID_COLS']
        self.random_state = config['RANDOM_STATE']
        self.n_folds = config['N_FOLDS']
        self.n_optuna_trials = config['N_OPTUNA_TRIALS']
        self.base_xgb_params = config['BASE_XGB_PARAMS']

        # Data attributes
        self.train_df = None
        self.test_df = None
        self.train_processed = None
        self.test_processed = None
        self.original_test_ids = None

        # Model and feature attributes
        self.optimal_features = None
        self.final_params = None
        self.final_model = None
        self.correction_factor = 1.0

    def load_data(self):
        """Loads training and testing data from the specified S3 path."""
        print("--- [Step 1] Loading Data ---")
        try:
            self.train_df = pd.read_csv(self.bucket_path + self.train_file)
            self.test_df = pd.read_csv(self.bucket_path + self.test_file)
            self.original_test_ids = self.test_df[self.id_cols].copy()
            print(f"Train data loaded: {self.train_df.shape}")
            print(f"Test data loaded: {self.test_df.shape}")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    # --- Preprocessing & Feature Engineering Methods ---

    def _clean_basic_features(self):
        """Applies basic data cleaning and simple feature creation."""
        print("... 2a. Applying basic cleaning...")
        for df in [self.train_processed, self.test_processed]:
            # Clean Item_Fat_Content
            df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
                'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'
            })
            # Create Item_Category
            df['Item_Category_Raw'] = df['Item_Identifier'].apply(lambda x: x[0:2])
            df['Item_Category'] = df['Item_Category_Raw'].map({
                'FD': 'Food', 'DR': 'Drinks', 'NC': 'Non-Consumable'
            })
            # Correct fat content for non-consumables
            df.loc[df['Item_Category'] == 'Non-Consumable', 'Item_Fat_Content'] = 'Non-Edible'
            # Create Outlet_Age
            df['Outlet_Age'] = CURRENT_YEAR - df['Outlet_Establishment_Year']
            # Create flags
            df['is_Grocery_Store'] = (df['Outlet_Type'] == 'Grocery Store').astype(int)
            df['is_Supermarket_Type1'] = (df['Outlet_Type'] == 'Supermarket Type1').astype(int)

    def _impute_missing_values(self):
        """Imputes missing values for Outlet_Size, Item_Weight, and Item_Visibility."""
        print("... 2b. Imputing missing values...")
        # Outlet_Size (simple fill)
        self.train_processed['Outlet_Size'].fillna('Small', inplace=True)
        self.test_processed['Outlet_Size'].fillna('Small', inplace=True)

        # Item_Weight (complex fill)
        item_avg_weight = self.train_processed.pivot_table(values='Item_Weight', index='Item_Identifier')
        self.train_processed['Item_Weight'].fillna(self.train_processed['Item_Identifier'].map(item_avg_weight['Item_Weight']), inplace=True)
        self.test_processed['Item_Weight'].fillna(self.test_processed['Item_Identifier'].map(item_avg_weight['Item_Weight']), inplace=True)
        
        item_type_avg_weight = self.train_processed.groupby('Item_Type')['Item_Weight'].transform('mean')
        self.train_processed['Item_Weight'].fillna(item_type_avg_weight, inplace=True)
        self.test_processed['Item_Weight'].fillna(self.test_processed.groupby('Item_Type')['Item_Weight'].transform('mean'), inplace=True)

        # Item_Visibility (complex fill)
        self.train_processed['Item_Visibility'].replace(0, np.nan, inplace=True)
        self.test_processed['Item_Visibility'].replace(0, np.nan, inplace=True)
        
        visibility_avg = self.train_processed.groupby(['Outlet_Identifier', 'Item_Type'])['Item_Visibility'].transform('mean')
        self.train_processed['Item_Visibility'].fillna(visibility_avg, inplace=True)
        self.test_processed['Item_Visibility'].fillna(self.test_processed.groupby(['Outlet_Identifier', 'Item_Type'])['Item_Visibility'].transform('mean'), inplace=True)
        
        self.train_processed['Item_Visibility'].fillna(self.train_processed.groupby('Item_Type')['Item_Visibility'].transform('mean'), inplace=True)
        self.test_processed['Item_Visibility'].fillna(self.test_processed.groupby('Item_Type')['Item_Visibility'].transform('mean'), inplace=True)

    # --- NEW METHOD ---
    def _create_rank_features(self):
        """Creates percentile rank features for price and visibility within category."""
        print("... 2c. Creating rank features...")
        
        # Combine train and test to calculate global ranks
        # This is safe as MRP and Visibility are intrinsic item properties, not target-related
        train_len = len(self.train_processed)
        combined_df = pd.concat([self.train_processed, self.test_processed])

        # Ensure 'Item_Type' (which we group by) is present
        if 'Item_Type' not in combined_df.columns:
             print("Warning: 'Item_Type' not found, cannot create rank features.")
             return
        
        combined_df['Price_Rank_in_Category'] = combined_df.groupby('Item_Type')['Item_MRP'].rank(pct=True)
        combined_df['Visibility_Rank_in_Category'] = combined_df.groupby('Item_Type')['Item_Visibility'].rank(pct=True)

        # Separate back into train and test
        self.train_processed = combined_df.iloc[:train_len].copy()
        self.test_processed = combined_df.iloc[train_len:].copy()


    def _create_visibility_features(self):
        """Creates visibility-related ratio features."""
        print("... 2d. Creating visibility features...")
        item_avg_visibility = self.train_processed.groupby('Item_Identifier')['Item_Visibility'].mean().reset_index().rename(columns={'Item_Visibility': 'Item_Mean_Visibility'})
        self.train_processed = self.train_processed.merge(item_avg_visibility, on='Item_Identifier', how='left')
        self.test_processed = self.test_processed.merge(item_avg_visibility, on='Item_Identifier', how='left')
        
        global_mean_visibility = item_avg_visibility['Item_Mean_Visibility'].mean()
        self.test_processed['Item_Mean_Visibility'].fillna(global_mean_visibility, inplace=True)
        
        self.train_processed['Visibility_to_Item_Mean_Ratio'] = self.train_processed['Item_Visibility'] / (self.train_processed['Item_Mean_Visibility'] + 1e-6)
        self.test_processed['Visibility_to_Item_Mean_Ratio'] = self.test_processed['Item_Visibility'] / (self.test_processed['Item_Mean_Visibility'] + 1e-6)

    def _create_contextual_features(self):
        """Creates features based on item/outlet context."""
        print("... 2e. Creating contextual features (popularity, age)...")
        # Item Popularity
        item_counts = self.train_processed['Item_Identifier'].value_counts()
        self.train_processed['Item_Popularity'] = self.train_processed['Item_Identifier'].map(item_counts)
        self.test_processed['Item_Popularity'] = self.test_processed['Item_Identifier'].map(item_counts).fillna(item_counts.median())

        # Outlet Stats
        outlet_stats = self.train_processed.groupby('Outlet_Identifier').agg(
            {'Item_Identifier': 'nunique', self.target: 'mean'}
        ).rename(columns={'Item_Identifier': 'Outlet_Item_Count', self.target: 'Outlet_Mean_Sales'})
        self.train_processed = self.train_processed.merge(outlet_stats, on='Outlet_Identifier', how='left')
        self.test_processed = self.test_processed.merge(outlet_stats, on='Outlet_Identifier', how='left')
        self.test_processed['Outlet_Item_Count'].fillna(outlet_stats['Outlet_Item_Count'].median(), inplace=True)
        self.test_processed['Outlet_Mean_Sales'].fillna(outlet_stats['Outlet_Mean_Sales'].mean(), inplace=True)

        # Item Age
        item_first_appearance = self.train_processed.groupby('Item_Identifier')['Outlet_Establishment_Year'].min().reset_index().rename(columns={'Outlet_Establishment_Year': 'Item_First_Appearance_Year'})
        self.train_processed = self.train_processed.merge(item_first_appearance, on='Item_Identifier', how='left')
        self.test_processed = self.test_processed.merge(item_first_appearance, on='Item_Identifier', how='left')
        self.train_processed['Item_Age_Proxy'] = CURRENT_YEAR - self.train_processed['Item_First_Appearance_Year']
        self.test_processed['Item_Age_Proxy'] = CURRENT_YEAR - self.test_processed['Item_First_Appearance_Year']
        self.test_processed['Item_Age_Proxy'].fillna(self.train_processed['Item_Age_Proxy'].median(), inplace=True)
        
        # Outlet Age vs Item Age
        self.train_processed['Outlet_to_Item_Age_Ratio'] = self.train_processed['Outlet_Age'] / (self.train_processed['Item_Age_Proxy'] + 1)
        self.test_processed['Outlet_to_Item_Age_Ratio'] = self.test_processed['Outlet_Age'] / (self.test_processed['Item_Age_Proxy'] + 1)

    def _create_ratio_features(self):
        """Creates various ratio and profile features."""
        print("... 2f. Creating advanced ratio/profile features...")
        
        # Outlet Category Ratios
        outlet_category_dist = self.train_processed.groupby(['Outlet_Identifier', 'Item_Category'])['Item_Identifier'].nunique().unstack(fill_value=0)
        outlet_category_dist['Total_Unique_Items'] = outlet_category_dist.sum(axis=1)
        outlet_category_dist['Outlet_Food_Ratio'] = outlet_category_dist['Food'] / outlet_category_dist['Total_Unique_Items']
        outlet_category_dist['Outlet_Drinks_Ratio'] = outlet_category_dist['Drinks'] / outlet_category_dist['Total_Unique_Items']
        outlet_category_dist['Outlet_NC_Ratio'] = outlet_category_dist['Non-Consumable'] / outlet_category_dist['Total_Unique_Items']
        cat_ratio_cols = ['Outlet_Food_Ratio', 'Outlet_Drinks_Ratio', 'Outlet_NC_Ratio']
        self.train_processed = self.train_processed.merge(outlet_category_dist[cat_ratio_cols], on='Outlet_Identifier', how='left')
        self.test_processed = self.test_processed.merge(outlet_category_dist[cat_ratio_cols], on='Outlet_Identifier', how='left')
        self.test_processed[cat_ratio_cols] = self.test_processed[cat_ratio_cols].fillna(self.train_processed[cat_ratio_cols].median())

        # Relative Visibility
        avg_vis_in_outlet_by_type = self.train_processed.groupby(['Outlet_Identifier', 'Item_Type'])['Item_Visibility'].mean().reset_index().rename(columns={'Item_Visibility': 'Avg_Vis_Type_in_Outlet'})
        self.train_processed = self.train_processed.merge(avg_vis_in_outlet_by_type, on=['Outlet_Identifier', 'Item_Type'], how='left')
        self.test_processed = self.test_processed.merge(avg_vis_in_outlet_by_type, on=['Outlet_Identifier', 'Item_Type'], how='left')
        self.train_processed['Relative_Visibility'] = self.train_processed['Item_Visibility'] / self.train_processed['Avg_Vis_Type_in_Outlet']
        self.test_processed['Relative_Visibility'] = self.test_processed['Item_Visibility'] / self.test_processed['Avg_Vis_Type_in_Outlet']
        self.test_processed['Relative_Visibility'].fillna(1, inplace=True)
        self.train_processed['Relative_Visibility'].fillna(1, inplace=True)

        # Location/Category Sales Mean
        loc_cat_sales = self.train_processed.groupby(['Outlet_Location_Type', 'Item_Category'])[self.target].mean().reset_index().rename(columns={self.target: 'Loc_Cat_Sales_Mean'})
        self.train_processed = self.train_processed.merge(loc_cat_sales, on=['Outlet_Location_Type', 'Item_Category'], how='left')
        self.test_processed = self.test_processed.merge(loc_cat_sales, on=['Outlet_Location_Type', 'Item_Category'], how='left')
        self.test_processed['Loc_Cat_Sales_Mean'].fillna(self.train_df[self.target].mean(), inplace=True)

    def _create_mrp_features(self):
        """Creates features based on Item_MRP."""
        print("... 2g. Creating MRP-based features...")
        for df in [self.train_processed, self.test_processed]:
            df['High_Visibility_Flag'] = (df['Item_Visibility'] > 0.19).astype(int)
            df['Visibility_per_MRP'] = df['Item_Visibility'] / (df['Item_MRP'] + 1e-6)
            
            bin_edges = [0, 70, 135, 200, np.inf]
            bin_labels = ['Low_MRP', 'Mid_MRP', 'High_MRP', 'VHigh_MRP']
            df['MRP_Bin'] = pd.cut(df['Item_MRP'], bins=bin_edges, labels=bin_labels, right=False)
            
            df['Price_per_Unit_Weight'] = df['Item_MRP'] / (df['Item_Weight'] + 1e-6)
            df['Item_MRP_sq'] = df['Item_MRP']**2
            df['MRP_Bin'] = df['MRP_Bin'].astype(str) # For groupby

        # Outlet/MRP Ratios
        outlet_avg_mrp = self.train_processed.groupby('Outlet_Identifier')['Item_MRP'].mean().reset_index().rename(columns={'Item_MRP': 'Outlet_Mean_MRP'})
        self.train_processed = self.train_processed.merge(outlet_avg_mrp, on='Outlet_Identifier', how='left')
        self.test_processed = self.test_processed.merge(outlet_avg_mrp, on='Outlet_Identifier', how='left')
        self.test_processed['Outlet_Mean_MRP'].fillna(self.train_processed['Outlet_Mean_MRP'].median(), inplace=True)
        self.train_processed['Item_MRP_to_Outlet_Mean_MRP_Ratio'] = self.train_processed['Item_MRP'] / self.train_processed['Outlet_Mean_MRP']
        self.test_processed['Item_MRP_to_Outlet_Mean_MRP_Ratio'] = self.test_processed['Item_MRP'] / self.test_processed['Outlet_Mean_MRP']

        # Outlet MRP Profile
        outlet_mrp_dist = self.train_processed.groupby(['Outlet_Identifier', 'MRP_Bin'])['Item_Identifier'].nunique().unstack(fill_value=0)
        outlet_mrp_dist['Total_Unique_Items'] = outlet_mrp_dist.sum(axis=1)
        for bin_label in ['Low_MRP', 'Mid_MRP', 'High_MRP', 'VHigh_MRP']:
            outlet_mrp_dist[f'Outlet_{bin_label}_Ratio'] = outlet_mrp_dist[bin_label] / outlet_mrp_dist['Total_Unique_Items']
        mrp_ratio_cols = [f'Outlet_{label}_Ratio' for label in ['Low_MRP', 'Mid_MRP', 'High_MRP', 'VHigh_MRP']]
        self.train_processed = self.train_processed.merge(outlet_mrp_dist[mrp_ratio_cols], on='Outlet_Identifier', how='left')
        self.test_processed = self.test_processed.merge(outlet_mrp_dist[mrp_ratio_cols], on='Outlet_Identifier', how='left')
        self.test_processed[mrp_ratio_cols] = self.test_processed[mrp_ratio_cols].fillna(self.train_processed[mrp_ratio_cols].median())

    def _create_outlet_type_features(self):
        """Creates features based on Outlet_Type."""
        print("... 2h. Creating outlet type profile features...")
        outlet_type_dist = self.train_processed.groupby(['Outlet_Type', 'Item_Category'])['Item_Identifier'].nunique().unstack(fill_value=0)
        outlet_type_dist['Total_Unique_Items'] = outlet_type_dist.sum(axis=1)
        outlet_type_dist['Outlet_Type_Food_Ratio'] = outlet_type_dist['Food'] / outlet_type_dist['Total_Unique_Items']
        outlet_type_dist['Outlet_Type_Drinks_Ratio'] = outlet_type_dist['Drinks'] / outlet_type_dist['Total_Unique_Items']
        outlet_type_dist['Outlet_Type_NC_Ratio'] = outlet_type_dist['Non-Consumable'] / outlet_type_dist['Total_Unique_Items']
        type_ratio_cols = ['Outlet_Type_Food_Ratio', 'Outlet_Type_Drinks_Ratio', 'Outlet_Type_NC_Ratio']
        self.train_processed = self.train_processed.merge(outlet_type_dist[type_ratio_cols], on='Outlet_Type', how='left')
        self.test_processed = self.test_processed.merge(outlet_type_dist[type_ratio_cols], on='Outlet_Type', how='left')
        self.test_processed[type_ratio_cols] = self.test_processed[type_ratio_cols].fillna(self.train_processed[type_ratio_cols].median())

    def _create_target_encoded_features(self):
        """Creates leakage-safe target encoded features for item saleability."""
        print("... 2i. Creating leakage-safe target encoded features (Item Saleability)...")
        global_saleability = self.train_processed[self.target].mean() / (self.train_processed['Item_MRP'].mean() + 1e-6)
        self.train_processed['Item_Saleability_Ratio'] = np.nan
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        for train_idx, val_idx in kf.split(self.train_processed):
            train_fold = self.train_processed.iloc[train_idx]
            item_saleability_fold = train_fold.groupby('Item_Identifier').agg({self.target: 'mean', 'Item_MRP': 'mean'}).reset_index()
            item_saleability_fold['Item_Saleability_Ratio'] = item_saleability_fold[self.target] / (item_saleability_fold['Item_MRP'] + 1e-6)
            
            val_fold = self.train_processed.iloc[val_idx]
            merged_val = val_fold[['Item_Identifier']].merge(item_saleability_fold[['Item_Identifier', 'Item_Saleability_Ratio']], on='Item_Identifier', how='left')
            self.train_processed.loc[val_idx, 'Item_Saleability_Ratio'] = merged_val['Item_Saleability_Ratio'].values
            
        self.train_processed['Item_Saleability_Ratio'].fillna(global_saleability, inplace=True)
        
        # For test data, use the full training set
        item_saleability_full = self.train_processed.groupby('Item_Identifier').agg({self.target: 'mean', 'Item_MRP': 'mean'}).reset_index()
        item_saleability_full['Item_Saleability_Ratio'] = item_saleability_full[self.target] / (item_saleability_full['Item_MRP'] + 1e-6)
        self.test_processed = self.test_processed.merge(item_saleability_full[['Item_Identifier', 'Item_Saleability_Ratio']], on='Item_Identifier', how='left')
        self.test_processed['Item_Saleability_Ratio'].fillna(global_saleability, inplace=True)

    def _create_outlet_efficiency_feature(self):
        """Creates a leakage-safe target encoded feature for outlet efficiency."""
        print("... 2j. Creating leakage-safe target encoded features (Outlet Efficiency)...")

        # --- Calculate global stats first (for test set and NaN filling) ---
        outlet_stats_full = self.train_processed.groupby('Outlet_Identifier').agg(
            Outlet_Mean_Sales=(self.target, 'mean'),
            Outlet_Mean_MRP=('Item_MRP', 'mean')
        )
        outlet_stats_full['Outlet_Sales_Efficiency'] = outlet_stats_full['Outlet_Mean_Sales'] / (outlet_stats_full['Outlet_Mean_MRP'] + 1e-6)
        global_avg_efficiency = outlet_stats_full['Outlet_Sales_Efficiency'].mean()

        # --- Create feature for training set (leakage-safe) ---
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        self.train_processed['Outlet_Sales_Efficiency'] = np.nan

        for train_idx, val_idx in kf.split(self.train_processed):
            train_fold = self.train_processed.iloc[train_idx]
            
            # Calculate stats using ONLY the training part of the fold
            outlet_stats_fold = train_fold.groupby('Outlet_Identifier').agg(
                Outlet_Mean_Sales=(self.target, 'mean'),
                Outlet_Mean_MRP=('Item_MRP', 'mean')
            )
            outlet_stats_fold['Outlet_Sales_Efficiency'] = outlet_stats_fold['Outlet_Mean_Sales'] / (outlet_stats_fold['Outlet_Mean_MRP'] + 1e-6)
            
            # Apply these stats to the validation part of the fold
            val_fold = self.train_processed.iloc[val_idx]
            merged_val = val_fold[['Outlet_Identifier']].merge(
                outlet_stats_fold[['Outlet_Sales_Efficiency']],
                on='Outlet_Identifier',
                how='left'
            )
            self.train_processed.loc[val_idx, 'Outlet_Sales_Efficiency'] = merged_val['Outlet_Sales_Efficiency'].values

        # Fill any NaNs left over in train set
        self.train_processed['Outlet_Sales_Efficiency'].fillna(global_avg_efficiency, inplace=True)

        # --- Create feature for test set (using full train stats) ---
        self.test_processed = self.test_processed.merge(
            outlet_stats_full[['Outlet_Sales_Efficiency']], 
            on='Outlet_Identifier', 
            how='left'
        )
        # Handle any new outlets in test set (cold start)
        self.test_processed['Outlet_Sales_Efficiency'].fillna(global_avg_efficiency, inplace=True)


    def _create_experimental_features(self):
        """Creates additional experimental features."""
        print("... 2k. Creating experimental features...")
        
        # Price relative to Category
        item_type_mrp_map = self.train_processed.groupby('Item_Type')['Item_MRP'].mean()
        self.train_processed['Price_per_Category_Mean'] = self.train_processed['Item_MRP'] / (self.train_processed['Item_Type'].map(item_type_mrp_map) + 1e-6)
        self.test_processed['Price_per_Category_Mean'] = self.test_processed['Item_MRP'] / (self.test_processed['Item_Type'].map(item_type_mrp_map) + 1e-6)
        self.test_processed['Price_per_Category_Mean'].fillna(1, inplace=True)

        # Visibility Relative to Category
        item_type_vis_map = self.train_processed.groupby('Item_Type')['Item_Visibility'].mean()
        self.train_processed['Vis_per_Category_Mean'] = self.train_processed['Item_Visibility'] / (self.train_processed['Item_Type'].map(item_type_vis_map) + 1e-6)
        self.test_processed['Vis_per_Category_Mean'] = self.test_processed['Item_Visibility'] / (self.test_processed['Item_Type'].map(item_type_vis_map) + 1e-6)
        self.test_processed['Vis_per_Category_Mean'].fillna(1, inplace=True)
        
        # Outlet "Vintage" or "Era"
        def get_era(year):
            if year < 1990: return '1980s'
            elif year < 2000: return '1990s'
            else: return '2000s'
            
        self.train_processed['Outlet_Establishment_Year'] = CURRENT_YEAR - self.train_processed['Outlet_Age'] 
        self.test_processed['Outlet_Establishment_Year'] = CURRENT_YEAR - self.test_processed['Outlet_Age'] 
        self.train_processed['Outlet_Era'] = self.train_processed['Outlet_Establishment_Year'].apply(get_era)
        self.test_processed['Outlet_Era'] = self.test_processed['Outlet_Establishment_Year'].apply(get_era)

        # High-Level Interaction Feature
        self.train_processed['Location_x_Type'] = self.train_processed['Outlet_Location_Type'].astype(str) + '_' + self.train_processed['Outlet_Type'].astype(str)
        self.test_processed['Location_x_Type'] = self.test_processed['Outlet_Location_Type'].astype(str) + '_' + self.test_processed['Outlet_Type'].astype(str)

    def _apply_final_encoding(self):
        """Applies label encoding to all object columns and cleans up."""
        print("... 2l. Applying final encoding and cleanup...")
        
        cols_to_drop = [
            'Outlet_Establishment_Year', 'Item_First_Appearance_Year', 
            'Avg_Vis_Type_in_Outlet', 'Item_Category_Raw', 'Outlet_Mean_MRP', 
            'Item_Mean_Visibility'
        ]
        self.train_processed.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        self.test_processed.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        
        for col in self.train_processed.select_dtypes(include=['object']).columns:
            if col not in self.id_cols:
                le = LabelEncoder()
                self.train_processed[col] = le.fit_transform(self.train_processed[col].astype(str))
                
                # Handle unseen labels in test data
                self.test_processed[col] = self.test_processed[col].astype(str).map(lambda s: '<unknown>' if s not in le.classes_ else s)
                le.classes_ = np.append(le.classes_, '<unknown>')
                self.test_processed[col] = le.transform(self.test_processed[col])

    def preprocess_and_feature_engineer(self):
        """
        Main orchestration method for all preprocessing and
        feature engineering steps.
        """
        if self.train_df is None or self.test_df is None:
            print("Data not loaded. Please run load_data() first.")
            return

        print("\n--- [Step 2] Preprocessing & Feature Engineering ---")
        self.train_processed = self.train_df.copy()
        self.test_processed = self.test_df.copy()

        # Execute all preprocessing steps in order
        self._clean_basic_features()
        self._impute_missing_values()
        self._create_rank_features() # <-- NEW STEP
        self._create_visibility_features()
        self._create_contextual_features()
        self._create_ratio_features()
        self._create_mrp_features()
        self._create_outlet_type_features()
        self._create_target_encoded_features()
        self._create_outlet_efficiency_feature() 
        self._create_experimental_features()
        self._apply_final_encoding()

        print("--- Preprocessing & Feature Engineering Complete ---")

    # --- Model Training & Tuning Methods ---

    def optimize_hyperparameters(self, predictors):
        """Uses Optuna to find the best hyperparameters for the model."""
        print(f"Starting Optuna optimization for {len(predictors)} features...")
        y = self.train_processed[self.target]
        X = self.train_processed[predictors]
        dtrain = xgb.DMatrix(X, label=y)
        
        def objective(trial):
            params = {
                **self.base_xgb_params,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                
                'lambda': trial.suggest_float('lambda', 1.0, 50.0, log=True), # Increase L2
                'alpha': trial.suggest_float('alpha', 0.1, 50.0, log=True),   # Increase L1

                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True)
            }
            cv_results = xgb.cv(
                params, dtrain, 1000, self.n_folds,
                early_stopping_rounds=50, metrics={'rmse'},
                seed=self.random_state, verbose_eval=False
            )
            return cv_results['test-rmse-mean'].min()

        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(objective, n_trials=self.n_optuna_trials)
        
        print(f"Optuna complete. Best trial RMSE: {study.best_value:.4f}")
        return {**self.base_xgb_params, **study.best_params}

    def perform_sfs(self, base_predictors, engineered_features):
        """Performs Sequential Forward Selection to find the optimal feature set."""
        print("🚀 Starting Sequential Forward Selection (SFS)...")
        start_time = time.time()
        
        selected_features = base_predictors.copy()
        remaining_features = [f for f in engineered_features if f not in selected_features]
        
        # Use simple, fast params for SFS
        sfs_params = self.base_xgb_params.copy()
        sfs_params.update({'learning_rate': 0.1, 'max_depth': 5})
        
        def evaluate_features(predictors):
            # Ensure no duplicates
            predictors = list(dict.fromkeys(predictors))
            y = self.train_processed[self.target]
            X = self.train_processed[predictors]
            dtrain = xgb.DMatrix(X, label=y)
            cv_results = xgb.cv(
                sfs_params, dtrain, 1000, self.n_folds,
                early_stopping_rounds=50, metrics={'rmse'},
                seed=self.random_state, verbose_eval=False
            )
            return cv_results['test-rmse-mean'].min()

        best_overall_rmse = evaluate_features(selected_features)
        print(f"Baseline RMSE with {len(selected_features)} base features: {best_overall_rmse:.4f}\n")
        
        iteration = 1
        while remaining_features:
            print(f"--- SFS Iteration {iteration}: Testing {len(remaining_features)} remaining features ---")
            results_this_round = {}
            for feature in remaining_features:
                current_features_to_test = selected_features + [feature]
                current_rmse = evaluate_features(current_features_to_test)
                results_this_round[feature] = current_rmse
                # print(f"  -> Testing '{feature}': CV RMSE = {current_rmse:.4f}")

            best_feature_this_round, best_rmse_this_round = min(results_this_round.items(), key=lambda item: item[1])

            if best_rmse_this_round < best_overall_rmse:
                print(f"\n✅ Improvement Found! Adding '{best_feature_this_round}'.")
                print(f"  New Best RMSE: {best_rmse_this_round:.4f} (Old: {best_overall_rmse:.4f})\n")
                best_overall_rmse = best_rmse_this_round
                selected_features.append(best_feature_this_round)
                remaining_features.remove(best_feature_this_round)
            else:
                print("\n🛑 No further improvement found. Stopping SFS.")
                break
            iteration += 1

        end_time = time.time()
        print(f"\n--- SFS Complete in {end_time - start_time:.2f} seconds ---")
        print(f"Final Selected Features ({len(selected_features)}): {selected_features}")
        return list(dict.fromkeys(selected_features)), best_overall_rmse

    def train_xgboost_model(self, predictors, params):
        """Trains the final XGBoost model using optimal rounds from CV."""
        y = self.train_processed[self.target]
        X = self.train_processed[predictors]
        dtrain = xgb.DMatrix(X, label=y)
        
        # Use CV to find the best number of rounds
        cv_results = xgb.cv(
            params, dtrain, 1000, self.n_folds,
            early_stopping_rounds=100, metrics={'rmse'},
            seed=self.random_state, verbose_eval=False
        )
        best_boost_round = cv_results['test-rmse-mean'].idxmin() + 1
        best_rmse_mean = cv_results['test-rmse-mean'].min()
        print(f"Training final model with {best_boost_round} rounds.")
        print(f"  (This model's mean CV RMSE was: {best_rmse_mean:.4f})")
        
        # Train the final model
        final_model = xgb.train(params, dtrain, num_boost_round=best_boost_round)
        return final_model, best_rmse_mean

    def generate_oof_predictions_and_factor(self, predictors, params):
        """
        Generates Out-of-Fold (OOF) predictions and calculates the
        optimal bias correction factor.
        """
        print("... 5a. Generating OOF predictions for bias calculation...")
        y = self.train_processed[self.target]
        X = self.train_processed[predictors]

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        # Arrays to store OOF data
        oof_preds = np.zeros(X.shape[0])
        oof_true = np.zeros(X.shape[0])

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            # print(f"... Fold {fold+1}/{self.n_folds} ...")
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            # Train model with early stopping
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                evals=[(dval, 'val')],
                early_stopping_rounds=100,
                verbose_eval=False
            )
            
            # Store predictions and true values
            fold_preds = model.predict(dval, iteration_range=(0, model.best_iteration))
            oof_preds[val_idx] = fold_preds
            oof_true[val_idx] = y_val

        # Now calculate the factor and compare RMSEs
        print("... 5b. Calculating correction factor...")
        
        # 1. RMSE Before correction
        rmse_before = np.sqrt(mean_squared_error(oof_true, oof_preds))
        print(f"  OOF RMSE (Before Correction): {rmse_before:.4f}")

        # 2. Find optimal correction factor c
        numerator = (oof_true * oof_preds).sum()
        denominator = (oof_preds**2).sum()
        optimal_c = numerator / denominator
        print(f"  Optimal Correction Factor (c): {optimal_c:.6f}")

        # 3. RMSE After correction
        corrected_oof_preds = oof_preds * optimal_c
        rmse_after = np.sqrt(mean_squared_error(oof_true, corrected_oof_preds))
        print(f"  OOF RMSE (After Correction):  {rmse_after:.4f}")
        
        return optimal_c

    def generate_submission(self, model, predictors, model_name="final_model"):
        """Generates predictions and saves the submission file."""
        print(f"Generating predictions with model: {model_name}")
        
        dtest = xgb.DMatrix(self.test_processed[predictors])
        predictions = model.predict(dtest)
        
        # --- Apply Correction ---
        print(f"... Applying correction factor of: {self.correction_factor:.6f}")
        corrected_predictions = predictions * self.correction_factor
        
        # Ensure predictions are non-negative
        corrected_predictions = np.clip(corrected_predictions, 0, None)
        
        submission_df = pd.DataFrame({
            'Item_Identifier': self.original_test_ids['Item_Identifier'],
            'Outlet_Identifier': self.original_test_ids['Outlet_Identifier'],
            self.target: corrected_predictions # Use corrected predictions
        })
        
        # print("--- Submission Dataframe Head ---")
        # print(submission_df.head())
        
        submission_df.display()
        print(f"\n✅ Submission data on display")

    # --- Main Pipeline Execution ---

    def run(self, run_sfs=False, base_features=None, engineered_features=None):
        """
        Executes the end-to-end pipeline.
        
        Args:
            run_sfs (bool): If True, run SFS. If False, use manual features.
            base_features (list): List of base features to use.
            engineered_features (list): List of engineered features to test (SFS)
                                            or use (manual).
        """
        start_pipeline_time = time.time()
        
        # Step 1: Load Data
        self.load_data()
        
        # Step 2: Preprocess
        self.preprocess_and_feature_engineer()

        # Clean up feature lists to ensure no duplicates
        base_features = list(dict.fromkeys(base_features))
        engineered_features = list(dict.fromkeys(f for f in engineered_features if f not in base_features))
        
        # Step 3: Feature Selection
        if run_sfs:
            print("\n--- [Step 3] Running Sequential Forward Selection ---")
            self.optimal_features, _ = self.perform_sfs(
                base_predictors=base_features,
                engineered_features=engineered_features
            )
        else:
            print("\n--- [Step 3] Skipping SFS, using manual features ---")
            self.optimal_features = list(dict.fromkeys(base_features + engineered_features))
            print(f"Using {len(self.optimal_features)} manually selected features.")

        # Step 4: Hyperparameter Tuning
        print("\n--- [Step 4] Final Hyperparameter Tuning ---")
        self.final_params = self.optimize_hyperparameters(self.optimal_features)

        # Step 5: Calculate Bias Correction Factor
        print("\n--- [Step 5] Calculating Bias Correction Factor ---")
        self.correction_factor = self.generate_oof_predictions_and_factor(
            self.optimal_features,
            self.final_params
        )

        # Step 6: Train Final Model
        print("\n--- [Step 6] Training Final Model ---")
        self.final_model, final_rmse = self.train_xgboost_model(
            self.optimal_features,
            self.final_params
        )
        print(f"🎉 Final Model CV RMSE (uncorrected): {final_rmse:.4f}")
        
        # Step 7: Generate Submission
        print("\n--- [Step 7] Generating Submission ---")
        self.generate_submission(
            self.final_model,
            self.optimal_features,
            model_name="XGBoost_Optimized_Corrected_Efficient_Ranked"
        )
        
        end_pipeline_time = time.time()
        print(f"\n--- Pipeline finished in {end_pipeline_time - start_pipeline_time:.2f} seconds ---")


# --- 4. EXECUTION ---

if __name__ == '__main__':
    
    # Pack all configuration into a dictionary
    pipeline_config = {
        'S3_BUCKET_PATH': S3_BUCKET_PATH,
        'TRAIN_FILE': TRAIN_FILE,
        'TEST_FILE': TEST_FILE,
        'SUBMISSION_FILE': SUBMISSION_FILE,
        'TARGET_VARIABLE': TARGET_VARIABLE,
        'ID_COLS': ID_COLS,
        'RANDOM_STATE': RANDOM_STATE,
        'N_FOLDS': N_FOLDS,
        'N_OPTUNA_TRIALS': N_OPTUNA_TRIALS,
        'BASE_XGB_PARAMS': BASE_XGB_PARAMS
    }
    
    try:
        # Initialize the pipeline
        pipeline = SalesPredictor(config=pipeline_config)
        
        # Run the pipeline
        pipeline.run(
            run_sfs=False,  # <-- SET TO TRUE TO RUN SFS
            base_features=MANUAL_BASE_FEATURES,
            engineered_features=MANUAL_ENGINEERED_FEATURES
        )
        
    except Exception as e:
        print(f"An error occurred during pipeline execution: {e}")
        raise # Re-raise the exception to see the traceback