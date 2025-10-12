import numpy as np # linear algebra
import pandas as pd # data processing
import math
# The imports below were not strictly necessary for the ML pipeline execution but kept for completeness
# from matplotlib import pyplot as plt
# import seaborn as sns 

# Optuna/Scipy for hyperparameter search
import optuna

from sklearn.model_selection import GridSearchCV, KFold, GroupKFold # Added GroupKFold for future use
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor # ADDED: LightGBM Regressor
from sklearn.ensemble import RandomForestRegressor # ADDED: Random Forest Regressor

# Ignore warnings ;)
import warnings
warnings.simplefilter("ignore")

# --- INITIAL SETUP ---
S3_BUCKET_PATH = 's3://useast1-nlsn-dscihdamrpd-zoo-archive/users/vipulverma/Prediction_Tests_CSVs/'
TRAIN_FILE = 'train_v9rqX0R.csv'
TEST_FILE = 'test_AbJTz2l.csv'

# set seed for reproductibility
np.random.seed(0)

# Define CURRENT_YEAR for feature engineering
CURRENT_YEAR = 2013

train_s3_url = S3_BUCKET_PATH + TRAIN_FILE
test_s3_url = S3_BUCKET_PATH + TEST_FILE

print(f"Loading training data from: {train_s3_url}")
# Mock data loading since I cannot access S3, replace this with your actual s3 loading
try:
    train = pd.read_csv(train_s3_url)
    test = pd.read_csv(test_s3_url)
    print("✅ Data loaded successfully from S3.")
except Exception as e:
    print(f"❌ Cannot access S3 path. Using dummy data for execution. Error: {e}")
    # --- START DUMMY DATA CREATION ---
    data = {
        'Item_Identifier': [f'FD{i:03d}' for i in range(100)] * 10,
        'Item_Weight': np.random.rand(1000) * 10 + 5,
        'Item_Fat_Content': np.random.choice(['Low Fat', 'Regular', 'LF', 'reg', 'low fat'], 1000),
        'Item_Visibility': np.random.rand(1000) * 0.1,
        'Item_Type': np.random.choice([f'Type_{i}' for i in range(16)], 1000),
        'Item_MRP': np.random.rand(1000) * 150 + 50,
        'Outlet_Identifier': [f'OUT0{i}' for i in range(10)] * 100,
        'Outlet_Establishment_Year': np.random.choice([1985, 1987, 1997, 1999, 2002, 2004, 2007, 2009], 1000),
        'Outlet_Size': np.random.choice(['Small', 'Medium', 'High', np.nan], 1000),
        'Outlet_Location_Type': np.random.choice(['Tier 1', 'Tier 2', 'Tier 3'], 1000),
        'Outlet_Type': np.random.choice(['Supermarket Type1', 'Supermarket Type2', 'Grocery Store'], 1000),
        'Item_Outlet_Sales': np.random.rand(1000) * 4000
    }
    df_temp = pd.DataFrame(data)
    train = df_temp.sample(frac=0.8, random_state=0).reset_index(drop=True)
    test = df_temp.drop(train.index).reset_index(drop=True)
    train['Item_Weight'].iloc[0:5] = np.nan
    train['Outlet_Size'].iloc[5:10] = np.nan
    test['Outlet_Size'].iloc[10:15] = np.nan
    print("✅ Dummy data created and loaded successfully.")
    # --- END DUMMY DATA CREATION ---

# Store the target variable and identifiers
target = train['Item_Outlet_Sales']
train_len = train.shape[0]

# --- K-FOLD TARGET ENCODING FUNCTION (Kept as is) ---
def kfold_target_encoding(X, X_test, target, columns, n_splits=5):
    """
    Performs K-Fold Target Encoding (Mean Encoding) to prevent data leakage 
    on the training set and uses global mean for the test set.
    """
    X_encoded = X.copy()
    X_test_encoded = X_test.copy()
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    global_mean = target.mean()
    
    print(f"Applying K-Fold Target Encoding for: {columns}")
    
    for col in columns:
        col_encoded_name = f'{col}_Avg_Sales'
        X_encoded[col_encoded_name] = np.nan
        
        # 1. K-Fold Encoding for Training Set
        for train_index, val_index in kf.split(X_encoded):
            X_train_fold = X_encoded.iloc[train_index]
            y_train_fold = target.iloc[train_index]
            
            encoding_map = y_train_fold.groupby(X_train_fold[col]).mean()
            
            X_encoded.loc[X_encoded.index[val_index], col_encoded_name] = X_encoded.iloc[val_index][col].map(encoding_map).fillna(global_mean)

        X_encoded[col_encoded_name].fillna(global_mean, inplace=True)
        
        # 2. Global Encoding for Test Set
        global_encoding_map = target.groupby(X[col]).mean()
        X_test_encoded[col_encoded_name] = X_test_encoded[col].map(global_encoding_map)
        X_test_encoded[col_encoded_name].fillna(global_mean, inplace=True)
            
    return X_encoded, X_test_encoded

# --- Out-of-fold mean encoding helper (Kept as is) ---
def add_oof_mean_encoding(df, feature, target_col='Item_Outlet_Sales', n_splits=5):
    """Performs Out-of-Fold mean encoding on the combined train/test set."""
    mean_enc_col = f'{feature}_mean_sales_enc' 
    df[mean_enc_col] = np.nan
    
    # Use KFold only on the training indices
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_idx = df[df['is_train']].index
    
    if len(train_idx) == 0:
        return df

    # 1. OOF Encoding for Training Set
    for trn_idx, val_idx in kf.split(train_idx):
        trn_ids, val_ids = train_idx[trn_idx], train_idx[val_idx]
        # Calculate mean using the training fold
        means = df.loc[trn_ids].groupby(feature)[target_col].mean()
        # Map mean to the validation fold
        df.loc[val_ids, mean_enc_col] = df.loc[val_ids, feature].map(means)
        
    # 2. Global Encoding for Test Set
    # Calculate global mean from full train set
    global_means = df[df['is_train']].groupby(feature)[target_col].mean()
    # Map global mean to test set
    df.loc[~df['is_train'], mean_enc_col] = df.loc[~df['is_train'], feature].map(global_means)
    
    # Fill remaining NaNs (e.g., categories only in test/not in train) with global target mean
    df[mean_enc_col].fillna(df[target_col].mean(), inplace=True) 
    
    return df
# --------------------------------------------------------------------------------


# --- Feature Engineering and Imputation Functions (Kept as is) ---

def preprocess_data(df):
    """Applies general feature engineering steps."""
    
    # 1. Standardise Item_Fat_Content
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'})

    # 2. Outlet_Years and Item_Type_Combined
    df['Outlet_Years'] = CURRENT_YEAR - df['Outlet_Establishment_Year']
    
    def get_broad_category(x):
        return 'Food' if x[0:2] == 'FD' else ('Drinks' if x[0:2] == 'DR' else 'Non-Consumable')
        
    df['Item_Type_Combined'] = df['Item_Identifier'].apply(get_broad_category)
    
    # Consistency Fix: Non-Consumable items must be Non-Edible fat content
    df.loc[df['Item_Type_Combined'] == "Non-Consumable", 'Item_Fat_Content'] = "Non-Edible"
    
    # 3. Fix Zero Item_Visibility (using Item_Identifier mean)
    zero_vis_mask = df['Item_Visibility'] == 0
    id_avg = df.loc[~zero_vis_mask].groupby('Item_Identifier')['Item_Visibility'].transform('mean')
    
    df.loc[zero_vis_mask, 'Item_Visibility'] = id_avg.loc[zero_vis_mask].fillna(df['Item_Visibility'].mean())
    
    # NEW FEATURE: Visibility Ratio (relative to outlet)
    df['average_visibility'] = df.groupby('Outlet_Identifier')['Item_Visibility'].transform('mean')
    df['Visibility_Ratio'] = df['Item_Visibility'] / (df['average_visibility'] + 1e-5) # Added epsilon
    
    df = df.drop(['Outlet_Establishment_Year', 'Item_Type'], axis=1, errors='ignore')
    
    return df

def impute_missing_values(df):
    """
    Imputes missing values using Item_Identifier-specific mean for Item_Weight 
    and improved, consistent logic for Outlet_Size.
    """
    
    # 1. Item_Weight imputation: (REMAINS UNCHANGED)
    df['Item_Weight'] = df.groupby('Item_Identifier')['Item_Weight'].transform(lambda x: x.fillna(x.mean()))
    df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)
    
    # 2. Outlet_Size imputation (MODIFIED FOR CONSISTENCY)
    
    # A. Impute using the mode of the same Outlet_Identifier (Consistency Rule)
    df['Outlet_Size'] = df.groupby('Outlet_Identifier')['Outlet_Size'].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan)
    )
    
    # B. Fallback Imputation for Outlets Still Missing Size 
    mode_size_by_group = df.groupby(['Outlet_Location_Type', 'Outlet_Type'])['Outlet_Size'].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan)
    )
    
    # Only fill where 'Outlet_Size' is still NaN
    missing_mask = df['Outlet_Size'].isnull()
    df.loc[missing_mask, 'Outlet_Size'] = mode_size_by_group.loc[missing_mask]

    # C. Global mode fallback (REMAINS UNCHANGED)
    global_mode = df['Outlet_Size'].mode()
    if not global_mode.empty:
        df['Outlet_Size'].fillna(global_mode[0], inplace=True)
    else:
        # Safely handle case where Outlet_Size is entirely NaN (e.g., in dummy data)
        df['Outlet_Size'].fillna('Unknown', inplace=True)
            
    return df


# --- COMBINE, PREPROCESS, EXTRACT TARGET FOR OOF ENCODING (Kept as is) ---

submission_identifiers = test[['Item_Identifier', 'Outlet_Identifier']].copy()

train_for_fe = train.drop('Item_Outlet_Sales', axis=1, errors='ignore')
combined_data = pd.concat([train_for_fe, test], ignore_index=True)
train_len_initial = train_for_fe.shape[0]

# 1. Apply Feature Engineering
combined_data = preprocess_data(combined_data.copy())

# 2. Apply Improved Imputation
combined_data = impute_missing_values(combined_data.copy())

# -----------------------------
# Additional Feature Engineering (NONLINEAR / INTERACTION) (Kept as is)
# -----------------------------
new_feature_names = [
    'Item_Price_Per_Weight', 
    'MRP_to_Visibility',
    'Weight_MRP_Interaction',
    'Size_Visibility_Interaction'
]

size_mapping = {'Small': 1, 'Medium': 2, 'High': 3, 'Unknown': 2} # Added Unknown for safety
combined_data['Outlet_Size_Num'] = combined_data['Outlet_Size'].map(size_mapping).fillna(2) # Fillna with Medium (2)

combined_data['Item_Price_Per_Weight'] = combined_data['Item_MRP'] / (combined_data['Item_Weight'] + 1e-5)
combined_data['Size_Visibility_Interaction'] = combined_data['Outlet_Size_Num'] * combined_data['Item_Visibility']
combined_data.drop('Outlet_Size_Num', axis=1, inplace=True) 

combined_data['MRP_to_Visibility'] = combined_data['Item_MRP'] / (combined_data['Item_Visibility'] + 1e-5)
combined_data['Weight_MRP_Interaction'] = combined_data['Item_Weight'] * combined_data['Item_MRP']


combined_data.replace([np.inf, -np.inf], np.nan, inplace=True)
for nf in new_feature_names:
    if nf in combined_data.columns:
        combined_data[nf].fillna(combined_data[nf].mean(), inplace=True)
# -----------------------------


# =========================================================================
# 🚀 ADDED: OOF AGGREGATES AND RATIO FEATURES 🚀 (Kept as is)
# =========================================================================

# Create 'is_train' flag and target column for aggregate calculations
combined_data['is_train'] = combined_data.index < train_len_initial
combined_data['Item_Outlet_Sales'] = pd.Series(np.nan, index=combined_data.index)
combined_data.loc[combined_data['is_train'], 'Item_Outlet_Sales'] = target.values 

df = combined_data.copy()

# --- NON-TARGET AGGREGATES (SAFE) ---
item_non_sales_aggs = (
    df
    .groupby('Item_Identifier')
    .agg({
        'Item_MRP': 'mean',
        'Item_Weight': 'mean',
        'Item_Visibility': 'mean'
    })
)
item_non_sales_aggs.columns = ['Item_MRP_Mean', 'Item_Weight_Mean', 'Item_Visibility_Mean']
df = df.merge(item_non_sales_aggs, on='Item_Identifier', how='left')

outlet_non_sales_aggs = (
    df
    .groupby('Outlet_Identifier')
    .agg({
        'Item_MRP': 'mean',
        'Item_Visibility': 'mean'
    })
)
outlet_non_sales_aggs.columns = ['Outlet_MRP_Mean', 'Outlet_Visibility_Mean']
df = df.merge(outlet_non_sales_aggs, on='Outlet_Identifier', how='left')


# --- SALES-BASED AGGREGATES / OOF ENCODINGS (LEAKAGE-FREE) ---
print("Applying OOF Target Encodings...")

df = add_oof_mean_encoding(df, 'Item_Identifier') 
df = add_oof_mean_encoding(df, 'Outlet_Identifier') 

df['Item_Outlet_Cross_ID'] = df['Item_Identifier'].astype(str) + '_' + df['Outlet_Identifier'].astype(str)
df = add_oof_mean_encoding(df, 'Item_Outlet_Cross_ID') 
df.drop('Item_Outlet_Cross_ID', axis=1, inplace=True)

# --- RELATIVE / RATIO FEATURES (Using safe, non-target-derived means) ---
df['Item_MRP_to_Avg_ItemMRP'] = df['Item_MRP'] / (df['Item_MRP_Mean'] + 1e-5)
df['Item_Weight_to_Avg_ItemWeight'] = df['Item_Weight'] / (df['Item_Weight_Mean'] + 1e-5)
df['Outlet_MRP_to_Avg_OutletMRP'] = df['Item_MRP'] / (df['Outlet_MRP_Mean'] + 1e-5)

# --- OOF MEAN ENCODINGS (Categorical Features) ---
df = add_oof_mean_encoding(df, 'Item_Type_Combined') 
df = add_oof_mean_encoding(df, 'Outlet_Type') 
df = add_oof_mean_encoding(df, 'Outlet_Location_Type') 


# Final cleanup of new aggregates/ratios (Impute remaining NaNs with mean)
new_agg_cols = [col for col in df.columns if any(s in col for s in ['_Mean', '_enc', 'Ratio', 'MRP_to'])]
for col in new_agg_cols:
    if df[col].dtype in [np.float64, np.float32]:
        df[col].fillna(df[col].mean(), inplace=True)

# Drop the temporary target and training flag
combined_data = df.drop(columns=['is_train', 'Item_Outlet_Sales'])
print("✅ OOF Aggregates and Ratios added.")
# =========================================================================


# 3. Split data back for Target Encoding
X = combined_data.iloc[:train_len_initial].copy()
X_test_final = combined_data.iloc[train_len_initial:].copy()

# 4. APPLY K-FOLD TARGET ENCODING (LEAKAGE-FREE FEATURES)
# NOTE: This step will overwrite Item/Outlet OOF encodings generated above.
target_encoding_cols = ['Item_Identifier', 'Outlet_Identifier']
X, X_test_final = kfold_target_encoding(X, X_test_final, target, target_encoding_cols, n_splits=5)

# Drop the now-redundant simple OOF encodings created earlier to avoid confusion/collinearity
cols_to_drop_oof = ['Item_Identifier_mean_sales_enc', 'Outlet_Identifier_mean_sales_enc']
for c in cols_to_drop_oof:
    if c in X.columns:
        X.drop(c, axis=1, inplace=True)
    if c in X_test_final.columns:
        X_test_final.drop(c, axis=1, inplace=True)


# --- CONTINUE WITH ENCODING & SCALING (Final Steps) (Kept as is) ---

# 5. Encoding Categorical Variables on Split Data (for proper fitting/transforming)

# Custom Label Encoding for ordinal features based on size assumption
tier_mapping = {'Tier 1': 3, 'Tier 2': 2, 'Tier 3': 1}
X['Outlet_Location_Type'] = X['Outlet_Location_Type'].map(tier_mapping).astype(int)
X_test_final['Outlet_Location_Type'] = X_test_final['Outlet_Location_Type'].map(tier_mapping).astype(int)

# Use the saved mapping for Outlet_Size
# Use fillna(2) for 'Unknown' which was handled in the interaction feature creation.
X['Outlet_Size'] = X['Outlet_Size'].map(size_mapping).fillna(2).astype(int)
X_test_final['Outlet_Size'] = X_test_final['Outlet_Size'].map(size_mapping).fillna(2).astype(int)

# Label Encoder for other ordinal-like features
encoder = LabelEncoder()
# Item_Type_Combined and Outlet_Type will be dropped as their OOF versions are kept
ordinal_features = ['Item_Fat_Content'] 
for feature in ordinal_features:
    X[feature] = encoder.fit_transform(X[feature])
    X_test_final[feature] = encoder.transform(X_test_final[feature])
    
# Re-Combine for processing
combined_data_ohe = pd.concat([X, X_test_final], ignore_index=True)

# Drop raw categorical features and OHE versions (rely on Target/OOF/LE)
cols_to_drop = [
    'Item_Identifier', # Target Encoded
    'Outlet_Identifier', # Target Encoded
    'Item_Type_Combined', # OOF Encoded
    'Outlet_Type', # OOF Encoded
]

combined_data_ohe.drop(labels=[c for c in cols_to_drop if c in combined_data_ohe.columns], axis=1, inplace=True)


# 6. Split data back for final scaling and modeling
X = combined_data_ohe.iloc[:train_len_initial].copy()
y = target.copy()
X_test_final = combined_data_ohe.iloc[train_len_initial:].copy()

# Final safety check: impute any possible leftover NaNs with the mean
for col in X.columns:
    if X[col].isnull().any():
        mean_val = X[col].mean()
        X[col].fillna(mean_val, inplace=True)
        X_test_final[col].fillna(mean_val, inplace=True)


# 7. Normalization (Standard Scaling) of Continuous Features
scaler = StandardScaler()

# UPDATED: Added all new aggregate and ratio features
continuous_features = [
    'Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Years', 'average_visibility', 'Visibility_Ratio', 
    'Item_Identifier_Avg_Sales', 'Outlet_Identifier_Avg_Sales',
    # Newly added nonlinear / interaction features
    'Item_Price_Per_Weight', 'MRP_to_Visibility', 'Weight_MRP_Interaction', 'Size_Visibility_Interaction',
    # NEW AGGREGATES AND OOF ENCODINGS
    'Item_MRP_Mean', 'Item_Weight_Mean', 'Item_Visibility_Mean', 'Outlet_MRP_Mean', 'Outlet_Visibility_Mean',
    'Item_Outlet_Cross_ID_mean_sales_enc', 
    'Item_Type_Combined_mean_sales_enc', 'Outlet_Type_mean_sales_enc', 'Outlet_Location_Type_mean_sales_enc',
    'Item_MRP_to_Avg_ItemMRP', 'Item_Weight_to_Avg_ItemWeight', 'Outlet_MRP_to_Avg_OutletMRP'
]

continuous_features = [c for c in continuous_features if c in X.columns]

X[continuous_features] = scaler.fit_transform(X[continuous_features])
X_test_final[continuous_features] = scaler.transform(X_test_final[continuous_features])


# Splitting training data for final model evaluation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# -----------------------------------------------------------
## 8) Model Tuning with GridSearchCV
# -----------------------------------------------------------

# Linear models must use the pipeline, as they rely on PolynomialFeatures
def get_best_model(model, params, X, y, cv_folds=5, random_state=0):
    """Performs hyperparameter tuning using GridSearchCV"""
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    search = GridSearchCV(
        estimator=model,
        param_grid=params, 
        scoring='neg_mean_squared_error',
        cv=kf,
        verbose=0,
        n_jobs=-1
    )
    
    try:
        num_fits = 1
        if isinstance(params, dict):
            lens = [len(v) for v in params.values() if hasattr(v, '__len__')]
            num_fits = max(1, math.prod(lens))
        print(f"Starting GridSearchCV with approx {num_fits * cv_folds} total fits...")
    except Exception:
        print("Starting GridSearchCV...")
    
    search.fit(X, y) 
    
    best_rmse = np.sqrt(-search.best_score_)
    
    return search.best_estimator_, search.best_params_, best_rmse

# Define Models and Parameter Grids (Now using discrete values for Grid Search)

# Lasso Pipeline with Poly Features 
linear_pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)), 
    ('model', None) # Placeholder for Lasso
])

# Expanded Lasso Parameters (Discrete values) - KEPT
lasso_params = {
    'model': [Lasso(fit_intercept=True, max_iter=5000)],
    'model__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0], 
    'poly__degree': [1, 2] 
}

# XGBoost Model - KEPT
xgb_model = XGBRegressor(random_state=42, tree_method='hist', n_jobs=-1, eval_metric='rmse', verbosity=0)
# XGBoost Parameters (Discrete values)
xgb_params = {
    'n_estimators': [500, 800], # Reduced search space for speed
    'max_depth': [3, 6, 8], # Reduced search space for speed
    'learning_rate': [0.03, 0.1], # Reduced search space for speed
    'reg_alpha': [1, 5],
    'reg_lambda': [1, 5],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

# ADDED: Random Forest Model
rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
# Random Forest Parameters (Discrete values)
rf_params = {
    'n_estimators': [300, 500, 800],
    'max_depth': [5, 10, 20],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [3, 5]
}

# ADDED: LightGBM Model
lgbm_model = LGBMRegressor(random_state=42, n_jobs=-1, metric='rmse', verbosity=-1)
# LightGBM Parameters (Discrete values)
lgbm_params = {
    'n_estimators': [300, 500, 800],
    'max_depth': [4, 8, 12],
    'learning_rate': [0.03, 0.1],
    'reg_alpha': [0.1, 1],
    'reg_lambda': [0.1, 1]
}

# 14b) Tune Models 
print("\n--- Starting Model Tuning with GridSearchCV ---")

print("Tuning Lasso...")
best_lasso_model, best_lasso_params, best_lasso_rmse = get_best_model(linear_pipe, lasso_params, X, y)

print("Tuning XGBoost...")
best_xgb_model, best_xgb_params, best_xgb_rmse = get_best_model(xgb_model, xgb_params, X, y)

print("Tuning Random Forest...")
best_rf_model, best_rf_params, best_rf_rmse = get_best_model(rf_model, rf_params, X, y)

print("Tuning LightGBM...")
best_lgbm_model, best_lgbm_params, best_lgbm_rmse = get_best_model(lgbm_model, lgbm_params, X, y)


# -----------------------------------------------------------
## 9) Determine and Train the Best Model
# -----------------------------------------------------------

model_results = {
    'Lasso': {'RMSE': best_lasso_rmse, 'Params': best_lasso_params},
    'XGBoost': {'RMSE': best_xgb_rmse, 'Params': best_xgb_params},
    'RandomForest': {'RMSE': best_rf_rmse, 'Params': best_rf_params}, # ADDED
    'LightGBM': {'RMSE': best_lgbm_rmse, 'Params': best_lgbm_params}   # ADDED
}

best_model_name = min(model_results, key=lambda k: model_results[k]['RMSE'])
best_model_rmse = model_results[best_model_name]['RMSE']
best_model_params = model_results[best_model_name]['Params']

print("\n" + "="*50)
print("🏆 BEST MODEL FROM CROSS-VALIDATION 🏆")
print(f"Model: {best_model_name}")
print(f"Cross-Validation RMSE: {best_model_rmse:.4f}")
print("Parameters:", best_model_params)
print("="*50)

# Re-train the best model on the full training set (X)

final_model_params = {k: v[0] if isinstance(v, list) else v for k, v in best_model_params.items()}

if best_model_name == 'XGBoost':
    final_model = XGBRegressor(random_state=42, tree_method='hist', n_jobs=-1, eval_metric='rmse', verbosity=0, **final_model_params)
    final_model.fit(X, y)
    
elif best_model_name == 'LightGBM':
    final_model = LGBMRegressor(random_state=42, n_jobs=-1, metric='rmse', verbosity=-1, **final_model_params)
    final_model.fit(X, y)

elif best_model_name == 'RandomForest':
    final_model = RandomForestRegressor(random_state=42, n_jobs=-1, **final_model_params)
    final_model.fit(X, y)

elif best_model_name == 'Lasso':
    final_model = linear_pipe.set_params(poly__degree=best_model_params['poly__degree'], 
                                         model=Lasso(alpha=best_model_params['model__alpha'], fit_intercept=True, max_iter=5000))
    final_model.fit(X, y)


# 10) Final Predictions On The Test Dataset
final_test_preds = final_model.predict(X_test_final)

# 11) Final Evaluation on Validation Set (for comparison)
val_preds = final_model.predict(X_val)
rmse_val = np.sqrt(mean_squared_error(y_val, val_preds))
r2_val = r2_score(y_val, val_preds)

print(f"\nValidation Set RMSE (Best Model): {rmse_val:.4f}")
print(f"Validation Set R-squared (Best Model): {r2_val:.4f}")


# FINAL STEP: Create and Save the Submission DataFrame
submission = submission_identifiers.copy()
submission['Item_Outlet_Sales'] = final_test_preds

# Clean up negative predictions (sales must be non-negative)
submission['Item_Outlet_Sales'] = submission['Item_Outlet_Sales'].apply(lambda x: 0 if x < 0 else x)

# Display the first few rows
print("\n--- Final Submission DataFrame Head (Using Best Model) ---")
print(submission.head())