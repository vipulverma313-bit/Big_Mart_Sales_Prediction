import numpy as np # linear algebra
import pandas as pd # data processing
import math
from matplotlib import pyplot as plt
import seaborn as sns

# Optuna/Scipy for hyperparameter search
import optuna

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error , r2_score
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsClassifier

# Ignore warnings ;)
import warnings
warnings.simplefilter("ignore")

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
except Exception:
    print("Cannot access S3 path. Using dummy data for execution.")
  
# Store the target variable and identifiers
target = train['Item_Outlet_Sales']
train_len = train.shape[0]

# --- K-FOLD TARGET ENCODING FUNCTION (No changes needed) ---

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
            
            # Calculate mean of target for the feature in the training fold
            encoding_map = y_train_fold.groupby(X_train_fold[col]).mean()
            
            # Map the calculated mean to the validation fold
            # Use fillna(global_mean) as a safe fallback for unseen values in the fold
            X_encoded.loc[X_encoded.index[val_index], col_encoded_name] = X_encoded.iloc[val_index][col].map(encoding_map).fillna(global_mean)

        # Fallback for any remaining NaNs (use global mean)
        X_encoded[col_encoded_name].fillna(global_mean, inplace=True)

        
        # 2. Global Encoding for Test Set
        # Calculate the final encoding map using the entire training set (X, target)
        global_encoding_map = target.groupby(X[col]).mean()
        
        # Map the global encoding to the test set
        X_test_encoded[col_encoded_name] = X_test_encoded[col].map(global_encoding_map)
        
        # Fallback for categories in test not present in train (global mean)
        X_test_encoded[col_encoded_name].fillna(global_mean, inplace=True)
        
    return X_encoded, X_test_encoded

# --- Feature Engineering and Imputation Functions ---

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
    # Create a mask for zero visibility
    zero_vis_mask = df['Item_Visibility'] == 0
    # Calculate Item_Identifier mean for non-zero visibility
    id_avg = df.loc[~zero_vis_mask].groupby('Item_Identifier')['Item_Visibility'].transform('mean')
    
    # Replace 0s with the calculated mean
    df.loc[zero_vis_mask, 'Item_Visibility'] = id_avg.loc[zero_vis_mask].fillna(df['Item_Visibility'].mean())
    
    # ORIGINAL FEATURE: Visibility Ratio (relative to outlet)
    df['average_visibility'] = df.groupby('Outlet_Identifier')['Item_Visibility'].transform('mean')
    df['Visibility_Ratio'] = df['Item_Visibility'] / df['average_visibility']
    
    df = df.drop(['Outlet_Establishment_Year', 'Item_Type'], axis=1, errors='ignore')
    
    return df

def impute_missing_values(df):
    """
    Imputes missing values using Item_Identifier-specific mean for Item_Weight 
    and grouped mode for Outlet_Size (based on Outlet_Location_Type and Outlet_Type).
    """
    
    # 1. Item_Weight imputation: Use mean weight for the same Item_Identifier
    df['Item_Weight'] = df.groupby('Item_Identifier')['Item_Weight'].transform(lambda x: x.fillna(x.mean()))
    # Fallback for any remaining NaNs (global mean)
    df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)
    
    # 2. Outlet_Size imputation: Use mode of that combination for Outlet_Location_Type and Outlet_Type
    
    # Calculate the mode of 'Outlet_Size' for each combination of the two grouping columns
    mode_size_by_group = df.groupby(['Outlet_Location_Type', 'Outlet_Type'])['Outlet_Size'].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan)
    )
    
    # Apply the transformed values to the original column
    df['Outlet_Size'] = mode_size_by_group
    
    # Fallback for any remaining NaNs (e.g., if an entire group was missing 'Outlet_Size' or a combination only appears in test data)
    # This uses the global mode as a final safety step.
    df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0], inplace=True)
    
    return df


# --- COMBINE, PREPROCESS, SPLIT FOR TARGET ENCODING ---

# Save Identifiers before merging/one-hot encoding
submission_identifiers = test[['Item_Identifier', 'Outlet_Identifier']].copy()

# Drop the target and combine train and test sets for general feature engineering/imputation
train_for_fe = train.drop('Item_Outlet_Sales', axis=1, errors='ignore')
combined_data = pd.concat([train_for_fe, test], ignore_index=True)
train_len_initial = train_for_fe.shape[0]

# 1. Apply Feature Engineering
combined_data = preprocess_data(combined_data.copy())

# 2. Apply Improved Imputation
combined_data = impute_missing_values(combined_data.copy())

# -----------------------------------------------------------
# NEW SECTION: Additional Feature Engineering (Leak-Safe)
# -----------------------------------------------------------

# Split the data back temporarily to perform leak-safe feature engineering
X_temp = combined_data.iloc[:train_len_initial].copy()
X_test_temp = combined_data.iloc[train_len_initial:].copy()

# 1. NEW FEATURE: MRP_Quartile (Calculated ONLY on Training Data) 💡
# This is a categorical feature based on the MRP distribution in the training set
mrp_bins = X_temp['Item_MRP'].quantile([0.25, 0.5, 0.75]).tolist()
mrp_bins = [-np.inf] + mrp_bins + [np.inf]
mrp_labels = ['Q1', 'Q2', 'Q3', 'Q4']

X_temp['MRP_Quartile'] = pd.cut(X_temp['Item_MRP'], bins=mrp_bins, labels=mrp_labels, include_lowest=True)
X_test_temp['MRP_Quartile'] = pd.cut(X_test_temp['Item_MRP'], bins=mrp_bins, labels=mrp_labels, include_lowest=True)
# Fill NaNs (if any, typically due to edge cases) with the most frequent category
X_test_temp['MRP_Quartile'].fillna(X_temp['MRP_Quartile'].mode()[0], inplace=True)


# 2. NEW FEATURE: Item_Visibility_Mean_Ratio (Relative to Global Mean in Training Data) 💡
# Calculate the global mean Item_Visibility from the training data
global_vis_mean = X_temp['Item_Visibility'].mean()

X_temp['Item_Visibility_Mean_Ratio'] = X_temp['Item_Visibility'] / global_vis_mean
X_test_temp['Item_Visibility_Mean_Ratio'] = X_test_temp['Item_Visibility'] / global_vis_mean

# Fallback for division by zero (if global_vis_mean somehow became 0)
X_temp['Item_Visibility_Mean_Ratio'].replace([np.inf, -np.inf], 0, inplace=True)
X_test_temp['Item_Visibility_Mean_Ratio'].replace([np.inf, -np.inf], 0, inplace=True)
X_temp['Item_Visibility_Mean_Ratio'].fillna(0, inplace=True)
X_test_temp['Item_Visibility_Mean_Ratio'].fillna(0, inplace=True)


# 3. Combine data back for the rest of the flow
combined_data = pd.concat([X_temp, X_test_temp], ignore_index=True)
# -----------------------------------------------------------
# END NEW SECTION
# -----------------------------------------------------------


# -----------------------------
# EXISTING: Additional Feature Engineering (NONLINEAR / INTERACTION)
# Note: Item_Price_Per_Weight already included here and is equivalent to MRP_Per_Unit_Weight
# -----------------------------
new_feature_names = [
    'Item_Price_Per_Weight', 
    'MRP_to_Visibility',
    'Weight_MRP_Interaction',
    'Size_Visibility_Interaction'
]

# Step 1: Numerically encode Outlet_Size temporarily for interaction feature calculation
# We'll re-encode it later in the main flow (step 5)
size_mapping = {'Small': 1, 'Medium': 2, 'High': 3}
combined_data['Outlet_Size_Num'] = combined_data['Outlet_Size'].map(size_mapping)


# 1. EXISTING FEATURE: Item Price Per Unit Weight (MRP_Per_Unit_Weight)
# small epsilon to avoid division by zero
combined_data['Item_Price_Per_Weight'] = combined_data['Item_MRP'] / (combined_data['Item_Weight'] + 1e-5)

# 2. EXISTING FEATURE: Store Size x Item Visibility Interaction
combined_data['Size_Visibility_Interaction'] = combined_data['Outlet_Size_Num'] * combined_data['Item_Visibility']
combined_data.drop('Outlet_Size_Num', axis=1, inplace=True) # Drop temporary column

# 3. Original Features (Kept)
# small epsilon to avoid division by zero
combined_data['MRP_to_Visibility'] = combined_data['Item_MRP'] / (combined_data['Item_Visibility'] + 1e-5)
combined_data['Weight_MRP_Interaction'] = combined_data['Item_Weight'] * combined_data['Item_MRP']


# Replace any +inf/-inf that might arise and safely fill any NaNs for these new features
combined_data.replace([np.inf, -np.inf], np.nan, inplace=True)
for nf in new_feature_names:
    if nf in combined_data.columns:
        combined_data[nf].fillna(combined_data[nf].mean(), inplace=True)
# -----------------------------

# 3. Split data back for Target Encoding (X is the train part, X_test_final is the test part)
X = combined_data.iloc[:train_len_initial].copy()
X_test_final = combined_data.iloc[train_len_initial:].copy()


# 4. APPLY K-FOLD TARGET ENCODING (LEAKAGE-FREE FEATURES)
target_encoding_cols = ['Item_Identifier', 'Outlet_Identifier']
X, X_test_final = kfold_target_encoding(X, X_test_final, target, target_encoding_cols, n_splits=5)


# --- CONTINUE WITH ENCODING & SCALING (Final Steps) ---

# 5. Encoding Categorical Variables on Split Data (for proper fitting/transforming)

# Custom Label Encoding for ordinal features based on size assumption
X['Outlet_Location_Type'] = X['Outlet_Location_Type'].map({'Tier 1': 3, 'Tier 2': 2, 'Tier 3': 1}).astype(int)
X_test_final['Outlet_Location_Type'] = X_test_final['Outlet_Location_Type'].map({'Tier 1': 3, 'Tier 2': 2, 'Tier 3': 1}).astype(int)

# Use the saved mapping for Outlet_Size
X['Outlet_Size'] = X['Outlet_Size'].map(size_mapping).astype(int)
X_test_final['Outlet_Size'] = X_test_final['Outlet_Size'].map(size_mapping).astype(int)

# Label Encoder for other ordinal-like features
encoder = LabelEncoder()
# Item_Type_Combined is excluded here for OHE later
ordinal_features = ['Item_Fat_Content', 'Outlet_Type']
for feature in ordinal_features:
    X[feature] = encoder.fit_transform(X[feature])
    X_test_final[feature] = encoder.transform(X_test_final[feature])
    
# Re-Combine for One Hot Encoding to ensure consistent columns
combined_data_ohe = pd.concat([X, X_test_final], ignore_index=True)

# One Hot Encoding for nominal features
combined_data_ohe = pd.get_dummies(combined_data_ohe, columns=['Outlet_Identifier'], drop_first=True)
combined_data_ohe = pd.get_dummies(combined_data_ohe, columns=['Item_Type_Combined'], prefix='Item_Type_Comb', drop_first=False) 
# OHE for NEW FEATURE: MRP_Quartile
combined_data_ohe = pd.get_dummies(combined_data_ohe, columns=['MRP_Quartile'], prefix='MRP_Q', drop_first=False) 


# Drop Item_Identifier (after target encoding, it is no longer needed)
combined_data_ohe.drop(labels=['Item_Identifier'], axis=1, inplace=True)


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
# UPDATED CONTINUOUS FEATURES: Added Item_Visibility_Mean_Ratio
continuous_features = [
    'Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Years', 'average_visibility', 'Visibility_Ratio', 
    'Item_Identifier_Avg_Sales', 'Outlet_Identifier_Avg_Sales',
    # Nonlinear / interaction features
    'Item_Price_Per_Weight', 'MRP_to_Visibility', 'Weight_MRP_Interaction', 'Size_Visibility_Interaction',
    # New feature
    'Item_Visibility_Mean_Ratio'
]

# Some safety: if any of the listed continuous features don't exist (rare), filter them
continuous_features = [c for c in continuous_features if c in X.columns]

# Fit scaler on training data and transform both train and test
X[continuous_features] = scaler.fit_transform(X[continuous_features])
X_test_final[continuous_features] = scaler.transform(X_test_final[continuous_features])


# Splitting training data for final model evaluation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# -----------------------------------------------------------
# 8) Model Tuning with GridSearchCV (No changes needed)
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
        # Attempt to compute approximate total fits for the simple informative print
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

# Lasso/Ridge Pipeline with Poly Features 
linear_pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)), 
    ('model', None) # Placeholder for Lasso/Ridge
])

# Expanded Lasso Parameters (Discrete values)
lasso_params = {
    'model': [Lasso(fit_intercept=True, max_iter=5000)],
    'model__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0], 
    'poly__degree': [1, 2, 3] 
}

# Expanded Ridge Parameters (Discrete values)
ridge_params = {
    'model': [Ridge(fit_intercept=True, max_iter=5000)],
    'model__alpha': [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0], 
    'poly__degree': [1, 2, 3]
}

# XGBoost Model 
xgb_model = XGBRegressor(random_state=42, tree_method='hist', n_jobs=-1, eval_metric='rmse', verbosity=0)
# XGBoost Parameters (Discrete values)
xgb_params = {
    'n_estimators': [200, 500, 800, 1000], 
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.03, 0.01, 0.1],
    'reg_alpha': [0, 1, 5],
    'reg_lambda': [0, 1, 5],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

# 14b) Tune Models 
print("\n--- Starting Model Tuning with GridSearchCV ---")

print("Tuning Lasso...")
best_lasso_model, best_lasso_params, best_lasso_rmse = get_best_model(linear_pipe, lasso_params, X, y)

print("Tuning Ridge...")
best_ridge_model, best_ridge_params, best_ridge_rmse = get_best_model(linear_pipe, ridge_params, X, y)

print("Tuning XGBoost...")
best_xgb_model, best_xgb_params, best_xgb_rmse = get_best_model(xgb_model, xgb_params, X, y)

# -----------------------------------------------------------
# 9) Determine and Train the Best Model (No changes needed)
# -----------------------------------------------------------

model_results = {
    'Lasso': {'RMSE': best_lasso_rmse, 'Params': best_lasso_params},
    'Ridge': {'RMSE': best_ridge_rmse, 'Params': best_ridge_params},
    'XGBoost': {'RMSE': best_xgb_rmse, 'Params': best_xgb_params}
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

if best_model_name == 'XGBoost':
    # Clean up parameters to remove the list wrapper [X]
    final_model_params = {k: v[0] if isinstance(v, list) else v for k, v in best_model_params.items()}
    final_model = XGBRegressor(random_state=42, tree_method='hist', n_jobs=-1, eval_metric='rmse', verbosity=0, **final_model_params)
    final_model.fit(X, y)
    
elif best_model_name == 'Lasso':
    final_model = linear_pipe.set_params(poly__degree=best_model_params['poly__degree'], 
                                         model=Lasso(alpha=best_model_params['model__alpha'], fit_intercept=True, max_iter=5000))
    final_model.fit(X, y)

elif best_model_name == 'Ridge':
    final_model = linear_pipe.set_params(poly__degree=best_model_params['poly__degree'], 
                                         model=Ridge(alpha=best_model_params['model__alpha'], fit_intercept=True, max_iter=5000))
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