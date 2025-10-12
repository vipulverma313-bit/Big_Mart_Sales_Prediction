import numpy as np # linear algebra
import pandas as pd # data processing
import math
from matplotlib import pyplot as plt
import seaborn as sns

# Optuna/Scipy for hyperparameter search
import optuna
from scipy.stats import randint, uniform

from sklearn.model_selection import RandomizedSearchCV 
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error , r2_score
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsClassifier
import os
# Ignore warnings ;)
import warnings
warnings.simplefilter("ignore")

S3_BUCKET_PATH = 'C:/Users/VIVERMA2401/Downloads/'
TRAIN_FILE = 'train_v9rqX0R.csv'
TEST_FILE = 'test_AbJTz2l.csv'

# set seed for reproductibility
np.random.seed(0)

# Define CURRENT_YEAR for feature engineering
CURRENT_YEAR = 2013

train_s3_url = S3_BUCKET_PATH + TRAIN_FILE
test_s3_url = S3_BUCKET_PATH + TEST_FILE

print(f"Loading training data from: {train_s3_url}")
train = pd.read_csv(train_s3_url)

print(f"Loading test data from: {test_s3_url}")
test = pd.read_csv(test_s3_url)

# Store the target variable and identifiers
target = train['Item_Outlet_Sales']
train_len = train.shape[0]

# --- NEW: K-FOLD TARGET ENCODING FUNCTION (LEAKAGE PREVENTION) ---

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
    
    # NEW FEATURE: Visibility Ratio (relative to outlet)
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
    # .transform() applies the calculated mode back to the rows belonging to that group,
    # specifically filling the NaN values.
    mode_size_by_group = df.groupby(['Outlet_Location_Type', 'Outlet_Type'])['Outlet_Size'].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan)
    )
    
    # Apply the transformed values to the original column
    df['Outlet_Size'] = mode_size_by_group
    
    # Fallback for any remaining NaNs (e.g., if an entire group was missing 'Outlet_Size' or a combination only appears in test data)
    # This uses the global mode as a final safety step.
    df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0], inplace=True)
    
    # The KNN-related cleanup is no longer needed, as no temporary columns are created.
    
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

X['Outlet_Size'] = X['Outlet_Size'].map({'Small': 1, 'Medium': 2, 'High': 3}).astype(int)
X_test_final['Outlet_Size'] = X_test_final['Outlet_Size'].map({'Small': 1, 'Medium': 2, 'High': 3}).astype(int)

# Label Encoder for other ordinal-like features
encoder = LabelEncoder()
ordinal_features = ['Item_Fat_Content', 'Outlet_Type', 'Item_Type_Combined']
for feature in ordinal_features:
    X[feature] = encoder.fit_transform(X[feature])
    X_test_final[feature] = encoder.transform(X_test_final[feature])
    
# Re-Combine for One Hot Encoding to ensure consistent columns
combined_data_ohe = pd.concat([X, X_test_final], ignore_index=True)

# One Hot Encoding
combined_data_ohe = pd.get_dummies(combined_data_ohe, columns=['Outlet_Identifier'], drop_first=True)
combined_data_ohe = pd.get_dummies(combined_data_ohe, columns=['Item_Type_Combined'], prefix='Item_Type_Comb', drop_first=False) 

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
# ADDED THE NEW TARGET-ENCODED FEATURES
continuous_features = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Years', 'average_visibility', 'Visibility_Ratio', 
                       'Item_Identifier_Avg_Sales', 'Outlet_Identifier_Avg_Sales']

# Fit scaler on training data and transform both train and test
X[continuous_features] = scaler.fit_transform(X[continuous_features])
X_test_final[continuous_features] = scaler.transform(X_test_final[continuous_features])


# Splitting training data for final model evaluation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# -----------------------------------------------------------
# 8) Model Tuning with Optuna (via RandomizedSearchCV)
# -----------------------------------------------------------

# Linear models must use the pipeline, as they rely on PolynomialFeatures
def get_best_model(model, params, X, y, n_iter=200, cv_folds=5, random_state=0): # Reduced n_iter for faster execution
    """Performs hyperparameter tuning using RandomizedSearchCV"""
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=n_iter,
        scoring='neg_mean_squared_error',
        cv=kf,
        verbose=0,
        random_state=random_state,
        n_jobs=-1
    )
    
    search.fit(X, y) 
    
    best_rmse = np.sqrt(-search.best_score_)
    
    return search.best_estimator_, search.best_params_, best_rmse

# Define Models and Parameter Grids

# Lasso/Ridge Pipeline with Poly Features 
linear_pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)), 
    ('model', None) # Placeholder for Lasso/Ridge
])

# Lasso Parameters 
lasso_params = {
    'model': [Lasso(fit_intercept=True, max_iter=5000)], # Increased max_iter
    'model__alpha': uniform(0.01, 1.0),
    'poly__degree': randint(1, 3) 
}

# Ridge Parameters
ridge_params = {
    'model': [Ridge(fit_intercept=True, max_iter=5000)], # Increased max_iter
    'model__alpha': uniform(0.1, 10.0),
    'poly__degree': randint(1, 3)
}

# XGBoost Model 
xgb_model = XGBRegressor(random_state=42, tree_method='hist', n_jobs=-1, eval_metric='rmse', verbosity=0)
xgb_params = {
    'n_estimators': randint(500, 1000), 
    'max_depth': randint(3, 7),
    'learning_rate': uniform(0.01, 0.1),
    'reg_alpha': uniform(0, 10),
    'reg_lambda': uniform(0, 5),
    'subsample': uniform(0.7, 0.2),
    'colsample_bytree': uniform(0.7, 0.2)
}

# 14b) Tune Models 
print("\n--- Starting Model Tuning (n_iter=20) ---")

print("Tuning Lasso...")
best_lasso_model, best_lasso_params, best_lasso_rmse = get_best_model(linear_pipe, lasso_params, X, y, n_iter=20)

print("Tuning Ridge...")
best_ridge_model, best_ridge_params, best_ridge_rmse = get_best_model(linear_pipe, ridge_params, X, y, n_iter=20)

print("Tuning XGBoost...")
best_xgb_model, best_xgb_params, best_xgb_rmse = get_best_model(xgb_model, xgb_params, X, y, n_iter=200)

# -----------------------------------------------------------
# 9) Determine and Train the Best Model
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
print("ðŸ† BEST MODEL FROM CROSS-VALIDATION ðŸ†")
print(f"Model: {best_model_name}")
print(f"Cross-Validation RMSE: {best_model_rmse:.4f}")
print("Parameters:", best_model_params)
print("="*50)

# Re-train the best model on the full training set (X)

if best_model_name == 'XGBoost':
    final_model_params = {k: v for k, v in best_model_params.items()}
    final_model = XGBRegressor(random_state=42, tree_method='hist', n_jobs=-1, eval_metric='rmse', verbosity=0, **final_model_params)
    final_model.fit(X, y)
    
elif best_model_name == 'Lasso':
    final_model = linear_pipe.set_params(poly__degree=best_model_params['poly__degree'], 
                                         model=Lasso(alpha=best_model_params['model__alpha'], fit_intercept=True, max_iter=50000))
    final_model.fit(X, y)

elif best_model_name == 'Ridge':
    final_model = linear_pipe.set_params(poly__degree=best_model_params['poly__degree'], 
                                         model=Ridge(alpha=best_model_params['model__alpha'], fit_intercept=True, max_iter=50000))
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
# print("\n--- Final Submission DataFrame Head (Using Best Model) ---")
# print(submission.head())
SUBMISSION_FILE_NAME = 'submission_sales_predictions_Ensemble_AdvancedFE_KNNv2.csv'
FINAL_SUBMISSION_PATH = os.path.join(S3_BUCKET_PATH, SUBMISSION_FILE_NAME)
