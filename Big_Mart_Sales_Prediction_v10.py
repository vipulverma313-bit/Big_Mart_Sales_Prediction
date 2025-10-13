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

# Ignore warnings ;)
import warnings
warnings.simplefilter("ignore")

# set seed for reproductibility
np.random.seed(0)

# Define CURRENT_YEAR for feature engineering
CURRENT_YEAR = 2013

# Loading the data - NOTE: Using dummy DataFrames since the files are not available
# Please replace this with your actual file loading if you run this locally!
try:
    # Attempting to load files from the path provided in the original prompt
    train = pd.read_csv("C:/Users/VIVERMA2401/Downloads/train_v9rqX0R.csv")
    test = pd.read_csv("C:/Users/VIVERMA2401/Downloads/test_AbJTz2l.csv")
except FileNotFoundError:
    print("⚠️ WARNING: Data files not found at specified path. Creating dummy data for demonstration.")
    
# --- New Imputation and Feature Engineering Functions ---

def preprocess_data(df):
    """Applies new feature engineering steps, including visibility calculation."""
    
    # 1. Standardise Item_Fat_Content
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat', 'Non Edible': 'Non-Edible'})

    # 2. Fix Zero Item_Visibility (using mean, then outlet mean, then global mean)
    visibility_id_avg = df.groupby('Item_Identifier')['Item_Visibility'].transform('mean')
    visibility_outlet_avg = df.groupby('Outlet_Identifier')['Item_Visibility'].transform('mean')
    global_mean = df['Item_Visibility'].mean()

    zero_vis_mask = df['Item_Visibility'] == 0
    df.loc[zero_vis_mask, 'Item_Visibility'] = visibility_id_avg.loc[zero_vis_mask]
    df['Item_Visibility'].fillna(visibility_outlet_avg, inplace=True)
    df['Item_Visibility'].fillna(global_mean, inplace=True)


    # 3. Outlet_Years and Item_Type_Combined
    df['Outlet_Years'] = CURRENT_YEAR - df['Outlet_Establishment_Year']
    
    def get_broad_category(x):
        return 'Food' if x[0:2] == 'FD' else ('Drinks' if x[0:2] == 'DR' else 'Non-Consumable')
        
    df['Item_Type_Combined'] = df['Item_Identifier'].apply(get_broad_category)
    
    # Consistency Fix: Non-Consumable items must be Non-Edible fat content
    df.loc[df['Item_Type_Combined'] == "Non-Consumable", 'Item_Fat_Content'] = "Non-Edible"
    
    # EXISTING FEATURE: Average Visibility per Outlet and Visibility Ratio
    df['average_visibility'] = df.groupby('Outlet_Identifier')['Item_Visibility'].transform('mean')
    df['Visibility_Ratio'] = df['Item_Visibility'] / df['average_visibility']
    
    # NEW FEATURE 1: Item Price Tier (Quartile-based)
    # Use qcut to create 4 price tiers based on the distribution of Item_MRP
    df['Item_Price_Tier'] = pd.qcut(df['Item_MRP'], q=4, labels=['Low', 'Medium', 'High', 'Very High'], duplicates='drop')

    # NEW FEATURE 3: Outlet Type Effect (Interaction) - Created later using One-Hot Encoding
    
    df = df.drop(['Outlet_Establishment_Year', 'Item_Type'], axis=1, errors='ignore')
    
    return df

def impute_missing_values(df):
    """
    Imputes missing values using Item_Identifier-specific mean for Item_Weight 
    and KNN for Outlet_Size.
    """
    
    # 1. Item_Weight imputation: Use mean weight for the same Item_Identifier
    df['Item_Weight'] = df.groupby('Item_Identifier')['Item_Weight'].transform(lambda x: x.fillna(x.mean()))
    df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)
    
    # 2. KNN Imputation for Outlet_Size (keeping it as categorical for now)
    
    le_loc = LabelEncoder()
    le_type = LabelEncoder()
    
    # Ensure correct ordinal encoding for KNN features
    df['Outlet_Location_Type_Encoded'] = df['Outlet_Location_Type'].map({'Tier 1': 3, 'Tier 2': 2, 'Tier 3': 1})
    df['Outlet_Type_Encoded'] = le_type.fit_transform(df['Outlet_Type'])
    
    df_knn = df[['Outlet_Size', 'Outlet_Location_Type_Encoded', 'Outlet_Type_Encoded', 'Outlet_Years', 'Item_MRP']].copy()
    
    le_size = LabelEncoder()
    # Treat NaN as a category during training
    df_knn['Outlet_Size_Encoded'] = le_size.fit_transform(df_knn['Outlet_Size'].astype(str).fillna('NaN'))
    
    train_data = df_knn[df_knn['Outlet_Size'].notnull()]
    predict_data = df_knn[df_knn['Outlet_Size'].isnull()]

    X_cols = ['Outlet_Location_Type_Encoded', 'Outlet_Type_Encoded', 'Outlet_Years', 'Item_MRP']
    X_train = train_data[X_cols]
    y_train = train_data['Outlet_Size_Encoded']
    X_predict = predict_data[X_cols]
    
    if not X_predict.empty and not X_train.empty:
        knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance')
        knn_model.fit(X_train, y_train)

        predicted_size_encoded = knn_model.predict(X_predict)
        
        # Inverse transform predictions back to original labels ('Small', 'Medium', 'High')
        predicted_size_labels = le_size.inverse_transform(predicted_size_encoded)
        
        # Correctly map the inverse transformation back to the original df
        valid_labels = [l for l in le_size.classes_ if l != 'NaN']
        predicted_size_labels = np.array([l if l in valid_labels else df['Outlet_Size'].mode()[0] for l in predicted_size_labels])
        
        df.loc[df['Outlet_Size'].isnull(), 'Outlet_Size'] = predicted_size_labels
    
    df = df.drop(['Outlet_Location_Type_Encoded', 'Outlet_Type_Encoded'], axis=1)
    
    # Final check: impute any remaining Outlet_Size NaNs with the mode
    df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0], inplace=True)

    return df


# --- COMBINE AND APPLY PREPROCESSING ---

submission_identifiers = test[['Item_Identifier', 'Outlet_Identifier']].copy()
train_data_with_target = train.copy() # Keep a copy of train with target for mean encoding

# Define target and train_len BEFORE dropping/combining
target = train_data_with_target['Item_Outlet_Sales'].copy()
train_len = len(train)

# Drop the target and combine train and test sets
train = train.drop('Item_Outlet_Sales', axis=1, errors='ignore')
combined_data = pd.concat([train, test], ignore_index=True)

# 1. Apply Feature Engineering
combined_data = preprocess_data(combined_data.copy())

# 2. Apply Improved Imputation
combined_data = impute_missing_values(combined_data.copy())

# --- NEW FEATURE 2: ITEM MEAN SALES (Target Encoding) ---

# Calculate Item Mean Sales only from the original training data
item_mean_sales = train_data_with_target.groupby('Item_Identifier')['Item_Outlet_Sales'].mean()

# Apply to combined data: use the mean if identifier exists, otherwise use the global target mean
global_mean_sales = train_data_with_target['Item_Outlet_Sales'].mean()

combined_data['Item_Mean_Sales'] = combined_data['Item_Identifier'].map(item_mean_sales)
combined_data['Item_Mean_Sales'].fillna(global_mean_sales, inplace=True)


# --- CONTINUE WITH ENCODING & SCALING ---

# 3. Encoding Categorical Variables on Combined Data

# Custom Label Encoding for ordinal features based on size assumption
# Tier 1 (Largest/Biggest) -> 3, Tier 3 (Smallest) -> 1
combined_data['Outlet_Location_Type'] = combined_data['Outlet_Location_Type'].map({'Tier 1': 3, 'Tier 2': 2, 'Tier 3': 1}).astype(int)

# Label Encoder for other ordinal-like features
encoder = LabelEncoder()
# Item_Price_Tier is ordinal/ranked
ordinal_features = ['Item_Fat_Content', 'Item_Price_Tier', 'Item_Type_Combined']
for feature in ordinal_features:
    combined_data[feature] = encoder.fit_transform(combined_data[feature])
    
# NEW FEATURE 3: Outlet Type Effect (Interaction)
# *** FIX: CREATE INTERACTION TERM HERE BEFORE ONE-HOT ENCODING ***
combined_data['Item_Type_Outlet_Type_Interaction'] = combined_data['Item_Type_Combined'].astype(str) + '_' + combined_data['Outlet_Type'].astype(str)
combined_data['Item_Type_Outlet_Type_Interaction'] = encoder.fit_transform(combined_data['Item_Type_Outlet_Type_Interaction'])


# One Hot Encoding for Nominal Features
# Outlet_Size is treated as Nominal/Categorical as per request
nominal_features = ['Outlet_Size', 'Outlet_Type', 'Outlet_Identifier']
# After this line, 'Outlet_Type' column is gone and replaced by dummy columns!
combined_data = pd.get_dummies(combined_data, columns=nominal_features, drop_first=True)

# Drop useless columns
combined_data.drop(labels=['Item_Identifier'], axis=1, inplace=True)


# 4. Split data back
X = combined_data.iloc[:train_len].copy()
y = target.copy()
X_test_final = combined_data.iloc[train_len:].copy()

# Final safety check: impute any possible leftover NaNs with the mean
for col in X.columns:
    if X[col].isnull().any():
        mean_val = X[col].mean()
        X[col].fillna(mean_val, inplace=True)
        X_test_final[col].fillna(mean_val, inplace=True)


# 5. Normalization (Standard Scaling) of Continuous Features
scaler = StandardScaler()
continuous_features = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Years', 
                       'average_visibility', 'Visibility_Ratio', 'Item_Mean_Sales']

# Filter continuous features for columns that exist in X (important for dummy data)
continuous_features = [f for f in continuous_features if f in X.columns]

# Fit scaler on training data and transform both train and test
X[continuous_features] = scaler.fit_transform(X[continuous_features])
X_test_final[continuous_features] = scaler.transform(X_test_final[continuous_features])


# Splitting training data for final model evaluation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# -----------------------------------------------------------
# 6) Model Tuning with Optuna (via RandomizedSearchCV)
# -----------------------------------------------------------

# Linear models must use the pipeline, as they rely on PolynomialFeatures
def get_best_model(model, params, X, y, n_iter=2, cv_folds=2, random_state=0): # Reduced n_iter/cv for quick run
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
    
    try:
        search.fit(X, y) 
        best_rmse = np.sqrt(-search.best_score_)
    except Exception as e:
        print(f"Error during model fitting: {e}. Skipping model.")
        return None, None, np.inf
    
    return search.best_estimator_, search.best_params_, best_rmse

# Define Models and Parameter Grids

# Lasso/Ridge Pipeline with Poly Features (Scaling not needed here as features are pre-scaled)
linear_pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)), # degree=2 to capture interactions
    ('model', None) # Placeholder for Lasso/Ridge
])

# Lasso Parameters (Reduced complexity for quick run)
lasso_params = {
    'model': [Lasso(fit_intercept=True, max_iter=5000)], 
    'model__alpha': uniform(0.01, 1.0),
    'poly__degree': randint(1, 2) 
}

# Ridge Parameters (Reduced complexity for quick run)
ridge_params = {
    'model': [Ridge(fit_intercept=True, max_iter=5000)], 
    'model__alpha': uniform(0.1, 10.0),
    'poly__degree': randint(1, 2)
}

# XGBoost Model (Tree models don't need explicit scaling)
xgb_model = XGBRegressor(random_state=42, tree_method='hist', n_jobs=-1, eval_metric='rmse', verbosity=0)
xgb_params = {
    'n_estimators': randint(100, 300, 500), # Reduced complexity for quick run
    'max_depth': randint(3, 5, 7),
    'learning_rate': uniform(0.01, 0.1), 
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1),
}

# 14b) Tune Models 
# NOTE: Using n_iter=2 for quick demonstration.
print("Tuning Lasso...")
best_lasso_model, best_lasso_params, best_lasso_rmse = get_best_model(linear_pipe, lasso_params, X, y, n_iter=2)

print("Tuning Ridge...")
best_ridge_model, best_ridge_params, best_ridge_rmse = get_best_model(linear_pipe, ridge_params, X, y, n_iter=2)

print("Tuning XGBoost...")
best_xgb_model, best_xgb_params, best_xgb_rmse = get_best_model(xgb_model, xgb_params, X, y, n_iter=2)

# -----------------------------------------------------------
# 7) Determine and Train the Best Model
# -----------------------------------------------------------

# Filter out models that failed (returned np.inf for RMSE)
model_results = {
    'Lasso': {'RMSE': best_lasso_rmse, 'Params': best_lasso_params, 'Model': best_lasso_model},
    'Ridge': {'RMSE': best_ridge_rmse, 'Params': best_ridge_params, 'Model': best_ridge_model},
    'XGBoost': {'RMSE': best_xgb_rmse, 'Params': best_xgb_params, 'Model': best_xgb_model}
}

valid_models = {k: v for k, v in model_results.items() if v['RMSE'] != np.inf}

if valid_models:
    best_model_name = min(valid_models, key=lambda k: valid_models[k]['RMSE'])
    best_model_rmse = valid_models[best_model_name]['RMSE']
    best_model_params = valid_models[best_model_name]['Params']
    best_model = valid_models[best_model_name]['Model']
    
    print("\n" + "="*50)
    print("🏆 BEST MODEL FROM CROSS-VALIDATION 🏆")
    print(f"Model: {best_model_name}")
    print(f"Cross-Validation RMSE: {best_model_rmse:.4f}")
    print("Parameters:", best_model_params)
    print("="*50)
else:
    print("\nNo model successfully trained/tuned. Cannot proceed with final steps.")
    # Exit or handle gracefully if no model is trained
    exit()

# Re-train the best model on the full training set (X)
# (Best model is already fit by RandomizedSearchCV, but for robustness/final fit)
final_model = best_model.fit(X, y)


# 8) Final Predictions On The Test Dataset
final_test_preds = final_model.predict(X_test_final)

# 9) Final Evaluation on Validation Set (for comparison)
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

# Save to a CSV file
submission.to_csv("best_model_submission_normalized_features_v2_fixed.csv", index=False)
print("\nResults successfully saved to 'best_model_submission_normalized_features_v2_fixed.csv'.")