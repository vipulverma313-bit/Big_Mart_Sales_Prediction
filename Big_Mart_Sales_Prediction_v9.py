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

# Loading the data - NOTE: Update paths to valid local files if needed!
# Assuming files exist on the user's machine as requested.
train = pd.read_csv("C:/Users/VIVERMA2401/Downloads/train_v9rqX0R.csv")
test = pd.read_csv("C:/Users/VIVERMA2401/Downloads/test_AbJTz2l.csv")
# Store the target variable and identifiers
target = train['Item_Outlet_Sales']
train_len = train.shape[0]

# --- New Imputation and Feature Engineering Functions ---

def preprocess_data(df):
    """Applies new feature engineering steps, including visibility calculation."""
    
    # 1. Standardise Item_Fat_Content
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat', 'Non Edible': 'Non-Edible'})

    # 2. Fix Zero Item_Visibility (using mean, then outlet mean, then global mean)
    
    # Calculate means before attempting replacement (to avoid potential SettingWithCopyWarning)
    visibility_id_avg = df.groupby('Item_Identifier')['Item_Visibility'].transform('mean')
    visibility_outlet_avg = df.groupby('Outlet_Identifier')['Item_Visibility'].transform('mean')
    global_mean = df['Item_Visibility'].mean()

    # Step 1: Replace 0 with Item_Identifier mean
    # Create a mask for zero visibility
    zero_vis_mask = df['Item_Visibility'] == 0
    df.loc[zero_vis_mask, 'Item_Visibility'] = visibility_id_avg.loc[zero_vis_mask]
    
    # Step 2: Handle remaining NaNs (where Item_Identifier mean was also 0/NaN or original data was NaN)
    # Fill these NaNs with Outlet_Identifier mean
    df['Item_Visibility'].fillna(visibility_outlet_avg, inplace=True)

    # Step 3: Fallback for any remaining NaNs (global mean)
    df['Item_Visibility'].fillna(global_mean, inplace=True)


    # 3. Outlet_Years and Item_Type_Combined
    df['Outlet_Years'] = CURRENT_YEAR - df['Outlet_Establishment_Year']
    
    def get_broad_category(x):
        return 'Food' if x[0:2] == 'FD' else ('Drinks' if x[0:2] == 'DR' else 'Non-Consumable')
        
    df['Item_Type_Combined'] = df['Item_Identifier'].apply(get_broad_category)
    
    # Consistency Fix: Non-Consumable items must be Non-Edible fat content (re-added per user request)
    df.loc[df['Item_Type_Combined'] == "Non-Consumable", 'Item_Fat_Content'] = "Non-Edible"
    
    # NEW FEATURE: Average Visibility per Outlet and Visibility Ratio
    # Calculate average visibility per outlet using the cleaned Item_Visibility
    df['average_visibility'] = df.groupby('Outlet_Identifier')['Item_Visibility'].transform('mean')
    # Calculate the visibility ratio relative to the outlet's average
    df['Visibility_Ratio'] = df['Item_Visibility'] / df['average_visibility']
    
    df = df.drop(['Outlet_Establishment_Year', 'Item_Type'], axis=1, errors='ignore')
    
    return df

def impute_missing_values(df):
    """
    Imputes missing values using Item_Identifier-specific mean for Item_Weight 
    and KNN for Outlet_Size.
    """
    
    # 1. Item_Weight imputation: Use mean weight for the same Item_Identifier
    df['Item_Weight'] = df.groupby('Item_Identifier')['Item_Weight'].transform(lambda x: x.fillna(x.mean()))
    # Fallback for any remaining NaNs (global mean)
    df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)
    
    # 2. KNN Imputation for Outlet_Size
    
    # Temporarily encode Outlet_Location_Type and Outlet_Type for KNN
    le_loc = LabelEncoder()
    le_type = LabelEncoder()
    
    # Ensure correct ordinal encoding for KNN features (Tier 1 -> 3, Tier 3 -> 1)
    df['Outlet_Location_Type_Encoded'] = df['Outlet_Location_Type'].map({'Tier 1': 3, 'Tier 2': 2, 'Tier 3': 1})
    df['Outlet_Type_Encoded'] = le_type.fit_transform(df['Outlet_Type'])
    
    # Prepare data subset for KNN
    knn_features_encoded = ['Outlet_Location_Type_Encoded', 'Outlet_Type_Encoded', 'Outlet_Years', 'Item_MRP']
    df_knn = df[['Outlet_Size'] + knn_features_encoded].copy()
    
    # Target encoding: 'NaN' must be treated as a value for encoding
    le_size = LabelEncoder()
    df_knn['Outlet_Size_Encoded'] = le_size.fit_transform(df_knn['Outlet_Size'].astype(str))
    
    # Define train and prediction sets
    train_data = df_knn[df_knn['Outlet_Size'].notnull()]
    predict_data = df_knn[df_knn['Outlet_Size'].isnull()]

    # Features (X) for the model
    X_cols = knn_features_encoded
    X_train = train_data[X_cols]
    y_train = train_data['Outlet_Size_Encoded']
    X_predict = predict_data[X_cols]
    
    # Check if there's any data to predict
    if not X_predict.empty and not X_train.empty:
        # KNN Imputation
        knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance')
        knn_model.fit(X_train, y_train)

        # Predict missing values
        predicted_size_encoded = knn_model.predict(X_predict)
        
        # Inverse transform predictions back to labels
        predicted_size_labels = le_size.inverse_transform(predicted_size_encoded)
        
        # Integrate predictions back into the original DataFrame
        df.loc[df['Outlet_Size'].isnull(), 'Outlet_Size'] = predicted_size_labels
    
    # Clean up temporary columns
    df = df.drop(['Outlet_Location_Type_Encoded', 'Outlet_Type_Encoded'], axis=1)
    
    # Final check: impute any remaining Outlet_Size NaNs with the mode
    df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0], inplace=True)

    # NOTE: Price_Per_Unit calculation removed per user request.
    
    return df


# --- COMBINE AND APPLY NEW PREPROCESSING ---

# Save Identifiers before merging/one-hot encoding
submission_identifiers = test[['Item_Identifier', 'Outlet_Identifier']].copy()

# Drop the target and combine train and test sets
train = train.drop('Item_Outlet_Sales', axis=1, errors='ignore')
combined_data = pd.concat([train, test], ignore_index=True)

# 1. Apply Feature Engineering
combined_data = preprocess_data(combined_data.copy())

# 2. Apply Improved Imputation
combined_data = impute_missing_values(combined_data.copy())

# --- CONTINUE WITH ENCODING & SCALING ---

# 3. Encoding Categorical Variables on Combined Data

# Custom Label Encoding for ordinal features based on size assumption
# Tier 1 (Largest/Biggest) -> 3, Tier 3 (Smallest) -> 1
combined_data['Outlet_Location_Type'] = combined_data['Outlet_Location_Type'].map({'Tier 1': 3, 'Tier 2': 2, 'Tier 3': 1}).astype(int)

# Custom Label Encoding for Outlet_Size (assumed ordinal)
combined_data['Outlet_Size'] = combined_data['Outlet_Size'].map({'Small': 1, 'Medium': 2, 'High': 3}).astype(int)


# Label Encoder for other ordinal-like features
encoder = LabelEncoder()
ordinal_features = ['Item_Fat_Content', 'Outlet_Type', 'Item_Type_Combined']
for feature in ordinal_features:
    combined_data[feature] = encoder.fit_transform(combined_data[feature])
    
# One Hot Encoding
combined_data = pd.get_dummies(combined_data, columns=['Outlet_Identifier'], drop_first=True)
combined_data = pd.get_dummies(combined_data, columns=['Item_Type_Combined'], prefix='Item_Type_Comb', drop_first=False) 

# Drop useless columns
combined_data.drop(labels=['Item_Identifier'], axis=1, inplace=True)


# 4. Split data back
X = combined_data.iloc[:train_len].copy()
y = target.copy() # Using uncapped target, as no capping was performed.
X_test_final = combined_data.iloc[train_len:].copy()

# Final safety check: impute any possible leftover NaNs with the mean
for col in X.columns:
    if X[col].isnull().any():
        mean_val = X[col].mean()
        X[col].fillna(mean_val, inplace=True)
        X_test_final[col].fillna(mean_val, inplace=True)


# 5. Normalization (Standard Scaling) of Continuous Features
scaler = StandardScaler()
# IMPORTANT: Added 'average_visibility' and 'Visibility_Ratio' for scaling
continuous_features = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Years', 'average_visibility', 'Visibility_Ratio']

# Fit scaler on training data and transform both train and test
X[continuous_features] = scaler.fit_transform(X[continuous_features])
X_test_final[continuous_features] = scaler.transform(X_test_final[continuous_features])


# Splitting training data for final model evaluation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# -----------------------------------------------------------
# 6) Model Tuning with Optuna (via RandomizedSearchCV)
# -----------------------------------------------------------

# Linear models must use the pipeline, as they rely on PolynomialFeatures
def get_best_model(model, params, X, y, n_iter=200, cv_folds=5, random_state=0):
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
    
    # NOTE: This line performs the actual hyperparameter search, which may take time.
    search.fit(X, y) 
    
    best_rmse = np.sqrt(-search.best_score_)
    
    return search.best_estimator_, search.best_params_, best_rmse

# Define Models and Parameter Grids

# Lasso/Ridge Pipeline with Poly Features (Scaling not needed here as features are pre-scaled)
linear_pipe = Pipeline([
    # Standard Scaler is NOT needed here as continuous features are already scaled in step 5
    ('poly', PolynomialFeatures(degree=2, include_bias=False)), # include_bias=False as we fit intercept
    ('model', None) # Placeholder for Lasso/Ridge
])

# Lasso Parameters (Increased max_iter to 10000 to resolve ConvergenceWarning)
lasso_params = {
    'model': [Lasso(fit_intercept=True, max_iter=5000)], # Use Lasso as estimator
    'model__alpha': uniform(0.01, 1.0),
    'poly__degree': randint(1, 4) # Reduced complexity for quick run
}

# Ridge Parameters (Increased max_iter to 10000 to resolve ConvergenceWarning)
ridge_params = {
    'model': [Ridge(fit_intercept=True, max_iter=5000)], # Use Ridge as estimator
    'model__alpha': uniform(0.1, 10.0),
    'poly__degree': randint(1, 4)
}

# XGBoost Model (Tree models don't need explicit scaling)
xgb_model = XGBRegressor(random_state=42, tree_method='hist', n_jobs=-1, eval_metric='rmse', verbosity=0)
xgb_params = {
    'n_estimators': randint(500, 1500), # Reduced complexity for quick run
    'max_depth': randint(3, 8),
    'learning_rate': uniform(0.005, 0.1), # Reduced complexity for quick run
    'reg_alpha': uniform(0, 10),
    'reg_lambda': uniform(0, 10),
    'subsample': uniform(0.7, 0.2),
    'colsample_bytree': uniform(0.7, 0.2)
}

# 14b) Tune Models 
# NOTE: Using n_iter=20 for a reasonable search time. Adjust as needed.
print("Tuning Lasso...")
best_lasso_model, best_lasso_params, best_lasso_rmse = get_best_model(linear_pipe, lasso_params, X, y, n_iter=20)

print("Tuning Ridge...")
best_ridge_model, best_ridge_params, best_ridge_rmse = get_best_model(linear_pipe, ridge_params, X, y, n_iter=20)

print("Tuning XGBoost...")
best_xgb_model, best_xgb_params, best_xgb_rmse = get_best_model(xgb_model, xgb_params, X, y, n_iter=20)

# -----------------------------------------------------------
# 7) Determine and Train the Best Model
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
    # Filter out pipeline parameters
    # The actual best_xgb_params will contain the parameters prefixed with the model's original parameter names (e.g., 'n_estimators', 'max_depth')
    final_model_params = {k: v for k, v in best_model_params.items()}
    final_model = XGBRegressor(random_state=42, tree_method='hist', n_jobs=-1, eval_metric='rmse', verbosity=0, **final_model_params)
    final_model.fit(X, y) # Fit on full training data (X, y)
    
elif best_model_name == 'Lasso':
    # The parameters are prefixed with 'poly__' and 'model__' when returned by RandomizedSearchCV
    final_model = linear_pipe.set_params(poly__degree=best_model_params['poly__degree'], 
                                         model=Lasso(alpha=best_model_params['model__alpha'], fit_intercept=True, max_iter=100000)) # Updated max_iter
    final_model.fit(X, y)

elif best_model_name == 'Ridge':
    # The parameters are prefixed with 'poly__' and 'model__' when returned by RandomizedSearchCV
    final_model = linear_pipe.set_params(poly__degree=best_model_params['poly__degree'], 
                                         model=Ridge(alpha=best_model_params['model__alpha'], fit_intercept=True, max_iter=100000)) # Updated max_iter
    final_model.fit(X, y)


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
submission.to_csv("best_model_submission_normalized_features.csv", index=False)
print("\nResults successfully saved to 'best_model_submission_normalized_features.csv'.")
