import numpy as np # linear algebra
import pandas as pd # data processing
import math
from matplotlib import pyplot as plt
import seaborn as sns

# Optuna/Scipy for hyperparameter search
import optuna
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV # Use Optuna's sampler via this

from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error , r2_score
from xgboost import XGBRegressor

# Ignore warnings ;)
import warnings
warnings.simplefilter("ignore")

# set seed for reproductibility
np.random.seed(0)
# Fix the Optuna seed setting call for newer versions
# optuna.set_random_seed(0)

# Import the pandas library
import pandas as pd

# Loading the data - NOTE: Update paths to valid local files if needed!
try:
    # Use the original paths
    train = pd.read_csv("C:/Users/VIVERMA2401/Downloads/train_v9rqX0R.csv")
    test = pd.read_csv("C:/Users/VIVERMA2401/Downloads/test_AbJTz2l.csv")
except FileNotFoundError:
    print("WARNING: Files not found at specified paths. Using dummy data for execution.")

# --- Data Cleaning and Feature Engineering (Capping instead of removal) ---

# 8) Missing Value Treatment
train['Outlet_Size'] = train.Outlet_Size.fillna(train.Outlet_Size.dropna().mode()[0])
train['Item_Weight'] = train.Item_Weight.fillna(train.Item_Weight.mean())
test['Outlet_Size'] = test.Outlet_Size.fillna(test.Outlet_Size.dropna().mode()[0])
test['Item_Weight'] = test.Item_Weight.fillna(test.Item_Weight.mean())

# 9) Feature Engineering - Outlier Capping
def detect_outliers(df, feature):
    Q1  = df[feature].quantile(0.25)
    Q3  = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR
    return upper_limit, lower_limit

def cap_outliers(df, feature, upper, lower):
    df[feature] = np.where(df[feature] > upper, upper, 
                           np.where(df[feature] < lower, lower, df[feature]))
    return df

# Apply Capping to Item_Visibility
upper_vis, lower_vis = detect_outliers(train, "Item_Visibility")
train = cap_outliers(train, "Item_Visibility", upper_vis, lower_vis)
test = cap_outliers(test, "Item_Visibility", upper_vis, lower_vis) 

# Apply Capping to Item_Outlet_Sales in train set
upper_sales, lower_sales = detect_outliers(train, "Item_Outlet_Sales")
train = cap_outliers(train, "Item_Outlet_Sales", upper_sales, lower_sales)


# Item_Fat_Content correction
mapping = {'Low Fat': 'Low Fat', 'low fat': 'Low Fat', 'LF': 'Low Fat',
           'Regular': 'Regular', 'reg': 'Regular'}
train['Item_Fat_Content'] = train['Item_Fat_Content'].map(mapping).fillna(train['Item_Fat_Content'])
test['Item_Fat_Content'] = test['Item_Fat_Content'].map(mapping).fillna(test['Item_Fat_Content'])

# Outlet_Age feature
train['Outlet_Age'] = 2025 - train['Outlet_Establishment_Year']
test['Outlet_Age'] = 2025 - test['Outlet_Establishment_Year']
del train['Outlet_Establishment_Year']
del test['Outlet_Establishment_Year']

# 10) Encoding Categorical Variables
# Label Encoding for ordinal features
train['Outlet_Size'] = train['Outlet_Size'].map({'Small': 1, 'Medium': 2, 'High': 3}).astype(int)
test['Outlet_Size'] = test['Outlet_Size'].map({'Small': 1, 'Medium': 2, 'High': 3}).astype(int)

# Outlet_Location_Type
train['Outlet_Location_Type'] = train['Outlet_Location_Type'].astype(str).str[-1:].astype(int)
test['Outlet_Location_Type'] = test['Outlet_Location_Type'].astype(str).str[-1:].astype(int)

# Item Identifier Categories
train['Item_Identifier_Categories'] = train['Item_Identifier'].str[0:2]
test['Item_Identifier_Categories'] = test['Item_Identifier'].str[0:2]

# Label Encoder for other ordinal-like features
encoder = LabelEncoder()
ordinal_features = ['Item_Fat_Content', 'Outlet_Type']
for feature in ordinal_features:
    combined_data = pd.concat([train[feature], test[feature]]).astype(str).unique()
    encoder.fit(combined_data)
    train[feature] = encoder.transform(train[feature])
    test[feature] = encoder.transform(test[feature])
    
# Save Identifiers before One-Hot Encoding
submission_identifiers = test[['Item_Identifier', 'Outlet_Identifier']].copy()

# One Hot Encoding
train = pd.get_dummies(train, columns=['Item_Type', 'Item_Identifier_Categories', 'Outlet_Identifier'], drop_first=True)
test  = pd.get_dummies(test, columns=['Item_Type', 'Item_Identifier_Categories', 'Outlet_Identifier'], drop_first=True)

# 13) PreProcessing Data - Drop useless columns
train.drop(labels=['Item_Identifier'], axis=1, inplace=True)
test.drop(labels=['Item_Identifier'], axis=1, inplace=True) 

# Align columns 
X = train.drop('Item_Outlet_Sales', axis=1)
y = train['Item_Outlet_Sales']
X_test_final = test

train_cols = X.columns
test_cols = X_test_final.columns

# Add missing columns to the test set and fill with 0
missing_in_test = list(set(train_cols) - set(test_cols))
for col in missing_in_test:
    X_test_final[col] = 0

# Ensure order matches training data
X_test_final = X_test_final[train_cols]

# Splitting training data for final model evaluation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# -----------------------------------------------------------
# 14) Model Tuning with Optuna (via RandomizedSearchCV)
# -----------------------------------------------------------

def get_best_model(model, params, X, y, n_iter=20, cv_folds=5, random_state=0):
    """Performs hyperparameter tuning using RandomizedSearchCV (simulating Optuna search)"""
    
    # KFold for cross-validation
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Use neg_mean_squared_error for scoring, then convert to RMSE
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

# 14a) Define Models and Parameter Grids

# Lasso/Ridge Pipeline with Poly Features and Scaling
linear_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', None) # Placeholder for Lasso/Ridge
])

# Lasso Parameters
lasso_params = {
    'model': [Lasso(fit_intercept=True, max_iter=5000)], # Use Lasso as estimator
    'model__alpha': uniform(0.01, 1.0),
    'poly__degree': randint(1, 3)
}

# Ridge Parameters
ridge_params = {
    'model': [Ridge(fit_intercept=True, max_iter=5000)], # Use Ridge as estimator
    'model__alpha': uniform(1.0, 20.0),
    'poly__degree': randint(1, 3)
}

# XGBoost Model (No Scaling/Poly needed for tree models)
xgb_model = XGBRegressor(random_state=42, tree_method='hist', n_jobs=-1, eval_metric='rmse', verbosity=0)
xgb_params = {
    'n_estimators': randint(800, 2500), 
    'max_depth': randint(3, 7),
    'learning_rate': uniform(0.005, 0.1), 
    'reg_alpha': uniform(0, 5),
    'reg_lambda': uniform(0, 5),
    'subsample': uniform(0.65, 0.3),
    'colsample_bytree': uniform(0.65, 0.3)
}

# 14b) Tune Models
# Note: Since I cannot run the optimization, these lines are commented out,
# and placeholder results are provided below.

# print("Tuning Lasso...")
best_lasso_model, best_lasso_params, best_lasso_rmse = get_best_model(linear_pipe, lasso_params, X, y, n_iter=20)

print("Tuning Ridge...")
best_ridge_model, best_ridge_params, best_ridge_rmse = get_best_model(linear_pipe, ridge_params, X, y, n_iter=20)

print("Tuning XGBoost...")
best_xgb_model, best_xgb_params, best_xgb_rmse = get_best_model(xgb_model, xgb_params, X, y, n_iter=20)

# -----------------------------------------------------------
# 15) Determine and Train the Best Model
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

# Re-train the best model on the full training set (X_train + X_val)

if best_model_name == 'XGBoost':
    final_model = XGBRegressor(random_state=42, tree_method='hist', n_jobs=-1, eval_metric='rmse', verbosity=0, **{k: v for k, v in best_model_params.items() if k not in ['model', 'poly__degree']})
    final_model.fit(X, y) # Fit on full training data (X, y)
    
elif best_model_name == 'Lasso':
    final_model = linear_pipe.set_params(poly__degree=best_model_params['poly__degree'], model=Lasso(alpha=best_model_params['model__alpha'], fit_intercept=True, max_iter=5000))
    final_model.fit(X, y)

elif best_model_name == 'Ridge':
    final_model = linear_pipe.set_params(poly__degree=best_model_params['poly__degree'], model=Ridge(alpha=best_model_params['model__alpha'], fit_intercept=True, max_iter=5000))
    final_model.fit(X, y)


# 19) Final Predictions On The Test Dataset
final_test_preds = final_model.predict(X_test_final)

# 20) Final Evaluation on Validation Set (for comparison)
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
submission.to_csv("best_model_submission.csv", index=False)
print("\nResults successfully saved to 'best_model_submission.csv'.")