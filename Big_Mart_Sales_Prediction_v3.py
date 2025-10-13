import pandas as pd
import numpy as np
from scipy.stats import skew, boxcox
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set a random seed for reproducibility
np.random.seed(42)

# --- S3 Configuration (Local Path) ---
S3_BUCKET_PATH = 'C:/Users/VIVERMA2401/Downloads/'
TRAIN_FILE = 'train_v9rqX0R.csv'
TEST_FILE = 'test_AbJTz2l.csv'

# --- Data Loading and Initial Setup ---
def load_data_from_csv(csv_path, train_file, test_file):
    try:
        train_csv_url = os.path.join(csv_path, train_file)
        test_csv_url = os.path.join(csv_path, test_file)
        df_train = pd.read_csv(train_csv_url)
        df_test = pd.read_csv(test_csv_url)
        return df_train, df_test
    except Exception as e:
        print(f"FATAL: Data loading failed. Error: {e}")
        raise

try:
    df_train, df_test = load_data_from_csv(S3_BUCKET_PATH, TRAIN_FILE, TEST_FILE)
    original_test_identifiers = df_test[['Item_Identifier', 'Outlet_Identifier']].copy()
    
    df_train['source'] = 'train'
    df_test['source'] = 'test'
    
    target = 'Item_Outlet_Sales'
    train_sales = df_train[target].copy()
    df_train = df_train.drop([target], axis=1)
    
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    train_sales_log = np.log1p(train_sales)
    print(f"Combined data shape: {df_combined.shape}")
except Exception:
    exit()

# --- Preprocessing Functions ---

def preprocess_data(df):
    # 1. Standardise Item_Fat_Content
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'})

    # 2. Fix Zero Item_Visibility
    visibility_avg = df.groupby('Item_Identifier')['Item_Visibility'].transform('mean')
    df.loc[df['Item_Visibility'] == 0, 'Item_Visibility'] = visibility_avg

    # 3. Create New Features (Outlet_Years and Item_Type_Combined)
    df['Outlet_Years'] = 2025 - df['Outlet_Establishment_Year']
    
    def get_broad_category(x):
        return 'Food' if x[0:2] == 'FD' else ('Drinks' if x[0:2] == 'DR' else 'Non-Consumable')
        
    df['Item_Type_Combined'] = df['Item_Identifier'].apply(get_broad_category)
    df.loc[df['Item_Type_Combined'] == "Non-Consumable", 'Item_Fat_Content'] = "Non-Edible"
    df = df.drop(['Outlet_Establishment_Year'], axis=1)
    
    # 4. NEW FEATURE 1: Price per Unit
    df['Item_Weight_Clean'] = df['Item_Weight'].replace(0, 1e-6) # Replace 0 with tiny number
    df['Price_Per_Unit'] = df['Item_MRP'] / df['Item_Weight_Clean']
    df = df.drop('Item_Weight_Clean', axis=1)
    
    # 5. NEW FEATURE 2: Item_MRP Bins (Categorical)
    df['MRP_Tier'] = pd.cut(df['Item_MRP'], bins=[0, 70, 140, 210, df['Item_MRP'].max()], 
                           labels=['Low', 'Medium', 'High', 'Very_High'], right=False).astype('object') # Ensure this is object for OHE
    
    return df

def impute_missing_values(df):
    # Item_Weight imputation: Use mean weight for the same Item_Identifier
    df['Item_Weight'] = df.groupby('Item_Identifier')['Item_Weight'].transform(lambda x: x.fillna(x.mean()))
    
    # Outlet_Size imputation using mode from Outlet_Type/Location_Type
    df['Outlet_Size'] = df.groupby(['Outlet_Type', 'Outlet_Location_Type'])['Outlet_Size'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Medium'))
    
    return df

def fix_skewness_outliers(df, features):
    # Apply Box-Cox to Item_Visibility
    for col in features:
        if col == 'Item_Visibility':
            df[col], _ = boxcox(df[col] + 1e-6) 
    return df

# --- Execution of Preprocessing ---
df_combined = preprocess_data(df_combined.copy())
df_combined = impute_missing_values(df_combined.copy())
continuous_vars = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Years', 'Price_Per_Unit']
df_combined = fix_skewness_outliers(df_combined.copy(), continuous_vars)

# --- Feature Encoding ---
# Ensure ALL categorical columns are in this list, including the new 'MRP_Tier'
categorical_cols = df_combined.select_dtypes(include='object').columns.tolist()
cols_to_remove = ['source', 'Item_Identifier', 'Outlet_Identifier']
categorical_cols = [col for col in categorical_cols if col not in cols_to_remove]

print(f"\nColumns to One-Hot Encode (OHE): {categorical_cols}")

# One-Hot Encoding
df_combined = pd.get_dummies(df_combined, columns=categorical_cols, drop_first=True)

# --- Train/Test Split ---
X_train_full = df_combined[df_combined['source'] == 'train'].drop(cols_to_remove + ['source'], axis=1).copy()
X_test_full = df_combined[df_combined['source'] == 'test'].drop(cols_to_remove + ['source'], axis=1).copy()
y_train = train_sales_log 

# =======================================================================
# 🛑 CRITICAL FIX: Enforce Float Dtype for ALL columns
# This resolves both 'KeyError: category' and 'ValueError: could not convert string'
# =======================================================================
X_train = X_train_full.astype('float64')
X_test = X_test_full.astype('float64')

print(f"X_train shape after final cleaning: {X_train.shape}")

# --- FIXED LEAKAGE: Feature Scaling ---
print("\n--- Applying Feature Scaling (Leakage Fixed) ---")
scaler = StandardScaler()
# Find the columns that need scaling (must exist after OHE)
numeric_cols_to_scale = [col for col in continuous_vars if col in X_train.columns]

# Fit scaler only on the training data and transform X_train
X_train[numeric_cols_to_scale] = scaler.fit_transform(X_train[numeric_cols_to_scale])

# Transform X_test using the scaler fitted on X_train
X_test[numeric_cols_to_scale] = scaler.transform(X_test[numeric_cols_to_scale])

# --- Model Training (Expanded Grid Search) ---

# Custom Scorer for Actual RMSE
def actual_rmse_scorer(estimator, X, y_log):
    y_pred_log = estimator.predict(X)
    y_pred_actual = np.expm1(y_pred_log)
    y_actual = np.expm1(y_log)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred_actual))
    return -rmse 

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scoring_metric = 'neg_mean_squared_error'

## 1. XGBoost Regressor (EXPANDED TUNING)
print("\n--- Training XGBoost with Expanded Hyperparameter Grid ---")
xgb = XGBRegressor(random_state=42, tree_method='hist') # Use 'hist' for faster training
xgb_params = {
    'n_estimators': [500, 800, 1000, 1200],  
    'max_depth': [3, 4, 5, 7],            
    'learning_rate': [0.01, 0.03, 0.05, 0.1], 
    'reg_alpha': [0, 0.005, 0.01],     # L1 Regularization
    'gamma': [0, 0.1, 0.2]             # Minimum loss reduction
}

grid_search_xgb = GridSearchCV(xgb, xgb_params, cv=kf, scoring=scoring_metric, verbose=1, n_jobs=-1)
grid_search_xgb.fit(X_train, y_train)

best_xgb = grid_search_xgb.best_estimator_
xgb_cv_score_log = np.sqrt(-grid_search_xgb.best_score_)
print(f"Best XGBoost Params: {grid_search_xgb.best_params_}")
print(f"XGBoost Log-Scale CV RMSE: {xgb_cv_score_log:.4f}")

# Calculate Actual CV RMSE
xgb_cv_scores_actual = cross_val_score(best_xgb, X_train, y_train, cv=kf, scoring=actual_rmse_scorer, n_jobs=-1)
xgb_cv_score_actual = -xgb_cv_scores_actual.mean()
print(f"XGBoost Actual-Scale CV RMSE: {xgb_cv_score_actual:.4f}")

# --- 2. Generate Final Predictions using the Best XGBoost Model ---
final_model = best_xgb 

print(f"\n--- Generating CSV using the Best Tuned XGBoost Regressor (CV RMSE: {xgb_cv_score_actual:.2f}) ---")

# Make predictions
test_predictions_log = final_model.predict(X_test)
test_predictions_final = np.expm1(test_predictions_log)
test_predictions_final[test_predictions_final < 0] = 0

# --- 3. Create and Save Submission ---
submission = pd.DataFrame({
    'Item_Identifier': original_test_identifiers['Item_Identifier'],
    'Outlet_Identifier': original_test_identifiers['Outlet_Identifier'],
    'Item_Outlet_Sales': test_predictions_final
})

SUBMISSION_FILE_NAME = 'submission_sales_predictions_XGBoost_Tuned.csv'
FINAL_SUBMISSION_PATH = os.path.join(S3_BUCKET_PATH, SUBMISSION_FILE_NAME)

print("\nSubmission Preview:")
print(submission.head())

try:
    submission.to_csv(FINAL_SUBMISSION_PATH, index=False)
    print(f"\n✅ Successfully saved submission file to: {FINAL_SUBMISSION_PATH}")
except Exception as e:
    print(f"\n❌ Error saving file: {e}")