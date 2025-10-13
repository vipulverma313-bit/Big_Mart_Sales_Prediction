import pandas as pd
import numpy as np
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler
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

# --- CONFIG ---
S3_BUCKET_PATH = 'C:/Users/VIVERMA2401/Downloads/'
TRAIN_FILE = 'train_v9rqX0R.csv'
TEST_FILE = 'test_AbJTz2l.csv'
TARGET_COL = 'Item_Outlet_Sales'

# --- Data Loading ---
def load_data_and_split(csv_path, train_file, test_file, target_col):
    try:
        df_train = pd.read_csv(os.path.join(csv_path, train_file))
        df_test = pd.read_csv(os.path.join(csv_path, test_file))
        
        original_test_identifiers = df_test[['Item_Identifier', 'Outlet_Identifier']].copy()
        train_sales = df_train[target_col].copy()
        df_train = df_train.drop([target_col], axis=1)
        
        df_train['source'] = 'train'
        df_test['source'] = 'test'
        df_combined = pd.concat([df_train, df_test], ignore_index=True)
        train_sales_log = np.log1p(train_sales)
        
        return df_combined, original_test_identifiers, train_sales, train_sales_log
    except Exception as e:
        print(f"FATAL: Data loading failed. Error: {e}")
        raise

df_combined, original_test_identifiers, train_sales, train_sales_log = \
    load_data_and_split(S3_BUCKET_PATH, TRAIN_FILE, TEST_FILE, TARGET_COL)
print(f"Combined data shape: {df_combined.shape}")

# --- Feature Engineering and Imputation ---

def preprocess_data(df):
    # 1. Standardise Item_Fat_Content
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'})

    # 2. Fix Zero Item_Visibility (using mean)
    visibility_avg = df.groupby('Item_Identifier')['Item_Visibility'].transform('mean')
    df.loc[df['Item_Visibility'] == 0, 'Item_Visibility'] = visibility_avg

    # 3. Outlet_Years and Item_Type_Combined
    df['Outlet_Years'] = 2025 - df['Outlet_Establishment_Year']
    
    def get_broad_category(x):
        return 'Food' if x[0:2] == 'FD' else ('Drinks' if x[0:2] == 'DR' else 'Non-Consumable')
        
    df['Item_Type_Combined'] = df['Item_Identifier'].apply(get_broad_category)
    df.loc[df['Item_Type_Combined'] == "Non-Consumable", 'Item_Fat_Content'] = "Non-Edible"
    df = df.drop(['Outlet_Establishment_Year', 'Item_Type'], axis=1, errors='ignore')
    
    # 4. Core Feature: Price per Unit
    df['Item_Weight_Clean'] = df['Item_Weight'].replace(0, 1e-6)
    df['Price_Per_Unit'] = df['Item_MRP'] / df['Item_Weight_Clean']
    df = df.drop('Item_Weight_Clean', axis=1)

    return df

def impute_missing_values(df):
    # Item_Weight imputation: Use mean weight for the same Item_Identifier
    df['Item_Weight'] = df.groupby('Item_Identifier')['Item_Weight'].transform(lambda x: x.fillna(x.mean()))
    
    # Outlet_Size Imputation using known pattern
    size_map = {
        'Grocery Store': 'Small',
        'Supermarket Type1': 'Medium', 
        'Supermarket Type2': 'Medium', 
        'Supermarket Type3': 'Medium'  
    }
    df['Outlet_Size'] = df.apply(lambda row: size_map.get(row['Outlet_Type'], 'Medium') if pd.isnull(row['Outlet_Size']) else row['Outlet_Size'], axis=1)
    
    return df

def fix_skewness_outliers(df):
    if 'Item_Visibility' in df.columns:
        df['Item_Visibility'], _ = boxcox(df['Item_Visibility'] + 1e-6) 
    return df

# --- NEW OUTLET INTERACTION FEATURES ---
def create_outlet_features(df):
    # 1. Ordinal Encoding: Convert size and tier to numerical ranks
    size_map = {'Small': 1, 'Medium': 2, 'High': 3}
    tier_map = {'Tier 3': 1, 'Tier 2': 2, 'Tier 1': 3} # Tier 1 is highest score
    
    df['Outlet_Size_Score'] = df['Outlet_Size'].map(size_map).fillna(0)
    df['Outlet_Location_Score'] = df['Outlet_Location_Type'].map(tier_map).fillna(0)
    
    # 2. Interaction Feature: Combined score (Multiplicative is usually best)
    # This captures the combined effect: Tier 1 & High Size = 9 (3*3)
    df['Outlet_Interaction_Score'] = df['Outlet_Size_Score'] * df['Outlet_Location_Score']
    
    # 3. Drop the temporary scores, keeping the Interaction Score
    df = df.drop(['Outlet_Size_Score', 'Outlet_Location_Score'], axis=1)
    
    return df

df_combined = preprocess_data(df_combined.copy())
df_combined = impute_missing_values(df_combined.copy())
df_combined = fix_skewness_outliers(df_combined.copy())
df_combined = create_outlet_features(df_combined.copy()) # New step

# --- Target Encoding (Retained from previous best attempt) ---
def create_target_features(df, sales_log):
    df_train = df[df['source'] == 'train'].copy()
    df_train['sales_log'] = sales_log
    
    mean_by_type = df_train.groupby(['Item_Type_Combined', 'Outlet_Type'])['sales_log'].mean()
    df['Item_Outlet_Type_Mean_Sales'] = df.apply(
        lambda row: mean_by_type.get((row['Item_Type_Combined'], row['Outlet_Type']), mean_by_type.mean()), axis=1
    )
    
    mean_by_id = df_train.groupby(['Item_Type_Combined', 'Outlet_Identifier'])['sales_log'].mean()
    df['Item_Outlet_ID_Mean_Sales'] = df.apply(
        lambda row: mean_by_id.get((row['Item_Type_Combined'], row['Outlet_Identifier']), mean_by_id.mean()), axis=1
    )
    
    return df

df_combined = create_target_features(df_combined, train_sales_log)
continuous_vars = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Years', 
                   'Price_Per_Unit', 'Item_Outlet_Type_Mean_Sales', 'Item_Outlet_ID_Mean_Sales', 
                   'Outlet_Interaction_Score'] # ADDED new feature

# --- Feature Encoding ---
categorical_cols = df_combined.select_dtypes(include='object').columns.tolist()
cols_to_remove = ['source', 'Item_Identifier', 'Outlet_Identifier']
categorical_cols = [col for col in categorical_cols if col not in cols_to_remove]

print(f"\nColumns to One-Hot Encode (OHE): {categorical_cols}")
df_combined = pd.get_dummies(df_combined, columns=categorical_cols, drop_first=True)

# --- Final Split, Dtype Enforcement, and Scaling FIX ---
X_train_full = df_combined[df_combined['source'] == 'train'].drop(cols_to_remove + ['source'], axis=1).copy()
X_test_full = df_combined[df_combined['source'] == 'test'].drop(cols_to_remove + ['source'], axis=1).copy()
y_train = train_sales_log 

# CRITICAL FIX: Enforce Float Dtype
X_train = X_train_full.astype('float64')
X_test = X_test_full.astype('float64')

# FIXED LEAKAGE: Feature Scaling
print("\n--- Applying Feature Scaling (Leakage Fixed) ---")
scaler = StandardScaler()
numeric_cols_to_scale = [col for col in continuous_vars if col in X_train.columns]

X_train[numeric_cols_to_scale] = scaler.fit_transform(X_train[numeric_cols_to_scale])
X_test[numeric_cols_to_scale] = scaler.transform(X_test[numeric_cols_to_scale])

# --- Model Training (Targeted XGBoost Grid Search) ---
def actual_rmse_scorer(estimator, X, y_log):
    y_pred_log = estimator.predict(X)
    y_pred_actual = np.expm1(y_pred_log)
    y_actual = np.expm1(y_log)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred_actual))
    return -rmse 

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scoring_metric = 'neg_mean_squared_error'

print("\n--- Training XGBoost with Targeted Hyperparameter Grid ---")
xgb = XGBRegressor(random_state=42, tree_method='hist', n_jobs=-1) 

# Targeted Grid: Increased n_estimators slightly
xgb_params = {
    'n_estimators': [1000, 1500],  
    'max_depth': [3, 4],           
    'learning_rate': [0.01, 0.03], # The most important parameter for marginal gains
    'reg_alpha': [0, 0.01, 5, 10] 
}

grid_search_xgb = GridSearchCV(xgb, xgb_params, cv=kf, scoring=scoring_metric, verbose=1, n_jobs=-1)
grid_search_xgb.fit(X_train, y_train)

best_xgb = grid_search_xgb.best_estimator_
xgb_cv_scores_actual = cross_val_score(best_xgb, X_train, y_train, cv=kf, scoring=actual_rmse_scorer, n_jobs=-1)
xgb_cv_score_actual = -xgb_cv_scores_actual.mean()
print(f"XGBoost Actual-Scale CV RMSE: {xgb_cv_score_actual:.4f}")

# --- 3. Final Prediction and Submission ---
final_model = best_xgb 

print(f"\n--- Generating CSV using the Best Tuned XGBoost Regressor (CV RMSE: {xgb_cv_score_actual:.2f}) ---")

test_predictions_log = final_model.predict(X_test)
test_predictions_final = np.expm1(test_predictions_log)
test_predictions_final[test_predictions_final < 0] = 0

submission = pd.DataFrame({
    'Item_Identifier': original_test_identifiers['Item_Identifier'],
    'Outlet_Identifier': original_test_identifiers['Outlet_Identifier'],
    'Item_Outlet_Sales': test_predictions_final
})

SUBMISSION_FILE_NAME = 'submission_sales_predictions_XGBoost_OutletFE.csv'
FINAL_SUBMISSION_PATH = os.path.join(S3_BUCKET_PATH, SUBMISSION_FILE_NAME)

try:
    submission.to_csv(FINAL_SUBMISSION_PATH, index=False)
    print(f"\n✅ Successfully saved submission file to: {FINAL_SUBMISSION_PATH}")
except Exception as e:
    print(f"\n❌ Error saving file: {e}")