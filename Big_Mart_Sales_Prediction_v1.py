import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, boxcox
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import VotingRegressor
import warnings
import os # Import os for path handling

# Suppress warnings during CV/GridSearch
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set a random seed for reproducibility
np.random.seed(42)

# --- S3 Configuration (Used for local path as per original code) ---
S3_BUCKET_PATH = 'C:/Users/VIVERMA2401/Downloads/'
TRAIN_FILE = 'train_v9rqX0R.csv'
TEST_FILE = 'test_AbJTz2l.csv'

def load_data_from_csv(csv_path, train_file, test_file):
    """Loads data from the specified local CSV files."""
    try:
        train_csv_url = os.path.join(csv_path, train_file)
        test_csv_url = os.path.join(csv_path, test_file)
        
        print(f"Loading training data from: {train_csv_url}")
        df_train = pd.read_csv(train_csv_url)
        
        print(f"Loading test data from: {test_csv_url}")
        df_test = pd.read_csv(test_csv_url)
        
        return df_train, df_test
    
    except Exception as e:
        print(f"An error occurred while reading from CSV. Check file paths.")
        print(f"Error: {e}")
        raise

# --- Load Data ---
try:
    df_train, df_test = load_data_from_csv(S3_BUCKET_PATH, TRAIN_FILE, TEST_FILE)

    # Store original test identifiers for submission
    original_test_identifiers = df_test[['Item_Identifier', 'Outlet_Identifier']].copy()

    # Create a column to identify train and test rows
    df_train['source'] = 'train'
    df_test['source'] = 'test'
    
    # Store the target variable and merge datasets for unified preprocessing
    target = 'Item_Outlet_Sales'
    train_sales = df_train[target].copy()
    df_train = df_train.drop([target], axis=1)
    
    # Combined dataset
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    print(f"Combined data shape: {df_combined.shape}")

except Exception as e:
    print("\nFATAL: Data loading failed. Please ensure the file path is correct.")
    raise

def preprocess_data(df):
    """Performs feature engineering and initial data quality checks."""
    # 1. Standardise Item_Fat_Content
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
        'LF': 'Low Fat',
        'reg': 'Regular',
        'low fat': 'Low Fat'
    })

    # 2. Fix Zero Item_Visibility
    visibility_avg = df.groupby('Item_Identifier')['Item_Visibility'].mean()
    
    def replace_zero_visibility(row):
        if row['Item_Visibility'] == 0:
            return visibility_avg.get(row['Item_Identifier'], 0.0)
        else:
            return row['Item_Visibility']

    df['Item_Visibility'] = df.apply(replace_zero_visibility, axis=1).astype(float)
    
    # 3. Create New Features
    df['Outlet_Years'] = 2025 - df['Outlet_Establishment_Year']
    
    # Get the broad category of the product
    df['Item_Type_Combined'] = df['Item_Identifier'].apply(lambda x: x[0:3]).replace({
        'FDA': 'Food', 'FDR': 'Food', 'FDN': 'Food', 'FDP': 'Food', 'FDO': 'Food',
        'FDD': 'Food', 'FDQ': 'Food', 'FDM': 'Food', 'FDJ': 'Food', 'FDH': 'Food',
        'FDL': 'Food', 'FDZ': 'Food', 'FDY': 'Food', 'FDX': 'Food', 'FDC': 'Food',
        'FDE': 'Food', 'FDF': 'Food', 'FDG': 'Food', 'FDI': 'Food', 'FDK': 'Food',
        'FDS': 'Food', 'FDT': 'Food', 'FDU': 'Food', 'FDV': 'Food', 'FDW': 'Food',
        'FDB': 'Food',
        'NCD': 'Non-Consumable', 'NCA': 'Non-Consumable', 'NCB': 'Non-Consumable',
        'NCC': 'Non-Consumable', 'NCE': 'Non-Consumable', 'NCF': 'Non-Consumable',
        'NCG': 'Non-Consumable', 'NCH': 'Non-Consumable', 'NCI': 'Non-Consumable',
        'NCJ': 'Non-Consumable', 'NCK': 'Non-Consumable', 'NCL': 'Non-Consumable',
        'NCM': 'Non-Consumable', 'NCN': 'Non-Consumable', 'NCO': 'Non-Consumable',
        'NCP': 'Non-Consumable', 'NCQ': 'Non-Consumable', 'NCR': 'Non-Consumable',
        'NCS': 'Non-Consumable', 'NCT': 'Non-Consumable', 'NCU': 'Non-Consumable',
        'NCV': 'Non-Consumable', 'NCW': 'Non-Consumable', 'NCX': 'Non-Consumable',
        'NCY': 'Non-Consumable', 'NCZ': 'Non-Consumable',
        'DRC': 'Drinks', 'DRP': 'Drinks', 'DRK': 'Drinks', 'DRB': 'Drinks',
        'DRD': 'Drinks', 'DRE': 'Drinks', 'DRF': 'Drinks', 'DRG': 'Drinks',
        'DRH': 'Drinks', 'DRI': 'Drinks', 'DRJ': 'Drinks', 'DRL': 'Drinks',
        'DRM': 'Drinks', 'DRN': 'Drinks', 'DRO': 'Drinks', 'DRQ': 'Drinks',
        'DRY': 'Drinks', 'DRZ': 'Drinks'
    })

    # Non-Consumable items have irrelevant fat content
    df.loc[df['Item_Type_Combined'] == "Non-Consumable", 'Item_Fat_Content'] = "Non-Edible"

    df = df.drop(['Outlet_Establishment_Year'], axis=1)
    
    return df

df_combined = preprocess_data(df_combined.copy())
train_sales_log = np.log1p(train_sales)

def impute_missing_values(df):
    """Imputes missing values in Item_Weight and Outlet_Size."""
    # Item_Weight imputation: Use mean weight for the same Item_Identifier
    df['Item_Weight'] = df.groupby('Item_Identifier')['Item_Weight'].transform(lambda x: x.fillna(x.mean()))
    
    # Outlet_Size imputation using Clustering
    le = LabelEncoder()
    df['Outlet_Location_Type_Encoded'] = le.fit_transform(df['Outlet_Location_Type'])
    df['Outlet_Type_Encoded'] = le.fit_transform(df['Outlet_Type'])
    
    X_cluster = df[['Outlet_Location_Type_Encoded', 'Outlet_Type_Encoded']]
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_cluster)
    
    # Calculate the mode of Outlet_Size for each Cluster and Outlet_Type combination
    size_map = df.groupby(['Cluster', 'Outlet_Type'])['Outlet_Size'].agg(lambda x: x.mode()[0] if not x.mode().empty else 'Small').to_dict()
    
    def fill_outlet_size(row):
        if pd.isnull(row['Outlet_Size']):
            try:
                return size_map[(row['Cluster'], row['Outlet_Type'])]
            except KeyError:
                return 'Medium'
        return row['Outlet_Size']

    df['Outlet_Size'] = df.apply(fill_outlet_size, axis=1)
    
    # Drop the temporary encoded columns and the cluster column
    df = df.drop(['Outlet_Location_Type_Encoded', 'Outlet_Type_Encoded', 'Cluster'], axis=1)
    
    return df

df_combined = impute_missing_values(df_combined.copy())
print(f"Missing values after imputation:\n{df_combined.isnull().sum()}")

# Identify continuous features
continuous_vars = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Years']

# 1. Skewness Fix (Apply Box-Cox to Item_Visibility)
def fix_skewness_outliers(df, features):
    """Applies transformations to reduce skewness in Item_Visibility."""
    for col in features:
        if col in ['Item_Visibility']:
            # Apply Box-Cox (requires values > 0)
            df[col], _ = boxcox(df[col] + 1e-6) 
    return df

df_combined = fix_skewness_outliers(df_combined.copy(), continuous_vars)

# 2. Identify and encode categorical features
categorical_cols = df_combined.select_dtypes(include='object').columns.tolist()
categorical_cols.remove('source')
if 'Item_Identifier' in categorical_cols:
     categorical_cols.remove('Item_Identifier')
if 'Outlet_Identifier' in categorical_cols:
     categorical_cols.remove('Outlet_Identifier')

# 3. One-Hot Encoding for Categorical Features
df_combined = pd.get_dummies(df_combined, columns=categorical_cols, drop_first=True)

# 4. Final data split BEFORE SCALING
X_train = df_combined[df_combined['source'] == 'train'].drop(['source', 'Item_Identifier', 'Outlet_Identifier'], axis=1).copy()
X_test = df_combined[df_combined['source'] == 'test'].drop(['source', 'Item_Identifier', 'Outlet_Identifier'], axis=1).copy()
y_train = train_sales_log # Using log-transformed sales

# --- 5. CORRECTED: Feature Scaling (StandardScaler) to prevent Leakage ---
print("\n--- Applying Feature Scaling (Leakage Fixed) ---")
scaler = StandardScaler()
numeric_cols_final = [col for col in continuous_vars if col in X_train.columns]

# Fit scaler only on the training data and transform X_train
X_train[numeric_cols_final] = scaler.fit_transform(X_train[numeric_cols_final])

# Transform X_test using the scaler fitted on X_train
X_test[numeric_cols_final] = scaler.transform(X_test[numeric_cols_final])
# --------------------------------------------------------------------------

# Function to calculate Actual RMSE from log-transformed predictions
def actual_rmse_scorer(estimator, X, y_log):
    """Custom scorer for cross_val_score to compute Actual RMSE."""
    y_pred_log = estimator.predict(X)
    y_pred_actual = np.expm1(y_pred_log)
    y_actual = np.expm1(y_log)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred_actual))
    return -rmse # Return negative for cross_val_score to maximize

# K-Fold Cross-Validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scoring_metric = 'neg_mean_squared_error'

## 1. XGBoost Regressor (Tuning)
print("--- Training XGBoost ---")
xgb = XGBRegressor(random_state=42)
xgb_params = {
    'n_estimators': [100, 300, 500, 800], # Reduced n_estimators for speed
    'max_depth': [3, 5, 7, 9, 11],
    'learning_rate': [0.05, 0.01, 0.1] # Reduced learning rates
}

grid_search_xgb = GridSearchCV(xgb, xgb_params, cv=kf, scoring=scoring_metric, verbose=0, n_jobs=-1)
grid_search_xgb.fit(X_train, y_train)

best_xgb = grid_search_xgb.best_estimator_
xgb_cv_score_log = np.sqrt(-grid_search_xgb.best_score_)
print(f"Best XGBoost Params: {grid_search_xgb.best_params_}")
print(f"XGBoost Log-Scale CV RMSE: {xgb_cv_score_log:.4f}")

# Calculate Actual CV RMSE for XGBoost (using the custom scorer)
xgb_cv_scores_actual = cross_val_score(best_xgb, X_train, y_train, cv=kf, scoring=actual_rmse_scorer, n_jobs=-1)
xgb_cv_score_actual = -xgb_cv_scores_actual.mean()
print(f"XGBoost Actual-Scale CV RMSE: {xgb_cv_score_actual:.4f}")

# 2. SVR and Ridge are skipped for brevity and focus on XGBoost (The previous code calculated them)
# If you need their scores, uncomment and run the original grid search blocks.

# --- 3. Generate Final Predictions using the Best XGBoost Model ---
final_model = best_xgb 

print(f"\n--- Generating CSV using the Best XGBoost Regressor (CV RMSE: {xgb_cv_score_actual:.2f}) ---")

# Make predictions on the test set (log-scale)
test_predictions_log = final_model.predict(X_test)

# Inverse-transform the predictions to the original sales scale
test_predictions_final = np.expm1(test_predictions_log)

# Set any negative predictions to 0
test_predictions_final[test_predictions_final < 0] = 0

# --- 4. Create Submission DataFrame ---
submission = pd.DataFrame({
    'Item_Identifier': original_test_identifiers['Item_Identifier'],
    'Outlet_Identifier': original_test_identifiers['Outlet_Identifier'],
    'Item_Outlet_Sales': test_predictions_final
})

# --- 5. Write Submission to the Specified Path ---
SUBMISSION_FILE_NAME = 'submission_sales_predictions_XGBoost_FixedLeakage.csv'
FINAL_SUBMISSION_PATH = os.path.join(S3_BUCKET_PATH, SUBMISSION_FILE_NAME)

print("\nSubmission Preview:")
print(submission.head())

try:
    submission.to_csv(FINAL_SUBMISSION_PATH, index=False)
    print(f"\n✅ Successfully saved submission file to: {FINAL_SUBMISSION_PATH}")
except Exception as e:
    print(f"\n❌ Error saving file to the specified path: {FINAL_SUBMISSION_PATH}")
    print(f"Error: {e}")