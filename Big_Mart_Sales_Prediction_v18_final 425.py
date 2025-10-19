import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import warnings
from optuna.samplers import TPESampler
import time

# --- Configuration Section ---
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

							   
S3_BUCKET_PATH = 's3://useast1-nlsn-dscihdamrpd-zoo-archive/users/vipulverma/Prediction_Tests_CSVs/'
TRAIN_FILE = 'train_v9rqX0R.csv'
TEST_FILE = 'test_AbJTz2l.csv'
SUBMISSION_FILE = 'submission_sfs_optimized.csv'
CURRENT_YEAR = 2013
TARGET_VARIABLE = 'Item_Outlet_Sales'
ID_COLS = ['Item_Identifier', 'Outlet_Identifier']
RANDOM_STATE = 42
N_FOLDS = 5
N_OPTUNA_TRIALS = 100 # Reduced for a quicker demonstration; 100 is good for thoroughness

# --- Data Loading and Preprocessing Functions (Unchanged) ---

def load_data(bucket_path, train_file, test_file):
    """Loads training and testing data."""
    print("Step 1: Loading data...")
    try:
        train_df = pd.read_csv(bucket_path + train_file)
        test_df = pd.read_csv(bucket_path + test_file)
        return train_df, test_df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

						   
										 

def preprocess_and_feature_engineer(train_df, test_df):
    """Applies all preprocessing and feature engineering steps without data leakage."""
    print("\nStep 2: Starting preprocessing and feature engineering...")
    train_processed = train_df.copy()
    test_processed = test_df.copy()

    # --- Basic Cleaning and Feature Creation ---
    for df in [train_processed, test_processed]:
        df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})
        df['Item_Category_Raw'] = df['Item_Identifier'].apply(lambda x: x[0:2])
        df['Item_Category'] = df['Item_Category_Raw'].map({'FD': 'Food', 'DR': 'Drinks', 'NC': 'Non-Consumable'})
        df.loc[df['Item_Category'] == 'Non-Consumable', 'Item_Fat_Content'] = 'Non-Edible'
        df['Outlet_Age'] = CURRENT_YEAR - df['Outlet_Establishment_Year']
    
    # --- Missing Value Imputation ---
    print("Imputing missing values...")
    train_processed['Outlet_Size'].fillna('Small', inplace=True)
    test_processed['Outlet_Size'].fillna('Small', inplace=True)
    item_avg_weight = train_processed.pivot_table(values='Item_Weight', index='Item_Identifier')
    train_processed['Item_Weight'].fillna(train_processed['Item_Identifier'].map(item_avg_weight['Item_Weight']), inplace=True)
    test_processed['Item_Weight'].fillna(test_processed['Item_Identifier'].map(item_avg_weight['Item_Weight']), inplace=True)
    item_type_avg_weight = train_processed.groupby('Item_Type')['Item_Weight'].transform('mean')
    train_processed['Item_Weight'].fillna(item_type_avg_weight, inplace=True)
    test_processed['Item_Weight'].fillna(test_processed.groupby('Item_Type')['Item_Weight'].transform('mean'), inplace=True)
    train_processed['Item_Visibility'].replace(0, np.nan, inplace=True)
    test_processed['Item_Visibility'].replace(0, np.nan, inplace=True)
    visibility_avg = train_processed.groupby(['Outlet_Identifier', 'Item_Type'])['Item_Visibility'].transform('mean')
    train_processed['Item_Visibility'].fillna(visibility_avg, inplace=True)
    test_processed['Item_Visibility'].fillna(test_processed.groupby(['Outlet_Identifier', 'Item_Type'])['Item_Visibility'].transform('mean'), inplace=True)
    train_processed['Item_Visibility'].fillna(train_processed.groupby('Item_Type')['Item_Visibility'].transform('mean'), inplace=True)
    test_processed['Item_Visibility'].fillna(test_processed.groupby('Item_Type')['Item_Visibility'].transform('mean'), inplace=True)

    # --- Advanced Feature Engineering ---
    print("Creating advanced context-based features...")
    item_counts = train_processed['Item_Identifier'].value_counts()
    train_processed['Item_Popularity'] = train_processed['Item_Identifier'].map(item_counts)
    test_processed['Item_Popularity'] = test_processed['Item_Identifier'].map(item_counts).fillna(item_counts.median())
    outlet_stats = train_processed.groupby('Outlet_Identifier').agg({'Item_Identifier': 'nunique', TARGET_VARIABLE: 'mean'}).rename(columns={'Item_Identifier': 'Outlet_Item_Count', TARGET_VARIABLE: 'Outlet_Mean_Sales'})
    train_processed = train_processed.merge(outlet_stats, on='Outlet_Identifier', how='left')
    test_processed = test_processed.merge(outlet_stats, on='Outlet_Identifier', how='left')
    test_processed['Outlet_Item_Count'].fillna(outlet_stats['Outlet_Item_Count'].median(), inplace=True)
    test_processed['Outlet_Mean_Sales'].fillna(outlet_stats['Outlet_Mean_Sales'].mean(), inplace=True)

    # --- "Super" Advanced Features ---
    print("Creating 'wild' context-based features...")
    item_first_appearance = train_processed.groupby('Item_Identifier')['Outlet_Establishment_Year'].min().reset_index().rename(columns={'Outlet_Establishment_Year': 'Item_First_Appearance_Year'})
    train_processed = train_processed.merge(item_first_appearance, on='Item_Identifier', how='left')
    test_processed = test_processed.merge(item_first_appearance, on='Item_Identifier', how='left')
    train_processed['Item_Age_Proxy'] = CURRENT_YEAR - train_processed['Item_First_Appearance_Year']
    test_processed['Item_Age_Proxy'] = CURRENT_YEAR - test_processed['Item_First_Appearance_Year']
    test_processed['Item_Age_Proxy'].fillna(train_processed['Item_Age_Proxy'].median(), inplace=True)
    outlet_category_dist = train_processed.groupby(['Outlet_Identifier', 'Item_Category'])['Item_Identifier'].nunique().unstack(fill_value=0)
    outlet_category_dist['Total_Unique_Items'] = outlet_category_dist.sum(axis=1)
    outlet_category_dist['Outlet_Food_Ratio'] = outlet_category_dist['Food'] / outlet_category_dist['Total_Unique_Items']
    outlet_category_dist['Outlet_Drinks_Ratio'] = outlet_category_dist['Drinks'] / outlet_category_dist['Total_Unique_Items']
    train_processed = train_processed.merge(outlet_category_dist[['Outlet_Food_Ratio', 'Outlet_Drinks_Ratio']], on='Outlet_Identifier', how='left')
    test_processed = test_processed.merge(outlet_category_dist[['Outlet_Food_Ratio', 'Outlet_Drinks_Ratio']], on='Outlet_Identifier', how='left')
    test_processed[['Outlet_Food_Ratio', 'Outlet_Drinks_Ratio']] = test_processed[['Outlet_Food_Ratio', 'Outlet_Drinks_Ratio']].fillna(train_processed[['Outlet_Food_Ratio', 'Outlet_Drinks_Ratio']].median())
    avg_vis_in_outlet_by_type = train_processed.groupby(['Outlet_Identifier', 'Item_Type'])['Item_Visibility'].mean().reset_index().rename(columns={'Item_Visibility': 'Avg_Vis_Type_in_Outlet'})
    train_processed = train_processed.merge(avg_vis_in_outlet_by_type, on=['Outlet_Identifier', 'Item_Type'], how='left')
    test_processed = test_processed.merge(avg_vis_in_outlet_by_type, on=['Outlet_Identifier', 'Item_Type'], how='left')
    train_processed['Relative_Visibility'] = train_processed['Item_Visibility'] / train_processed['Avg_Vis_Type_in_Outlet']
    test_processed['Relative_Visibility'] = test_processed['Item_Visibility'] / test_processed['Avg_Vis_Type_in_Outlet']
    test_processed['Relative_Visibility'].fillna(1, inplace=True)
    train_processed['Relative_Visibility'].fillna(1, inplace=True)
    loc_cat_sales = train_processed.groupby(['Outlet_Location_Type', 'Item_Category'])[TARGET_VARIABLE].mean().reset_index().rename(columns={TARGET_VARIABLE: 'Loc_Cat_Sales_Mean'})
    train_processed = train_processed.merge(loc_cat_sales, on=['Outlet_Location_Type', 'Item_Category'], how='left')
    test_processed = test_processed.merge(loc_cat_sales, on=['Outlet_Location_Type', 'Item_Category'], how='left')
    test_processed['Loc_Cat_Sales_Mean'].fillna(train_df[TARGET_VARIABLE].mean(), inplace=True)

    # --- "Crazy" Interaction and Ratio Features ---
    print("Creating 'crazy' interaction and ratio features...")

									 
    train_processed['Price_per_Unit_Weight'] = train_processed['Item_MRP'] / (train_processed['Item_Weight'] + 1e-6)
    test_processed['Price_per_Unit_Weight'] = test_processed['Item_MRP'] / (test_processed['Item_Weight'] + 1e-6)
	
												 
    outlet_avg_mrp = train_processed.groupby('Outlet_Identifier')['Item_MRP'].mean().reset_index().rename(columns={'Item_MRP': 'Outlet_Mean_MRP'})
    train_processed = train_processed.merge(outlet_avg_mrp, on='Outlet_Identifier', how='left')
    test_processed = test_processed.merge(outlet_avg_mrp, on='Outlet_Identifier', how='left')
    test_processed['Outlet_Mean_MRP'].fillna(train_processed['Outlet_Mean_MRP'].median(), inplace=True)
    train_processed['Item_MRP_to_Outlet_Mean_MRP_Ratio'] = train_processed['Item_MRP'] / train_processed['Outlet_Mean_MRP']
    test_processed['Item_MRP_to_Outlet_Mean_MRP_Ratio'] = test_processed['Item_MRP'] / test_processed['Outlet_Mean_MRP']

    # --- CORRECTED (LEAKAGE-SAFE) TARGET ENCODING ---
    print("Creating leakage-safe target encoded features...")
										  
																							
    global_saleability = train_processed[TARGET_VARIABLE].mean() / (train_processed['Item_MRP'].mean() + 1e-6)

												 
    train_processed['Item_Saleability_Ratio'] = np.nan
	
																	  
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, val_idx in kf.split(train_processed):
															 
        train_fold = train_processed.iloc[train_idx]
        item_saleability_fold = train_fold.groupby('Item_Identifier').agg({TARGET_VARIABLE: 'mean', 'Item_MRP': 'mean'}).reset_index()
        item_saleability_fold['Item_Saleability_Ratio'] = item_saleability_fold[TARGET_VARIABLE] / (item_saleability_fold['Item_MRP'] + 1e-6)
		
																
        val_fold = train_processed.iloc[val_idx]
        merged_val = val_fold[['Item_Identifier']].merge(item_saleability_fold[['Item_Identifier', 'Item_Saleability_Ratio']], on='Item_Identifier', how='left')
        train_processed.loc[val_idx, 'Item_Saleability_Ratio'] = merged_val['Item_Saleability_Ratio'].values

																									  
    train_processed['Item_Saleability_Ratio'].fillna(global_saleability, inplace=True)
		
																			 
    item_saleability_full = train_processed.groupby('Item_Identifier').agg({TARGET_VARIABLE: 'mean', 'Item_MRP': 'mean'}).reset_index()
    item_saleability_full['Item_Saleability_Ratio'] = item_saleability_full[TARGET_VARIABLE] / (item_saleability_full['Item_MRP'] + 1e-6)
    test_processed = test_processed.merge(item_saleability_full[['Item_Identifier', 'Item_Saleability_Ratio']], on='Item_Identifier', how='left')
	
															
    test_processed['Item_Saleability_Ratio'].fillna(global_saleability, inplace=True)
															  

    # --- Final Engineered Feature ---
    train_processed['Outlet_to_Item_Age_Ratio'] = train_processed['Outlet_Age'] / (train_processed['Item_Age_Proxy'] + 1)
    test_processed['Outlet_to_Item_Age_Ratio'] = test_processed['Outlet_Age'] / (test_processed['Item_Age_Proxy'] + 1)

    # --- Final Cleanup and Encoding ---
    cols_to_drop = ['Outlet_Establishment_Year', 'Item_First_Appearance_Year', 'Avg_Vis_Type_in_Outlet', 'Item_Category_Raw', 'Outlet_Mean_MRP']
    train_processed.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    test_processed.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    for col in train_processed.select_dtypes(include=['object']).columns:
        if col not in ID_COLS:
            le = LabelEncoder()
            train_processed[col] = le.fit_transform(train_processed[col].astype(str))
            test_processed[col] = test_processed[col].astype(str).map(lambda s: '<unknown>' if s not in le.classes_ else s)
            le.classes_ = np.append(le.classes_, '<unknown>')
            test_processed[col] = le.transform(test_processed[col])

    print("Preprocessing and feature engineering complete.")
    return train_processed, test_processed

# --- Model Training, Tuning, and Submission Functions (Unchanged) ---
def optimize_hyperparameters(train_data, predictors):
    """Uses Optuna to find the best hyperparameters for the XGBoost model."""
    print("\nStep 3: Starting XGBoost Hyperparameter Tuning...")
    y, X = train_data[TARGET_VARIABLE], train_data[predictors]
    dtrain = xgb.DMatrix(X, label=y)
    
    def objective(trial):
        params = {
            'objective': 'reg:squarederror', 'n_jobs': -1, 'seed': RANDOM_STATE,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
        }
        cv_results = xgb.cv(params, dtrain, 1000, N_FOLDS, early_stopping_rounds=50, metrics={'rmse'}, seed=RANDOM_STATE, verbose_eval=False)
        return cv_results['test-rmse-mean'].min()

    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS)
    print(f"Best trial RMSE from Optuna: {study.best_value:.4f}")
    return study.best_params

def train_xgboost_model(train_data, predictors, params):
    """Trains the final XGBoost model and returns the model and its score."""
												   
    y, X = train_data[TARGET_VARIABLE], train_data[predictors]
    dtrain = xgb.DMatrix(X, label=y)
    
    # We use CV to find the best number of rounds and get a reliable score
    cv_results = xgb.cv(params, dtrain, 1000, N_FOLDS, early_stopping_rounds=100, metrics={'rmse'}, seed=RANDOM_STATE, verbose_eval=False)
    best_boost_round = cv_results['test-rmse-mean'].idxmin() + 1
    best_rmse_mean = cv_results['test-rmse-mean'].min()
																				   
    
    # We only train the final model when we are ready to generate submissions
    final_model = xgb.train(params, dtrain, num_boost_round=best_boost_round)
    return final_model, best_rmse_mean

def generate_submission(model, test_data, test_ids, predictors, model_name):
    """Generates predictions and saves the submission file."""
    print(f"\n--- Generating predictions with the final model: {model_name} ---")
										   
	
    dtest = xgb.DMatrix(test_data[predictors])
    predictions = model.predict(dtest)
		
    predictions = np.clip(predictions, 0, None)
    
    submission_df = pd.DataFrame({
        'Item_Identifier': test_ids['Item_Identifier'],
        'Outlet_Identifier': test_ids['Outlet_Identifier'],
        TARGET_VARIABLE: predictions
    })
    
    print("--- Submission Dataframe Head ---")
    # print(submission_df.head())
    # submission_df.to_csv(SUBMISSION_FILE, index=False)
    # print(f"\nâœ… Submission file saved as '{SUBMISSION_FILE}'")
    submission_df.display()


# --- NEW: Sequential Forward Selection Function ---
def perform_sfs(train_data, base_predictors, engineered_features, model_params):
    """Performs Sequential Forward Selection to find the optimal feature set."""
    print("\nStep 4: ðŸš€ Starting Sequential Forward Selection (SFS)...")
    start_time = time.time()
    
    selected_features = base_predictors.copy()
    remaining_features = engineered_features.copy()
    
    # Use a simplified training function for evaluation inside the loop
    def evaluate_features(predictors):
        y, X = train_data[TARGET_VARIABLE], train_data[predictors]
        dtrain = xgb.DMatrix(X, label=y)
        cv_results = xgb.cv(model_params, dtrain, 1000, N_FOLDS, early_stopping_rounds=50, metrics={'rmse'}, seed=RANDOM_STATE, verbose_eval=False)
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
            print(f"  -> Testing '{feature}': CV RMSE = {current_rmse:.4f}")

        best_feature_this_round, best_rmse_this_round = min(results_this_round.items(), key=lambda item: item[1])

        if best_rmse_this_round < best_overall_rmse:
            print(f"\nâœ… Improvement Found! Adding '{best_feature_this_round}'.")
            print(f"   New Best RMSE: {best_rmse_this_round:.4f} (Old: {best_overall_rmse:.4f})\n")
            best_overall_rmse = best_rmse_this_round
            selected_features.append(best_feature_this_round)
            remaining_features.remove(best_feature_this_round)
        else:
            print("\nâ¹ï¸ No further improvement found. Stopping SFS.")
            break
        iteration += 1

    end_time = time.time()
    print(f"\n--- SFS Complete in {end_time - start_time:.2f} seconds ---")
    return best_overall_rmse, selected_features


# --- Main Execution Block ---
if __name__ == '__main__':
    train_df, test_df = load_data(S3_BUCKET_PATH, TRAIN_FILE, TEST_FILE)
    
    if train_df is not None and test_df is not None:
        original_test_df_ids = test_df[ID_COLS].copy()
        
        train_processed, test_processed = preprocess_and_feature_engineer(train_df, test_df)
        
        # Define the engineered features we want to test
        engineered_features = [
            'Item_Popularity', 'Outlet_Mean_Sales','Outlet_Item_Count', 'Item_Age_Proxy', 'Outlet_Food_Ratio', 'Outlet_Drinks_Ratio', 
            'Relative_Visibility', 'Loc_Cat_Sales_Mean', 'Price_per_Unit_Weight', 
            'Item_MRP_to_Outlet_Mean_MRP_Ratio', 'Item_Saleability_Ratio', 
            'Outlet_to_Item_Age_Ratio'
        ]
        
        # Define base predictors (all columns that are not target, id, or engineered)
        all_predictors = [col for col in train_processed.columns if col not in [TARGET_VARIABLE] + ID_COLS]
        base_predictors = [p for p in all_predictors if p not in engineered_features]

        # First, find the best hyperparameters using ALL features
        xgb_base_params = {'objective': 'reg:squarederror', 'seed': RANDOM_STATE, 'n_jobs': -1}
        best_xgb_params = optimize_hyperparameters(train_processed, all_predictors)
        xgb_base_params.update(best_xgb_params)
																									 
        
        # Next, run SFS to find the optimal subset of features
        optimal_rmse, optimal_features = perform_sfs(
            train_processed, 
            base_predictors, 
            engineered_features, 
            xgb_base_params
        )
        
        # Display the results from SFS
        print("\n--- ðŸ† Final SFS Results ---")
        print(f"Optimal CV RMSE achieved: {optimal_rmse:.4f}")
        print(f"Total number of optimal features: {len(optimal_features)}")
        added_features = [f for f in optimal_features if f in engineered_features]
        print(f"\nEngineered features selected by SFS ({len(added_features)}):")
        print(added_features)
        
        # Finally, train the single best model on the optimal feature set
        print("\nStep 5: Training final model with the optimal feature set...")
        final_model, final_rmse = train_xgboost_model(train_processed, optimal_features, xgb_base_params)
        print(f"Final model trained. CV RMSE: {final_rmse:.4f}")

        # Generate the submission file
        generate_submission(final_model, test_processed, original_test_df_ids, optimal_features, "XGBoost_SFS_Optimized")