import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
import gc

print("Loading data...")
train = pd.read_csv('playground-series-s6e5/train.csv')
test = pd.read_csv('playground-series-s6e5/test.csv')
sub = pd.read_csv('playground-series-s6e5/sample_submission.csv')

print("Feature Engineering...")
df_full = pd.concat([train.assign(is_train=1), test.assign(is_train=0, PitNextLap=-1)], ignore_index=True)

# 1. Basic interactions & remaining progress
df_full['Remaining_Progress'] = 1.0 - df_full['RaceProgress']
df_full['TyreLife_x_LapTime'] = df_full['TyreLife'] * df_full['LapTime (s)']
df_full['TyreLife_x_Degradation'] = df_full['TyreLife'] * df_full['Cumulative_Degradation']
df_full['LapNumber_x_RaceProgress'] = df_full['LapNumber'] * df_full['RaceProgress']
df_full['LapTime_ratio_TyreLife'] = df_full['LapTime (s)'] / (df_full['TyreLife'] + 1)
df_full['Degradation_ratio_TyreLife'] = df_full['Cumulative_Degradation'] / (df_full['TyreLife'] + 1)

# 2. Race & Year Context
race_year_stats = df_full.groupby(['Race', 'Year'])[['LapTime (s)', 'Cumulative_Degradation']].transform('mean')
df_full['LapTime_vs_RaceAvg'] = df_full['LapTime (s)'] - race_year_stats['LapTime (s)']
df_full['Degradation_vs_RaceAvg'] = df_full['Cumulative_Degradation'] - race_year_stats['Cumulative_Degradation']
df_full['Position_pressure'] = df_full['Position'] / (df_full['RaceProgress'] + 0.01)

compound_stats = df_full.groupby('Compound')['TyreLife'].transform('mean')
df_full['TyreLife_vs_CompoundAvg'] = df_full['TyreLife'] - compound_stats

# 3. Rank features within the same Race, Year, and LapNumber
df_full['LapTime_Rank'] = df_full.groupby(['Race', 'Year', 'LapNumber'])['LapTime (s)'].rank(pct=True)
df_full['Degradation_Rank'] = df_full.groupby(['Race', 'Year', 'LapNumber'])['Cumulative_Degradation'].rank(pct=True)

# 4. Grouped Statistics
driver_stats = df_full.groupby('Driver')[['LapTime (s)', 'Cumulative_Degradation', 'TyreLife']].transform('mean')
df_full['Driver_mean_LapTime'] = driver_stats['LapTime (s)']
df_full['Driver_mean_Degradation'] = driver_stats['Cumulative_Degradation']
df_full['Driver_mean_TyreLife'] = driver_stats['TyreLife']

race_stats = df_full.groupby('Race')[['LapTime (s)', 'Cumulative_Degradation', 'TyreLife']].transform('mean')
df_full['Race_mean_LapTime'] = race_stats['LapTime (s)']
df_full['Race_mean_Degradation'] = race_stats['Cumulative_Degradation']
df_full['Race_mean_TyreLife'] = race_stats['TyreLife']

# 5. Categorical Encoding (OHE for compound)
df_full = pd.get_dummies(df_full, columns=['Compound'], drop_first=True)

# Split back before target encoding to avoid leakage
train_fe = df_full[df_full['is_train'] == 1].drop(['is_train'], axis=1).copy()
test_fe = df_full[df_full['is_train'] == 0].drop(['is_train', 'PitNextLap'], axis=1).copy()

# 6. Target Encoding (K-Fold out-of-fold to avoid leakage)
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

for col in ['Driver', 'Race']:
    # Initialize features
    train_fe[f'{col}_TargetEnc'] = 0.0
    test_fe[f'{col}_TargetEnc'] = 0.0

    # Calculate global mean for smoothing/imputing
    global_mean = train_fe['PitNextLap'].mean()

    for train_idx, val_idx in skf.split(train_fe, train_fe['PitNextLap']):
        X_tr = train_fe.iloc[train_idx]

        # Calculate target mean on training fold
        target_mean = X_tr.groupby(col)['PitNextLap'].mean()

        # Map to validation fold
        train_fe.loc[train_fe.index[val_idx], f'{col}_TargetEnc'] = train_fe.loc[train_fe.index[val_idx], col].map(target_mean)

    # Map to test using whole training data
    test_target_mean = train_fe.groupby(col)['PitNextLap'].mean()
    test_fe[f'{col}_TargetEnc'] = test_fe[col].map(test_target_mean)

    # Impute missing values with global mean (if new categories appear in test or no data in fold)
    train_fe[f'{col}_TargetEnc'].fillna(global_mean, inplace=True)
    test_fe[f'{col}_TargetEnc'].fillna(global_mean, inplace=True)

# Also keep frequency encoding as it's safe and useful
for col in ['Race', 'Driver']:
    freq = train[col].value_counts() / len(train)
    train_fe[f'{col}_freq'] = train_fe[col].map(freq).fillna(0)
    test_fe[f'{col}_freq'] = test_fe[col].map(freq).fillna(0)

train_fe.drop(['Race', 'Driver'], axis=1, inplace=True)
test_fe.drop(['Race', 'Driver'], axis=1, inplace=True)


features = [c for c in train_fe.columns if c not in ['id', 'PitNextLap']]
X = train_fe[features]
y = train_fe['PitNextLap'].astype(int)
X_test = test_fe[features]

print(f"Number of features: {len(features)}")

# --- MODELING ---
oof_xgb = np.zeros(len(train_fe))
oof_lgb = np.zeros(len(train_fe))
oof_cat = np.zeros(len(train_fe))

test_preds_xgb = np.zeros(len(test_fe))
test_preds_lgb = np.zeros(len(test_fe))
test_preds_cat = np.zeros(len(test_fe))

print(f"\nStarting {N_FOLDS}-Fold Cross Validation...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold+1} ---")
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

    # XGBoost
    xgb_model = XGBClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=7, subsample=0.8,
        colsample_bytree=0.8, random_state=42+fold, use_label_encoder=False,
        eval_metric='logloss', early_stopping_rounds=50
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=0)
    oof_xgb[val_idx] = xgb_model.predict_proba(X_va)[:, 1]
    test_preds_xgb += xgb_model.predict_proba(X_test)[:, 1] / N_FOLDS
    print(f"XGBoost Fold {fold+1} Log Loss: {log_loss(y_va, oof_xgb[val_idx]):.5f}")

    # LightGBM
    lgb_model = LGBMClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=7, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, random_state=42+fold, objective='binary', n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
    oof_lgb[val_idx] = lgb_model.predict_proba(X_va)[:, 1]
    test_preds_lgb += lgb_model.predict_proba(X_test)[:, 1] / N_FOLDS
    print(f"LightGBM Fold {fold+1} Log Loss: {log_loss(y_va, oof_lgb[val_idx]):.5f}")

    # CatBoost
    cat_model = CatBoostClassifier(
        iterations=1000, learning_rate=0.03, depth=7, random_seed=42+fold,
        eval_metric='Logloss', early_stopping_rounds=50, verbose=0,
        subsample=0.8
    )
    cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    oof_cat[val_idx] = cat_model.predict_proba(X_va)[:, 1]
    test_preds_cat += cat_model.predict_proba(X_test)[:, 1] / N_FOLDS
    print(f"CatBoost Fold {fold+1} Log Loss: {log_loss(y_va, oof_cat[val_idx]):.5f}")

# --- EVALUATION AND STACKING ---
print("\n=== BASE MODELS OOF SCORES ===")
print(f"XGBoost OOF Log Loss: {log_loss(y, oof_xgb):.5f}")
print(f"LightGBM OOF Log Loss: {log_loss(y, oof_lgb):.5f}")
print(f"CatBoost OOF Log Loss: {log_loss(y, oof_cat):.5f}")

print("\n--- Training Meta-Model (Stacking) ---")
# Create meta-features
S_train = np.column_stack((oof_xgb, oof_lgb, oof_cat))
S_test = np.column_stack((test_preds_xgb, test_preds_lgb, test_preds_cat))

# Meta model
meta_model = LogisticRegression(random_state=42)
meta_model.fit(S_train, y)

# Meta predictions
oof_stack = meta_model.predict_proba(S_train)[:, 1]
test_stack = meta_model.predict_proba(S_test)[:, 1]

print(f"Stacking OOF Log Loss: {log_loss(y, oof_stack):.5f} | AUC: {roc_auc_score(y, oof_stack):.5f}")

# Meta model weights
print("Meta-Model Weights (Coefficients):")
print(f"XGBoost: {meta_model.coef_[0][0]:.4f}")
print(f"LightGBM: {meta_model.coef_[0][1]:.4f}")
print(f"CatBoost: {meta_model.coef_[0][2]:.4f}")

# --- SUBMISSION ---
print("\nSaving stacked ensemble prediction...")
sub['PitNextLap'] = test_stack
sub.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")
