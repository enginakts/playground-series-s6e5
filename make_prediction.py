import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
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

# 3. NEW: Rank features within the same Race, Year, and LapNumber
# This shows how a driver is performing compared to other drivers *exactly right now* on this lap.
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

# 5. Categorical Encoding
df_full = pd.get_dummies(df_full, columns=['Compound'], drop_first=True)

for col in ['Race', 'Driver']:
    freq = train[col].value_counts() / len(train)
    df_full[f'{col}_freq'] = df_full[col].map(freq).fillna(0)

df_full.drop(['Race', 'Driver'], axis=1, inplace=True)

# Split back
train_fe = df_full[df_full['is_train'] == 1].drop(['is_train'], axis=1).copy()
test_fe = df_full[df_full['is_train'] == 0].drop(['is_train', 'PitNextLap'], axis=1).copy()

features = [c for c in train_fe.columns if c not in ['id', 'PitNextLap']]

X = train_fe[features]
y = train_fe['PitNextLap'].astype(int)
X_test = test_fe[features]

print(f"Number of features: {len(features)}")

# --- STRATIFIED K-FOLD CV ---
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

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

    # 1. XGBoost
    xgb_model = XGBClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=7, subsample=0.8,
        colsample_bytree=0.8, random_state=42+fold, use_label_encoder=False,
        eval_metric='logloss', early_stopping_rounds=50
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=0)
    oof_xgb[val_idx] = xgb_model.predict_proba(X_va)[:, 1]
    test_preds_xgb += xgb_model.predict_proba(X_test)[:, 1] / N_FOLDS
    print(f"XGBoost Fold {fold+1} Log Loss: {log_loss(y_va, oof_xgb[val_idx]):.5f}")

    # 2. LightGBM
    lgb_model = LGBMClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=7, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, random_state=42+fold, objective='binary', n_jobs=-1
    )
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)]) # No easy early stopping without callbacks hack, so let it run
    oof_lgb[val_idx] = lgb_model.predict_proba(X_va)[:, 1]
    test_preds_lgb += lgb_model.predict_proba(X_test)[:, 1] / N_FOLDS
    print(f"LightGBM Fold {fold+1} Log Loss: {log_loss(y_va, oof_lgb[val_idx]):.5f}")

    # 3. CatBoost
    cat_model = CatBoostClassifier(
        iterations=1000, learning_rate=0.03, depth=7, random_seed=42+fold,
        eval_metric='Logloss', early_stopping_rounds=50, verbose=0,
        subsample=0.8
    )
    cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    oof_cat[val_idx] = cat_model.predict_proba(X_va)[:, 1]
    test_preds_cat += cat_model.predict_proba(X_test)[:, 1] / N_FOLDS
    print(f"CatBoost Fold {fold+1} Log Loss: {log_loss(y_va, oof_cat[val_idx]):.5f}")

# --- EVALUATION ---
print("\n=== OVERALL OOF SCORES ===")
oof_ensemble = (oof_xgb * 0.4) + (oof_lgb * 0.3) + (oof_cat * 0.3)
print(f"XGBoost OOF Log Loss: {log_loss(y, oof_xgb):.5f} | AUC: {roc_auc_score(y, oof_xgb):.5f}")
print(f"LightGBM OOF Log Loss: {log_loss(y, oof_lgb):.5f} | AUC: {roc_auc_score(y, oof_lgb):.5f}")
print(f"CatBoost OOF Log Loss: {log_loss(y, oof_cat):.5f} | AUC: {roc_auc_score(y, oof_cat):.5f}")
print(f"Ensemble OOF Log Loss: {log_loss(y, oof_ensemble):.5f} | AUC: {roc_auc_score(y, oof_ensemble):.5f}")

# --- SUBMISSION ---
print("\nSaving final ensemble prediction...")
ensemble_preds_test = (test_preds_xgb * 0.4) + (test_preds_lgb * 0.3) + (test_preds_cat * 0.3)
sub['PitNextLap'] = ensemble_preds_test
sub.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")
