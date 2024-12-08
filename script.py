import time

import optuna
import pandas as pd
import numpy as np
import os
import shap
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier


RANDOM_SEED = 42

DATA_PATH = "data"
PREDICTION_PATH = "predictions"

# DEVICE = "CPU"
DEVICE = "GPU"
# DEVICE_XGB = "cpu"
DEVICE_XGB = "cuda"
DEVICE_VERY_SMALL = "CPU"
DEVICE_LIGHTGBM = "cpu"


DATA_SIZE_THRESHOLDS = (1000, 100_000, 1_000_000)
DATA_SMALL_CLASS_THRESHOLD = 30


def load_data(data_paths: list[str]) -> pd.DataFrame:
    data = pd.read_parquet(data_paths[0])
    return data


def train_preprocess(
    train_paths: list[str],
    nan_threshold=0.98,
    corr_sample=500000, corr_threshold=0.98
):
    data = load_data(train_paths)
    data = data.reset_index(drop=True)
    print(f"-- Data has shape: {data.shape}")

    data = data.drop(['smpl', 'id'], axis=1, errors='ignore')
    print("-- Dropped smpl and id")

    # Change float64 to float16
    float_cols = list(data.loc[:, data.dtypes == 'float64'].columns)
    data[float_cols] = data[float_cols].astype('float16')   # float32?
    print("-- Converted columns to float16")

    # Remove duplicates
    tmp = data.duplicated().sum()
    if tmp > 0:
        data = data.drop_duplicates().reset_index(drop=True)
        print(f"-- Removed {tmp} duplicates")
    else:
        print('-- no duplicates')

    target = data[['target']].astype('int8')
    data = data.drop(['target'], axis=1, errors='ignore')
    print("-- Dropped target")

    # Remove high nan cols
    if data.isna().sum().sum() > 0:
        na_threshold = len(data) * nan_threshold
        tmp = data.shape[1]
        data = data.loc[:, data.isna().sum() < na_threshold]
        tmp -= data.shape[1]
        print(f"-- Removed {tmp} high nan cols")
    else:
        print('-- no nan cols')

    # Remove high corr cols
    tmp = data.shape[1]
    if len(data) > corr_sample:
        corr_matrix = data.sample(corr_sample).corr(numeric_only=True).abs()
    else:
        corr_matrix = data.corr(numeric_only=True).abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    data = data.drop(to_drop, axis=1, errors='ignore')
    tmp -= data.shape[1]

    if tmp == 0:
        print('-- no high corr cols')
    else:
        print(f"-- Removed {tmp} high corr cols")

    # Change cat dtypes
    n_un_df = data.nunique().reset_index()
    n_un_df.columns = ['col', 'n_un']

    bin_cols = list(n_un_df[n_un_df['n_un'] == 2]['col'])
    cat_length = min(1000, int(len(data) * 0.2))
    cat_cols = list(n_un_df[(n_un_df['n_un'] > 2) & (n_un_df['n_un'] < cat_length)]['col'])

    data[bin_cols] = data[bin_cols].astype(bool)
    data[cat_cols] = data[cat_cols].astype('int16')
    print("-- Converted categorical cols to needed types")

    # OHE
    tmp = data.shape[1]
    data = pd.get_dummies(data, columns=cat_cols)
    tmp = data.shape[1] - tmp
    print(f"-- Added {tmp} cols after OneHotEncoding")
    print(f'-- Total amount of cols {data.shape[1]}')

    res = {
        "data": data,
        "target": target,
        "cat_cols": cat_cols
    }

    return res


def feature_selection(
    data: pd.DataFrame, y: pd.Series,
    max_features: int = 500
) -> pd.DataFrame:
    data_size = data.shape[0]
    number_of_cols = data.shape[1]

    if data_size < DATA_SIZE_THRESHOLDS[0] or y.value_counts().min() < DATA_SMALL_CLASS_THRESHOLD:
        data_size_type = "very_small"
    else:
        data_size_type = "other"

    if data_size_type == "very_small":
        print("** Feature selection is not needed...")
        return data

    y = y.values.ravel()
    x_train, x_valid, y_train, y_valid = train_test_split(
        data, y, test_size=0.1, stratify=y, random_state=RANDOM_SEED
    )

    print("** Fitting model for feature selection with early_stopping...")
    model = CatBoostClassifier(
        learning_rate=0.1, iterations=10000, eval_metric='AUC', early_stopping_rounds=50,
        border_count=254, verbose=False, task_type=DEVICE, random_state=RANDOM_SEED
    )
    model.fit(x_train, y_train, eval_set=(x_valid, y_valid))

    print("** Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_valid)

    imp = pd.DataFrame({'name': x_valid.columns, 'shap_value': np.mean(abs(shap_values), axis=0)})
    imp = imp.sort_values('shap_value', ascending=False)
    imp['shap_value'] /= imp['shap_value'].sum()

    print("** Selecting top features...")
    top_features = list(imp[imp['shap_value'] > 1e-5].head(max_features)['name'])

    data = data[top_features]

    number_of_cols -= data.shape[1]
    print(f'** Deleted {number_of_cols} cols after feature selection')
    print(f'** Total amount of cols after feature selection {data.shape[1]}')

    return data


def train(X: pd.DataFrame, y: pd.Series) -> dict:
    data_size = X.shape[0]
    ts = 0.2

    if data_size < DATA_SIZE_THRESHOLDS[0] or y.value_counts().min() < DATA_SMALL_CLASS_THRESHOLD:
        data_size_type = "very_small"
    elif data_size < DATA_SIZE_THRESHOLDS[1]:
        data_size_type = "small"
    elif data_size < DATA_SIZE_THRESHOLDS[2]:
        data_size_type = "medium"
    else:
        ts = 0.1
        data_size_type = "huge"

    X_mean = None
    X_std = None

    y = y.values.ravel()
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y)

    if data_size_type == "very_small":
        X_values = X.to_numpy(na_value=0.0, dtype=np.float32)
        X_mean = np.mean(X_values, axis=0)
        X_std = np.std(X_values, axis=0) + 1e-8

        X_values = (X_values - X_mean) / X_std

        model_svm = SVC(
            max_iter=10000, tol=1e-7, C=1e-5, kernel='linear', class_weight="balanced",
            probability=True, verbose=False, random_state=RANDOM_SEED
        )
        model_svm.fit(X_values, y)

        model_logreg = LogisticRegression(
            max_iter=3000, C=1e-5, solver='liblinear', class_weight="balanced",
            verbose=False, random_state=RANDOM_SEED, n_jobs=-1
        )
        model_logreg.fit(X_values, y)

        model_random_forest = RandomForestClassifier(
            n_estimators=300, n_jobs=-1, class_weight="balanced",
            verbose=False, random_state=RANDOM_SEED
        )
        model_random_forest.fit(X, y)

        model_lightgbm = LGBMClassifier(
            learning_rate=0.1, device_type=DEVICE_LIGHTGBM, is_unbalance=True,
            verbose=-1, random_state=RANDOM_SEED, n_jobs=-1
        )
        model_lightgbm.fit(X, y)

        model_xgboost = XGBClassifier(
            learning_rate=0.1, device=DEVICE_XGB,
            scale_pos_weight=class_weights[1] / class_weights[0],
            verbosity=0, random_state=RANDOM_SEED, n_jobs=-1
        )
        model_xgboost.fit(X, y)

        model_catboost = CatBoostClassifier(
            learning_rate=0.1, task_type=DEVICE_VERY_SMALL,
            verbose=False, random_state=RANDOM_SEED
        )
        model_catboost.fit(X, y)

        model_catboost_x2 = CatBoostClassifier(
            learning_rate=0.09, task_type=DEVICE_VERY_SMALL, verbose=False, random_state=RANDOM_SEED + 1
        )
        model_catboost_x2.fit(X, y)

        model_catboost_x3 = CatBoostClassifier(
            learning_rate=0.08, task_type=DEVICE_VERY_SMALL, verbose=False, random_state=RANDOM_SEED + 2
        )
        model_catboost_x3.fit(X, y)

        models = {
            "svm": model_svm,
            "logreg": model_logreg,
            "random_forest": model_random_forest,
            "lightgbm": model_lightgbm,
            "xgboost": model_xgboost,
            "catboost": model_catboost,
            "catboost_x2": model_catboost_x2,
            "catboost_x3": model_catboost_x3
        }

    elif data_size_type == "small":
        X_values = X.to_numpy(na_value=0.0, dtype=np.float32)

        x_train, x_valid, y_train, y_valid = train_test_split(
            X_values, y, test_size=ts, stratify=y, random_state=RANDOM_SEED
        )

        X_mean = np.mean(x_train, axis=0)
        X_std = np.std(x_train, axis=0) + 1e-8

        x_train_normalized = (x_train - X_mean) / X_std
        x_valid_normalized = (x_valid - X_mean) / X_std

        X_mean_full = np.mean(X_values, axis=0)
        X_std_full = np.std(X_values, axis=0) + 1e-8
        x_full_normalized = (X_values - X_mean_full) / X_std_full

        # SVM
        model_svm = SVC(
            max_iter=10000, tol=1e-7, C=1e-5, kernel='linear', class_weight="balanced",
            probability=True, verbose=False, random_state=RANDOM_SEED
        )
        model_svm.fit(x_train_normalized, y_train)
        pred_svm = model_svm.predict_proba(x_valid_normalized)[:, 1]

        model_svm = SVC(
            max_iter=10000, tol=1e-7, C=1e-5, kernel='linear', class_weight="balanced",
            probability=True, verbose=False, random_state=RANDOM_SEED
        )
        model_svm.fit(x_full_normalized, y)

        # Logistic Regression
        model_logreg = LogisticRegression(
            solver='liblinear', class_weight="balanced",
            verbose=False, random_state=RANDOM_SEED, n_jobs=-1
        )
        model_logreg.fit(x_train_normalized, y_train)
        pred_logreg = model_logreg.predict_proba(x_valid_normalized)[:, 1]

        model_logreg = LogisticRegression(
            solver='liblinear', class_weight="balanced",
            verbose=False, random_state=RANDOM_SEED, n_jobs=-1
        )
        model_logreg.fit(x_full_normalized, y)

        # Random Forest
        model_random_forest = RandomForestClassifier(
            n_estimators=300, n_jobs=-1, class_weight="balanced",
            verbose=False, random_state=RANDOM_SEED
        )
        model_random_forest.fit(x_train, y_train)
        pred_random_forest = model_random_forest.predict_proba(x_valid_normalized)[:, 1]

        model_random_forest = RandomForestClassifier(
            n_estimators=300, n_jobs=-1, class_weight="balanced",
            verbose=False, random_state=RANDOM_SEED
        )
        model_random_forest.fit(X_values, y)

        # LightGBM
        model_lightgbm = LGBMClassifier(
            learning_rate=0.1, device_type=DEVICE_LIGHTGBM, early_stopping_round=50, is_unbalance=True,
            verbose=-1, random_state=RANDOM_SEED, n_jobs=-1
        )
        model_lightgbm.fit(x_train, y_train, eval_set=(x_valid, y_valid), eval_metric='auc')
        lgbm_best_iter = model_lightgbm.best_iteration_
        pred_lightgbm = model_lightgbm.predict_proba(x_valid)[:, 1]

        model_lightgbm = LGBMClassifier(
            learning_rate=0.1, device_type=DEVICE_LIGHTGBM, n_estimators=lgbm_best_iter, is_unbalance=True,
            verbose=-1, random_state=RANDOM_SEED, n_jobs=-1
        )
        model_lightgbm.fit(X_values, y)

        # XGBoost early stopping training
        model_xgboost = XGBClassifier(
            learning_rate=0.1, device=DEVICE_XGB, eval_metric="auc", n_estimators=10000, early_stopping_rounds=50,
            scale_pos_weight=class_weights[1] / class_weights[0],
            verbosity=0, random_state=RANDOM_SEED, n_jobs=-1
        )
        model_xgboost.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
        xgboost_best_iter = model_xgboost.get_booster().best_iteration
        pred_xgboost = model_xgboost.predict_proba(x_valid)[:, 1]

        model_xgboost = XGBClassifier(
            learning_rate=0.1, device=DEVICE_XGB, n_estimators=xgboost_best_iter,
            scale_pos_weight=class_weights[1] / class_weights[0],
            verbosity=0, random_state=RANDOM_SEED, n_jobs=-1
        )
        model_xgboost.fit(X_values, y, verbose=False)

        # Catboost early stopping training
        train_pool = Pool(x_train, y_train)
        eval_pool = Pool(x_valid, y_valid)
        full_pool = Pool(X_values, y)

        model_catboost = CatBoostClassifier(
            learning_rate=0.1, task_type=DEVICE, eval_metric='AUC', iterations=10000, early_stopping_rounds=50,
            verbose=False, random_state=RANDOM_SEED
        )
        model_catboost.fit(train_pool, eval_set=eval_pool)
        catboost_best_iter = model_catboost.best_iteration_ + 1
        pred_catboost = model_catboost.predict_proba(x_valid)[:, 1]

        model_catboost = CatBoostClassifier(
            learning_rate=0.1, task_type=DEVICE, iterations=catboost_best_iter,
            verbose=False, random_state=RANDOM_SEED
        )
        model_catboost.fit(full_pool)

        # Catboost x2
        model_catboost_x2 = CatBoostClassifier(
            depth=7, learning_rate=0.05, task_type=DEVICE, eval_metric='AUC', iterations=10000,
            early_stopping_rounds=50, verbose=False, random_state=RANDOM_SEED + 1
        )
        model_catboost_x2.fit(train_pool, eval_set=eval_pool)
        catboost_x2_best_iter = model_catboost_x2.best_iteration_ + 1
        pred_catboost_x2 = model_catboost_x2.predict_proba(x_valid)[:, 1]

        model_catboost_x2 = CatBoostClassifier(
            depth=7, learning_rate=0.05, task_type=DEVICE, iterations=catboost_x2_best_iter,
            verbose=False, random_state=RANDOM_SEED + 1
        )
        model_catboost_x2.fit(full_pool)

        # Catboost x3
        model_catboost_x3 = CatBoostClassifier(
            depth=7, learning_rate=0.025, task_type=DEVICE, eval_metric='AUC', iterations=10000,
            early_stopping_rounds=50, verbose=False, random_state=RANDOM_SEED + 2
        )
        model_catboost_x3.fit(train_pool, eval_set=eval_pool)
        catboost_x3_best_iter = model_catboost_x3.best_iteration_ + 1
        pred_catboost_x3 = model_catboost_x3.predict_proba(x_valid)[:, 1]

        model_catboost_x3 = CatBoostClassifier(
            depth=7, learning_rate=0.025, task_type=DEVICE, iterations=catboost_x3_best_iter,
            verbose=False, random_state=RANDOM_SEED + 2
        )
        model_catboost_x3.fit(full_pool)

        optuna_coeffs = small_optuna_blender(
            y_valid, svm_preds=pred_svm, logreg_preds=pred_logreg, random_forest_preds=pred_random_forest,
            lightgbm_preds=pred_lightgbm, xgboost_preds=pred_xgboost, catboost_preds=pred_catboost,
            catboost_x2_preds=pred_catboost_x2, catboost_x3_preds=pred_catboost_x3
        )

        models = {
            "svm": model_svm,
            "logreg": model_logreg,
            "random_forest": model_random_forest,
            "lightgbm": model_lightgbm,
            "xgboost": model_xgboost,
            "catboost": model_catboost,
            "catboost_x2": model_catboost_x2,
            "catboost_x3": model_catboost_x3,
            "optuna_coeffs": optuna_coeffs
        }

    elif data_size_type == "medium":
        x_train, x_valid, y_train, y_valid = train_test_split(
            X, y, test_size=ts, stratify=y, random_state=RANDOM_SEED
        )

        # XGBoost early stopping training
        model_xgboost = XGBClassifier(
            learning_rate=0.1, device=DEVICE_XGB, eval_metric="auc", n_estimators=10000, early_stopping_rounds=50,
            scale_pos_weight=class_weights[1] / class_weights[0],
            verbosity=0, random_state=RANDOM_SEED, n_jobs=-1,
        )
        model_xgboost.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
        xgboost_best_iter = model_xgboost.get_booster().best_iteration
        pred_xgboost = model_xgboost.predict_proba(x_valid)[:, 1]

        # XGBoost full training
        model_xgboost = XGBClassifier(
            learning_rate=0.1, device=DEVICE_XGB, n_estimators=xgboost_best_iter,
            scale_pos_weight=class_weights[1] / class_weights[0],
            verbosity=0, random_state=RANDOM_SEED, n_jobs=-1,
        )
        model_xgboost.fit(X, y, verbose=False)

        # Catboost
        train_pool = Pool(x_train, y_train)
        eval_pool = Pool(x_valid, y_valid)
        full_pool = Pool(X, y)

        model_catboost = CatBoostClassifier(
            learning_rate=0.1, task_type=DEVICE, eval_metric='AUC', iterations=10000, early_stopping_rounds=50,
            verbose=False, random_state=RANDOM_SEED
        )
        model_catboost.fit(train_pool, eval_set=eval_pool)
        catboost_best_iter = model_catboost.best_iteration_ + 1
        pred_catboost = model_catboost.predict_proba(x_valid)[:, 1]

        model_catboost = CatBoostClassifier(
            learning_rate=0.1, task_type=DEVICE, iterations=catboost_best_iter,
            verbose=False, random_state=RANDOM_SEED
        )
        model_catboost.fit(full_pool)

        # Catboost x2
        model_catboost_x2 = CatBoostClassifier(
            depth=7, learning_rate=0.05, task_type=DEVICE, eval_metric='AUC', iterations=10000, early_stopping_rounds=50,
            verbose=False, random_state=RANDOM_SEED+1
        )
        model_catboost_x2.fit(train_pool, eval_set=eval_pool)
        catboost_x2_best_iter = model_catboost_x2.best_iteration_ + 1
        pred_catboost_x2 = model_catboost_x2.predict_proba(x_valid)[:, 1]

        model_catboost_x2 = CatBoostClassifier(
            depth=7, learning_rate=0.05, task_type=DEVICE, iterations=catboost_x2_best_iter,
            verbose=False, random_state=RANDOM_SEED+1
        )
        model_catboost_x2.fit(full_pool)

        # Catboost x3
        model_catboost_x3 = CatBoostClassifier(
            depth=8, learning_rate=0.03, task_type=DEVICE, eval_metric='AUC', iterations=10000, early_stopping_rounds=50,
            verbose=False, random_state=RANDOM_SEED+2
        )
        model_catboost_x3.fit(train_pool, eval_set=eval_pool)
        catboost_x3_best_iter = model_catboost_x3.best_iteration_ + 1
        pred_catboost_x3 = model_catboost_x3.predict_proba(x_valid)[:, 1]

        model_catboost_x3 = CatBoostClassifier(
            depth=8, learning_rate=0.03, task_type=DEVICE, iterations=catboost_x3_best_iter,
            verbose=False, random_state=RANDOM_SEED+2
        )
        model_catboost_x3.fit(full_pool)

        optuna_coeffs = big_optuna_blender(
            y_valid,
            xgboost_preds=pred_xgboost,
            catboost_preds=pred_catboost,
            catboost_x2_preds=pred_catboost_x2,
            catboost_x3_preds=pred_catboost_x3
        )

        models = {
            "xgboost": model_xgboost,
            "catboost": model_catboost,
            "catboost_x2": model_catboost_x2,
            "catboost_x3": model_catboost_x3,
            "optuna_coeffs": optuna_coeffs
        }

    else:
        x_train, x_valid, y_train, y_valid = train_test_split(
            X, y, test_size=ts, stratify=y, random_state=RANDOM_SEED
        )

        # XGBoost early stopping training
        model_xgboost = XGBClassifier(
            learning_rate=0.1, device=DEVICE_XGB, eval_metric="auc", n_estimators=10000, early_stopping_rounds=50,
            scale_pos_weight=class_weights[1] / class_weights[0],
            verbosity=0, random_state=RANDOM_SEED, n_jobs=-1,
        )
        model_xgboost.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
        pred_xgboost = model_xgboost.predict_proba(x_valid)[:, 1]

        train_pool = Pool(x_train, y_train)
        eval_pool = Pool(x_valid, y_valid)

        # Catboost early stopping training
        model_catboost = CatBoostClassifier(
            learning_rate=0.1, task_type=DEVICE, eval_metric='AUC', iterations=10000, early_stopping_rounds=50,
            verbose=False, random_state=RANDOM_SEED
        )
        model_catboost.fit(train_pool, eval_set=eval_pool)
        pred_catboost = model_catboost.predict_proba(x_valid)[:, 1]

        # Catboost x2 early stopping training
        model_catboost_x2 = CatBoostClassifier(
            depth=8, learning_rate=0.05, task_type=DEVICE, eval_metric='AUC', iterations=10000, early_stopping_rounds=50,
            verbose=False, random_state=RANDOM_SEED + 1
        )
        model_catboost_x2.fit(train_pool, eval_set=eval_pool)
        pred_catboost_x2 = model_catboost_x2.predict_proba(x_valid)[:, 1]

        # Catboost x3 early stopping training
        model_catboost_x3 = CatBoostClassifier(
            depth=10, learning_rate=0.05, task_type=DEVICE, eval_metric='AUC', iterations=10000, early_stopping_rounds=50,
            verbose=False, random_state=RANDOM_SEED + 2
        )
        model_catboost_x3.fit(train_pool, eval_set=eval_pool)
        pred_catboost_x3 = model_catboost_x3.predict_proba(x_valid)[:, 1]

        optuna_coeffs = big_optuna_blender(
            y_valid, xgboost_preds=pred_xgboost, catboost_preds=pred_catboost, catboost_x2_preds=pred_catboost_x2,
            catboost_x3_preds=pred_catboost_x3
        )

        models = {
            "xgboost": model_xgboost,
            "catboost": model_catboost,
            "catboost_x2": model_catboost_x2,
            "catboost_x3": model_catboost_x3,
            "optuna_coeffs": optuna_coeffs
        }

    res = {
        "X_mean": X_mean,
        "X_std": X_std,
        "models": models
    }

    return res


def small_optuna_blender(
    y_valid: np.ndarray,
    svm_preds: np.ndarray,
    logreg_preds: np.ndarray,
    random_forest_preds: np.ndarray,
    lightgbm_preds: np.ndarray,
    xgboost_preds: np.ndarray,
    catboost_preds: np.ndarray,
    catboost_x2_preds: np.ndarray,
    catboost_x3_preds: np.ndarray
) -> dict[str, float]:
    def objective(trial):
        svm = trial.suggest_float("svm", 0, 1, step=0.01)
        logreg = trial.suggest_float("logreg", 0, 1, step=0.01)
        random_forest = trial.suggest_float("random_forest", 0, 1, step=0.01)
        lightgbm = trial.suggest_float("lightgbm", 0, 1, step=0.01)
        xgboost = trial.suggest_float("xgboost", 0, 1, step=0.01)
        catboost = trial.suggest_float("catboost", 0, 1, step=0.01)
        catboost_x2 = trial.suggest_float("catboost_x2", 0, 1, step=0.01)
        catboost_x3 = trial.suggest_float("catboost_x3", 0, 1, step=0.01)

        s = svm + logreg + random_forest + lightgbm + xgboost + catboost + catboost_x2 + catboost_x3

        if s == 0:
            return 0
        else:
            svm_normed = svm / s
            logreg_normed = logreg / s
            random_forest_normed = random_forest / s
            lightgbm_normed = lightgbm / s
            xgboost_normed = xgboost / s
            catboost_normed = catboost / s
            catboost_x2_normed = catboost_x2 / s
            catboost_x3_normed = catboost_x3 / s

            new_probs = (
                svm_preds * svm_normed +
                logreg_preds * logreg_normed +
                random_forest_preds * random_forest_normed +
                lightgbm_preds * lightgbm_normed +
                xgboost_preds * xgboost_normed +
                catboost_preds * catboost_normed +
                catboost_x2_preds * catboost_x2_normed +
                catboost_x3_preds * catboost_x3_normed
            )

            return roc_auc_score(y_valid, new_probs)

    tpe_sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction='maximize', sampler=tpe_sampler)
    study.enqueue_trial({'svm': 0, 'logreg': 0, 'random_forest': 0, 'lightgbm': 0, 'xgboost': 0, 'catboost': 1, "catboost_x2": 0, "catboost_x3": 0})
    study.enqueue_trial({'svm': 0, 'logreg': 0, 'random_forest': 0, 'lightgbm': 0, 'xgboost': 1, 'catboost': 0, "catboost_x2": 0, "catboost_x3": 0})
    study.enqueue_trial({'svm': 0, 'logreg': 0, 'random_forest': 0, 'lightgbm': 1, 'xgboost': 0, 'catboost': 0, "catboost_x2": 0, "catboost_x3": 0})
    study.enqueue_trial({'svm': 0, 'logreg': 0, 'random_forest': 1, 'lightgbm': 0, 'xgboost': 0, 'catboost': 0, "catboost_x2": 0, "catboost_x3": 0})
    study.enqueue_trial({'svm': 0, 'logreg': 1, 'random_forest': 0, 'lightgbm': 0, 'xgboost': 0, 'catboost': 0, "catboost_x2": 0, "catboost_x3": 0})
    study.enqueue_trial({'svm': 1, 'logreg': 0, 'random_forest': 0, 'lightgbm': 0, 'xgboost': 0, 'catboost': 0, "catboost_x2": 0, "catboost_x3": 0})
    study.enqueue_trial({'svm': 0, 'logreg': 0, 'random_forest': 0, 'lightgbm': 0, 'xgboost': 0, 'catboost': 0, "catboost_x2": 1, "catboost_x3": 0})
    study.enqueue_trial({'svm': 0, 'logreg': 0, 'random_forest': 0, 'lightgbm': 0, 'xgboost': 0, 'catboost': 0, "catboost_x2": 0, "catboost_x3": 1})
    study.enqueue_trial({'svm': 1, 'logreg': 1, 'random_forest': 1, 'lightgbm': 1, 'xgboost': 1, 'catboost': 1, "catboost_x2": 1, "catboost_x3": 1})
    study.optimize(objective, n_trials=100)

    coefs = study.best_params

    sum_coefs = sum(coefs.values())

    coefs['svm'] /= sum_coefs
    coefs['logreg'] /= sum_coefs
    coefs['random_forest'] /= sum_coefs
    coefs['lightgbm'] /= sum_coefs
    coefs['xgboost'] /= sum_coefs
    coefs['catboost'] /= sum_coefs
    coefs['catboost_x2'] /= sum_coefs
    coefs['catboost_x3'] /= sum_coefs

    return coefs


def big_optuna_blender(
    y_valid: np.ndarray, xgboost_preds: np.ndarray, catboost_preds: np.ndarray, catboost_x2_preds: np.ndarray,
    catboost_x3_preds: np.ndarray
) -> dict[str, float]:
    def objective(trial):
        xgboost = trial.suggest_float("xgboost", 0, 1, step=0.01)
        catboost = trial.suggest_float("catboost", 0, 1, step=0.01)
        catboost_x2 = trial.suggest_float("catboost_x2", 0, 1, step=0.01)
        catboost_x3 = trial.suggest_float("catboost_x3", 0, 1, step=0.01)

        s = xgboost + catboost + catboost_x2 + catboost_x3

        if s == 0:
            return 0
        else:
            xgboost_normed = xgboost / s
            catboost_normed = catboost / s
            catboost_x2_normed = catboost_x2 / s
            catboost_x3_normed = catboost_x3 / s
            new_probs = (
                    xgboost_preds * xgboost_normed +
                    catboost_preds * catboost_normed +
                    catboost_x2_preds * catboost_x2_normed +
                    catboost_x3_preds * catboost_x3_normed
            )

            return roc_auc_score(y_valid, new_probs)

    tpe_sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction='maximize', sampler=tpe_sampler)
    study.enqueue_trial({'xgboost': 0, "catboost": 1, "catboost_x2": 0, "catboost_x3": 0})
    study.enqueue_trial({'xgboost': 1, "catboost": 0, "catboost_x2": 0, "catboost_x3": 0})
    study.enqueue_trial({'xgboost': 0, "catboost": 0, "catboost_x2": 1, "catboost_x3": 0})
    study.enqueue_trial({'xgboost': 0, "catboost": 0, "catboost_x2": 0, "catboost_x3": 1})
    study.enqueue_trial({'xgboost': 1, "catboost": 1, "catboost_x2": 1, "catboost_x3": 1})

    study.optimize(objective, n_trials=100)

    coefs = study.best_params

    sum_coefs = sum(coefs.values())

    coefs['xgboost'] /= sum_coefs
    coefs['catboost'] /= sum_coefs
    coefs['catboost_x2'] /= sum_coefs
    coefs['catboost_x3'] /= sum_coefs

    return coefs


def test_preprocess(
    test_paths: list[str],
    cat_cols: list[str],
    train_cols: list[str]
) -> dict[str, pd.DataFrame]:
    test_data = load_data(test_paths)
    print(f"-- Test data has shape: {test_data.shape}")

    # OHE
    print("-- OneHotEncoding on test data...")
    test_data = pd.get_dummies(test_data, columns=cat_cols)

    # Add missing columns from train data to test data (if OHE had different categories)
    test_cols = set(test_data.columns)
    for col in train_cols:
        if col not in test_cols:
            test_data[col] = 0

    # Get ids for submission predictions
    submit_data = test_data[["id"]]

    print("-- Get all train columns on test data...")
    # Reorder columns to match the train data
    test_data = test_data[train_cols]

    test_res = {
        "test_data": test_data,
        "submit_data": submit_data
    }

    return test_res


def predict(
    train_model_res: dict,
    test_data: pd.DataFrame,
    submit_data: pd.DataFrame
) -> pd.DataFrame:
    y_pred = []
    models = train_model_res["models"]

    if "svm" in models:
        X_mean = train_model_res["X_mean"]
        X_std = train_model_res["X_std"]

        X_test_values = test_data.to_numpy(na_value=0.0, dtype=np.float32)
        X_test_values = (X_test_values - X_mean) / X_std

        svm_model = models["svm"]
        y_pred.append(svm_model.predict_proba(X_test_values)[:, 1])

        logreg_model = models["logreg"]
        y_pred.append(logreg_model.predict_proba(X_test_values)[:, 1])

        random_forest_model = models["random_forest"]
        y_pred.append(random_forest_model.predict_proba(test_data)[:, 1])

        lightgbm_model = models["lightgbm"]
        y_pred.append(lightgbm_model.predict_proba(test_data)[:, 1])

    xgboost_model = models["xgboost"]
    y_pred.append(xgboost_model.predict_proba(test_data)[:, 1])

    catboost_model = models["catboost"]
    y_pred.append(catboost_model.predict_proba(test_data)[:, 1])

    catboost_x2_model = models["catboost_x2"]
    y_pred.append(catboost_x2_model.predict_proba(test_data)[:, 1])

    catboost_x3_model = models["catboost_x3"]
    y_pred.append(catboost_x3_model.predict_proba(test_data)[:, 1])

    print(models.get("optuna_coeffs", "HAHA"))
    if "optuna_coeffs" in models and len(models["optuna_coeffs"]) == 4:
        y_pred = (
                y_pred[0] * models["optuna_coeffs"]["xgboost"] +
                y_pred[1] * models["optuna_coeffs"]["catboost"] +
                y_pred[2] * models["optuna_coeffs"]["catboost_x2"] +
                y_pred[3] * models["optuna_coeffs"]["catboost_x3"]
        )
    elif "optuna_coeffs" in models:
        y_pred = (
                y_pred[0] * models["optuna_coeffs"]["svm"] +
                y_pred[1] * models["optuna_coeffs"]["logreg"] +
                y_pred[2] * models["optuna_coeffs"]["random_forest"] +
                y_pred[3] * models["optuna_coeffs"]["lightgbm"] +
                y_pred[4] * models["optuna_coeffs"]["xgboost"] +
                y_pred[5] * models["optuna_coeffs"]["catboost"] +
                y_pred[6] * models["optuna_coeffs"]["catboost_x2"] +
                y_pred[7] * models["optuna_coeffs"]["catboost_x3"]
        )
    else:
        y_pred = np.mean(y_pred, axis=0)

    # print(y_pred)

    submit_data['target'] = y_pred

    return submit_data


def modelling(dataset_folder: str) -> pd.DataFrame:
    print(f"Currently in dataset folder {dataset_folder}")
    current_data_path = os.path.join(DATA_PATH, dataset_folder)
    current_data_list = os.listdir(current_data_path)

    train_paths = [
        os.path.join(current_data_path, data)
        for data in current_data_list
        if data.endswith('train.parquet')
    ]
    test_paths = [
        os.path.join(current_data_path, data)
        for data in current_data_list
        if data.endswith('test.parquet')
    ]
    print(f"Got {train_paths} and {test_paths}")

    print("Start train preprocessing...")
    train_data_res = train_preprocess(train_paths)
    print("Train preprocessing done!")

    print(f"Start feature selection...")
    train_data_res['data'] = feature_selection(
        train_data_res['data'], train_data_res['target'],
    )
    print("Feature selection done!")

    print("Start training...")
    train_model_res = train(train_data_res['data'], train_data_res['target'])
    print("Training done!")

    print("Start test preprocessing...")
    test_res = test_preprocess(
        test_paths,
        cat_cols=train_data_res['cat_cols'],
        train_cols=train_data_res['data'].columns.tolist()
    )
    print("Test preprocessing done!")

    print("Start prediction...")
    prediction = predict(
        train_model_res,
        test_res["test_data"],
        test_res["submit_data"]
    )
    print("Prediction done!")

    return prediction


def main():
    # Запишем список датасетов в папке:
    print(f"Calculating with {DEVICE}")
    datasets_folders = os.listdir(DATA_PATH)

    if len(datasets_folders) == 0:
        raise Exception("No datasets!")

    # Создадим папку для предсказаний
    os.makedirs(PREDICTION_PATH, exist_ok=True)

    # Calculate total time
    total_time = 0
    num_valid_datasets = 0

    # Создадим цикл для прохождения по каждому файлу и генерации предсказания
    for dataset_folder in datasets_folders:
        if not os.path.isdir(os.path.join(DATA_PATH, dataset_folder)):
            print(f"Skipped {dataset_folder} (not a directory)")
            continue

        start_time = time.time()
        prediction = modelling(dataset_folder)

        # Сохраняем предсказание
        prediction.to_csv(os.path.join(PREDICTION_PATH, dataset_folder) + ".csv", index=False)

        dataset_time = time.time() - start_time
        total_time += dataset_time
        num_valid_datasets += 1
        print(f"Prediction is saved! Dataset time: {dataset_time}")
        print("\n================================")

    print(f"All datasets are done! Total time: {total_time}")


if __name__ == "__main__":
    main()
