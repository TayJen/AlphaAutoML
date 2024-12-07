import time

import pandas as pd
import numpy as np
import os
import shap
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.svm import SVC

RANDOM_SEED = 42
DATA_PATH = "data"
PREDICTION_PATH = "predictions"
# DEVICE = "CPU"
DEVICE = "GPU"
DEVICE_VERY_SMALL = "CPU"
DEVICE_LIGHTGBM = "cpu"


DATA_SIZE_THRESHOLDS = (1000, 100_000, 1_000_000)
DATA_SMALL_CLASS_THRESHOLD = 30


def load_data(data_paths: list[str]) -> pd.DataFrame:
    data = pd.read_parquet(data_paths[0])
    return data


def train_preprocess(
    train_paths: list[str],
    nan_threshold=0.98, const_threshold=0.98,
    corr_sample=500000, corr_threshold=0.98
):
    data = load_data(train_paths)
    data = data.reset_index(drop=True)
    print(f"-- Data has shape: {data.shape}")

    data = data.drop(['smpl', 'id'], axis=1, errors='ignore')
    print("-- Dropped smpl and id")

    # Change float64 to float16
    float_cols = list(data.loc[:, data.dtypes == 'float64'].columns)
    data[float_cols] = data[float_cols].astype('float32')   # float32?
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

    # Remove near const cols
    # to_drop = []
    # tmp = data.shape[1]
    # for i in data.columns:
    #     if data[i].value_counts(normalize=True).values[0] > const_threshold:
    #         to_drop.append(i)
    # if len(to_drop) > 0:
    #     data = data.drop(to_drop, axis=1)
    # tmp -= data.shape[1]
    # if tmp == 0:
    #     print('-- no high const cols')
    # else:
    #     print(f"-- Removed {tmp} high const cols")

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

    if data_size_type == "very_small":
        X_values = X.to_numpy(na_value=0.0, dtype=np.float32)
        X_mean = np.mean(X_values, axis=0)
        X_std = np.std(X_values, axis=0) + 1e-8

        X_values = (X_values - X_mean) / X_std

        model_svm = SVC(
            max_iter=10000, tol=1e-7, C=1e-5, kernel='linear',
            probability=True, verbose=False, random_state=RANDOM_SEED
        )
        model_svm.fit(X_values, y)

        model_logreg = LogisticRegression(
            max_iter=10000, C=1e-5, solver='liblinear',
            verbose=False, random_state=RANDOM_SEED, n_jobs=-1
        )
        model_logreg.fit(X_values, y)

        model_random_forest = RandomForestClassifier(
            n_estimators=300, n_jobs=-1,
            verbose=False, random_state=RANDOM_SEED
        )
        model_random_forest.fit(X, y)

        model_lightgbm = LGBMClassifier(
            learning_rate=0.1, device_type=DEVICE_LIGHTGBM,
            verbose=-1, random_state=RANDOM_SEED, n_jobs=-1
        )
        model_lightgbm.fit(X, y)

        model_catboost = CatBoostClassifier(
            learning_rate=0.1, task_type=DEVICE_VERY_SMALL,
            verbose=False, random_state=RANDOM_SEED
        )
        model_catboost.fit(X, y)

        models = {
            "svm": model_svm,
            "logreg": model_logreg,
            "random_forest": model_random_forest,
            "lightgbm": model_lightgbm,
            "catboost": model_catboost
        }

    elif data_size_type == "small":
        X_values = X.to_numpy(na_value=0.0, dtype=np.float32)

        X_mean = np.mean(X_values, axis=0)
        X_std = np.std(X_values, axis=0) + 1e-8

        X_values = (X_values - X_mean) / X_std

        model_svm = SVC(
            max_iter=10000, tol=1e-7, C=1e-5, kernel='linear',
            probability=True, verbose=False, random_state=RANDOM_SEED
        )
        model_svm.fit(X_values, y)

        model_logreg = LogisticRegression(
            max_iter=10000, C=1e-5, solver='liblinear',
            verbose=False, random_state=RANDOM_SEED, n_jobs=-1
        )
        model_logreg.fit(X_values, y)

        model_random_forest = RandomForestClassifier(
            n_estimators=300, n_jobs=-1,
            verbose=False, random_state=RANDOM_SEED
        )
        model_random_forest.fit(X, y)

        x_train, x_valid, y_train, y_valid = train_test_split(
            X, y, test_size=ts, stratify=y, random_state=RANDOM_SEED
        )

        # LightGBM early stopping training
        model_lightgbm = LGBMClassifier(
            learning_rate=0.1, device_type=DEVICE_LIGHTGBM, early_stopping_round=50,
            verbose=-1, random_state=RANDOM_SEED, n_jobs=-1
        )
        model_lightgbm.fit(x_train, y_train, eval_set=(x_valid, y_valid), eval_metric='auc')
        lgbm_best_iter = model_lightgbm.best_iteration_

        # LightGBM full training
        model_lightgbm = LGBMClassifier(
            learning_rate=0.1, device_type=DEVICE_LIGHTGBM, n_estimators=lgbm_best_iter,
            verbose=-1, random_state=RANDOM_SEED, n_jobs=-1
        )
        model_lightgbm.fit(X, y)

        # Catboost early stopping training
        model_catboost = CatBoostClassifier(
            learning_rate=0.1, task_type=DEVICE, eval_metric='AUC', early_stopping_rounds=50,
            verbose=False, random_state=RANDOM_SEED
        )
        model_catboost.fit(x_train, y_train, eval_set=(x_valid, y_valid))
        catboost_best_iter = model_catboost.best_iteration_ + 1

        # Catboost full training
        model_catboost = CatBoostClassifier(
            learning_rate=0.1, task_type=DEVICE, n_estimators=catboost_best_iter,
            verbose=False, random_state=RANDOM_SEED
        )
        model_catboost.fit(X, y)

        models = {
            "svm": model_svm,
            "logreg": model_logreg,
            "random_forest": model_random_forest,
            "lightgbm": model_lightgbm,
            "catboost": model_catboost
        }

    elif data_size_type == "medium":
        x_train, x_valid, y_train, y_valid = train_test_split(
            X, y, test_size=ts, stratify=y, random_state=RANDOM_SEED
        )

        # LightGBM early stopping training
        model_lightgbm = LGBMClassifier(
            learning_rate=0.1, device_type=DEVICE_LIGHTGBM, early_stopping_round=50,
            verbose=-1, random_state=RANDOM_SEED, n_jobs=-1
        )
        model_lightgbm.fit(x_train, y_train, eval_set=(x_valid, y_valid), eval_metric='auc')
        lgbm_best_iter = model_lightgbm.best_iteration_

        # LightGBM full training
        model_lightgbm = LGBMClassifier(
            learning_rate=0.1, device_type=DEVICE_LIGHTGBM, n_estimators=lgbm_best_iter,
            verbose=-1, random_state=RANDOM_SEED, n_jobs=-1
        )
        model_lightgbm.fit(X, y)

        # Catboost early stopping training
        model_catboost = CatBoostClassifier(
            learning_rate=0.1, task_type=DEVICE, eval_metric='AUC', early_stopping_rounds=50,
            verbose=False, random_state=RANDOM_SEED
        )
        model_catboost.fit(x_train, y_train, eval_set=(x_valid, y_valid))
        catboost_best_iter = model_catboost.best_iteration_ + 1

        # Catboost full training
        model_catboost = CatBoostClassifier(
            learning_rate=0.1, task_type=DEVICE, n_estimators=catboost_best_iter,
            verbose=False, random_state=RANDOM_SEED
        )
        model_catboost.fit(X, y)

        models = {
            "lightgbm": model_lightgbm,
            "catboost": model_catboost
        }

    else:
        x_train, x_valid, y_train, y_valid = train_test_split(
            X, y, test_size=ts, stratify=y, random_state=RANDOM_SEED
        )

        # LightGBM early stopping training
        model_lightgbm = LGBMClassifier(
            learning_rate=0.1, device_type=DEVICE_LIGHTGBM, early_stopping_round=50,
            verbose=-1, random_state=RANDOM_SEED, n_jobs=-1
        )
        model_lightgbm.fit(x_train, y_train, eval_set=(x_valid, y_valid), eval_metric='auc')

        # Catboost early stopping training
        model_catboost = CatBoostClassifier(
            learning_rate=0.1, task_type=DEVICE, eval_metric='AUC', early_stopping_rounds=50,
            verbose=False, random_state=RANDOM_SEED
        )
        model_catboost.fit(x_train, y_train, eval_set=(x_valid, y_valid))

        models = {
            "lightgbm": model_lightgbm,
            "catboost": model_catboost
        }

    res = {
        "X_mean": X_mean,
        "X_std": X_std,
        "models": models
    }

    return res


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

    if len(models) > 2:
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

    catboost_model = models["catboost"]
    y_pred.append(catboost_model.predict_proba(test_data)[:, 1])

    # print(y_pred)
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
