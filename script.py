import time

import pandas as pd
import numpy as np
import os
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier


RANDOM_SEED = 42
DATA_PATH = "data"
PREDICTION_PATH = "predictions"
# DEVICE = "CPU"
DEVICE = "GPU"


def load_data(data_paths: list[str]) -> pd.DataFrame:
    data = pd.read_parquet(data_paths[0])
    return data


def train_preprocess(
    train_paths: list[str],
    nan_threshold=0.5, const_threshold=0.98,
    corr_sample=500000, corr_threshold=0.98
):
    data = load_data(train_paths)
    data = data.reset_index(drop=True)
    print(f"-- Data has shape: {data.shape}")

    data = data.drop(['smpl', 'id'], axis=1, errors='ignore')
    print("-- Dropped smpl and id")

    # Change float64 to float16
    float_cols = list(data.loc[:, data.dtypes == 'float64'].columns)
    data[float_cols] = data[float_cols].astype('float16')
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
    to_drop = []
    tmp = data.shape[1]
    for i in data.columns:
        if data[i].value_counts(normalize=True).values[0] > const_threshold:
            to_drop.append(i)
    if len(to_drop) > 0:
        data = data.drop(to_drop, axis=1)
    tmp -= data.shape[1]
    if tmp == 0:
        print('-- no high const cols')
    else:
        print(f"-- Removed {tmp} high const cols")

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
    data[cat_cols] = data[cat_cols].astype('int8')
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
    mode: str = 'hybrid',
    max_features: int = 500
) -> pd.DataFrame:
    tmp = data.shape[1]

    if y.value_counts().min() == 1:
        print(f"** Fitting model for feature selection without early_stopping (total number of rows: {len(data)}...")
        model = CatBoostClassifier(
            learning_rate=0.1, verbose=False, random_state=RANDOM_SEED
        )
        model.fit(data, y)

        x_valid = data
    else:
        x_train, x_valid, y_train, y_valid = train_test_split(
            data, y, test_size=0.1, stratify=y, random_state=RANDOM_SEED
        )

        print("** Fitting model for feature selection with early_stopping...")
        model = CatBoostClassifier(
            learning_rate=0.1, iterations=10000, eval_metric='AUC', early_stopping_rounds=50,
            verbose=False, task_type=DEVICE, random_state=RANDOM_SEED
        )
        model.fit(x_train, y_train, eval_set=(x_valid, y_valid))

    print("** Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_valid)

    imp = pd.DataFrame({'name': x_valid.columns, 'shap_value': np.mean(abs(shap_values), axis=0)})
    imp = imp.sort_values('shap_value', ascending=False)
    imp['shap_value'] /= imp['shap_value'].sum()

    print("** Selecting top features...")
    if mode == 'hybrid':
        top_features = list(imp[imp['shap_value'] > 1e-5].head(max_features)['name'])
    elif mode == 'top_n':
        top_features = list(imp.head(max_features)['name'])
    else:
        top_features = list(imp[imp['shap_value'] > 1e-5]['name'])

    data = data[top_features]
    
    tmp -= data.shape[1]
    print(f'** Deleted {tmp} cols after feature selection')
    print(f'** Total amount of cols after feature selection {data.shape[1]}')
    return data


def train(x: pd.DataFrame, y: pd.Series) -> tuple[CatBoostClassifier, float]:
    if len(x) > 1_000_000:
        ts = 0.1
    else:
        ts = 0.2

    is_needed_full_training = False
    depth = 6

    if y.value_counts().min() == 1:
        print(f"++ Training without early_stopping (total number of rows: {len(x)}...")
        model = CatBoostClassifier(
            learning_rate=0.1, verbose=False, random_state=RANDOM_SEED
        )
        model.fit(x, y)

        x_valid, y_valid = x, y
    else:
        x_train, x_valid, y_train, y_valid = train_test_split(
            x, y, test_size=ts, stratify=y, random_state=RANDOM_SEED
        )

        print("++ Training with early_stopping...")
        model_base = CatBoostClassifier(
            iterations=10_000, learning_rate=0.1, eval_metric='AUC', early_stopping_rounds=50,
            verbose=False, task_type=DEVICE, random_state=RANDOM_SEED
        )
        model_base.fit(x_train, y_train, eval_set=(x_valid, y_valid))
        model_base_best_score = model_base.best_score_['validation']['AUC']

        model_medium = CatBoostClassifier(
            depth=8, iterations=10_000, learning_rate=0.03, eval_metric='AUC', early_stopping_rounds=50,
            verbose=False, task_type=DEVICE, random_state=RANDOM_SEED
        )
        model_medium.fit(x_train, y_train, eval_set=(x_valid, y_valid))
        model_medium_best_score = model_medium.best_score_['validation']['AUC']

        model_huge = CatBoostClassifier(
            depth=10, iterations=10_000, learning_rate=0.01, eval_metric='AUC', early_stopping_rounds=50,
            verbose=False, task_type=DEVICE, random_state=RANDOM_SEED
        )
        model_huge.fit(x_train, y_train, eval_set=(x_valid, y_valid))
        model_huge_best_score = model_huge.best_score_['validation']['AUC']

        best_score = max(model_base_best_score, model_medium_best_score, model_huge_best_score)
        if best_score == model_base_best_score:
            model = model_base
        elif best_score == model_medium_best_score:
            model = model_medium
            depth = 8
        else:
            model = model_huge
            depth = 10
        print(f"++ The best is the model with depth {depth}")
        print(f"++ The scores were {model_base_best_score:.3f}, {model_medium_best_score:.3f}, {model_huge_best_score:.3f}")

        is_needed_full_training = True

    y_pred = model.predict_proba(x_valid)[:, 1]
    roc_auc_validation = roc_auc_score(y_valid, y_pred)
    print(f"++ ROC-AUC on validation is {roc_auc_validation:.5f}")

    if is_needed_full_training and len(x) < 2_000_000:
        best_iter = model.best_iteration_ + 1

        print(f"++ Training with all data, best_iter: {best_iter} ...")
        model = CatBoostClassifier(
            depth=depth, iterations=best_iter, learning_rate=0.1, eval_metric='AUC',
            verbose=False, task_type=DEVICE, random_state=RANDOM_SEED
        )
        model.fit(x, y)
    else:
        print(f"++ Not training with all data, x_length is {len(x)}")

    return model, roc_auc_validation


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
    model: CatBoostClassifier,
    test_data: pd.DataFrame,
    submit_data: pd.DataFrame
) -> pd.DataFrame:
    y_pred = model.predict_proba(test_data)[:, 1]
    submit_data['target'] = y_pred

    return submit_data


def modelling(dataset_folder: str) -> tuple[pd.DataFrame, float]:
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
    train_res = train_preprocess(train_paths)
    print("Train preprocessing done!")

    print(f"Start feature selection...")
    train_res['data'] = feature_selection(train_res['data'], train_res['target'])
    print("Feature selection done!")

    print("Start training...")
    train_res["model"], roc_auc_validation = train(train_res['data'], train_res['target'])
    print("Training done!")

    print("Start test preprocessing...")
    test_res = test_preprocess(
        test_paths,
        cat_cols=train_res['cat_cols'],
        train_cols=train_res['data'].columns.tolist()
    )
    print("Test preprocessing done!")

    print("Start prediction...")
    prediction = predict(
        train_res["model"],
        test_res["test_data"],
        test_res["submit_data"]
    )
    print("Prediction done!")

    return prediction, roc_auc_validation


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
    total_roc_auc = 0.0
    num_valid_datasets = 0

    # Создадим цикл для прохождения по каждому файлу и генерации предсказания
    for dataset_folder in datasets_folders:
        if not os.path.isdir(os.path.join(DATA_PATH, dataset_folder)):
            print(f"Skipped {dataset_folder} (not a directory)")
            continue

        start_time = time.time()
        prediction, roc_auc_val = modelling(dataset_folder)
        total_roc_auc += roc_auc_val

        # Сохраняем предсказание
        prediction.to_csv(os.path.join(PREDICTION_PATH, dataset_folder) + ".csv", index=False)

        dataset_time = time.time() - start_time
        total_time += dataset_time
        num_valid_datasets += 1
        print(f"Prediction is saved! Dataset time: {dataset_time}")
        print("\n================================")

    print(f"All datasets are done! Total time: {total_time}")
    print(f"Average ROC-AUC between all the datasets: {total_roc_auc / num_valid_datasets}")


if __name__ == "__main__":
    main()
