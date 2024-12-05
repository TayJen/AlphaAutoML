# Импортируем модель классификатора CatBoost
from catboost import CatBoostClassifier, Pool
# Импортируем модуль для работы с данными
import pandas as pd
# Импортируем модуль для работы с системой
import os

# Функция для обучения на 1 наборе данных
def fitting(path):
    try:
        # Получим список файлов по выданному пути к папке
        current_data = os.listdir(path)
    # В случае ошибки отловим ее и прервем выполнение цикла
    except Exception:
        return "Не папка"
    # В случае отсутствия ошибки - сохраним данные
    else:
        current_data = os.listdir(path)
    # Выделим тренировочный датасет
    train_data = [data for data in current_data if data.endswith('train.parquet')][0]
    # Выделим тестовый датасет
    test_data = [data for data in current_data if data.endswith('test.parquet')][0]
    # Откроем тренировочные данные
    print("reading data", path)
    train_data = pd.read_parquet(path + f'/{train_data}')
    # Откроем тестовые данные
    test_data = pd.read_parquet(path + f'/{test_data}')
    # Выделим список категориальных столбцов
    print("Training started", path)
    cat_features = train_data.select_dtypes(include=['object', 'category']).columns.tolist()
    # Создаем Pool для обработки данных
    train_pool = Pool(
        # Выделяем часть с признаками
        data=train_data.drop('target', axis=1),
        # Выделяем часть с предсказаниями
        label=train_data['target'], 
        # Выделяем категориальные признакаи
        cat_features=cat_features
    )
    # Инициалищируем модель
    classificator = CatBoostClassifier(
        # Максимальное число итераций - 1000
        iterations=1000,
        # Число ядер для работы - 16
        thread_count=16,
        # Задаем максимальную глубину дерева - 6
        depth=6,
        # Явно укажем вид решаемой задачи - бинарная классификация
        loss_function='Logloss',
        # Указываем метрику задачи - ROC-AUC
        custom_metric='AUC',
        # Добавим информацию о том, как часто выводим информацию
        verbose = 1000,
        # Фиксируем случайные числа
        random_seed=42,
        # Обозначаем тип решаемой задачи
        task_type="GPU",
        # Обозначаем идентификатор GPU, на котором будем обучаться
        devices='0'
    )
    # Обучим модель на тренировочных данных
    classificator.fit(train_pool)
    # Создаем Pool для тестовых данных
    test_pool = Pool(
        # Выделяем часть с признаками
        data=test_data,
        # Выделяем категориальные признакаи
        cat_features=cat_features
    )
    # Создадим предсказание на тестовых данных
    model_pred = classificator.predict_proba(test_pool)[:, 1]
    # Объединим предсказание с метками
    test_data['target'] = model_pred
    # Отсортируем предсказание
    prediction = test_data[['id', 'target']].sort_values(by='id', ascending=True)
    # Вернем предсказание, как результат работы модели
    return prediction


# Функция смотрит на рабочее окружение и на папки с данными
def model():
    # Пропишем путь к файлам данных
    data = 'data'
    # Запишем список датасетов в папке:
    folders = os.listdir(data)
    # Создадим цикл для прохождения по каждому файлу и генерации предсказания
    for fold in folders:
        print("Training on", fold)
        # Запишем новый путь к данным
        data_path = data + f'/{fold}'
        # Вызовем функцию, передав в нее путь к папке для обучения
        prediction = fitting(path=data_path)
        # Сохраним полученное предсказание
        if type(prediction) is not str:
            # Сохраняем предсказание
            prediction.to_csv(f"predictions/{fold}.csv", index=False)
            print("Предсказание создано!")
        else:
            print("Невозможно создать предсказание!")

# Обозначаем действия при запуске кода
if __name__ == "__main__":
    # Запускаем модель
    model()