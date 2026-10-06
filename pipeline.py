import os
import warnings
from datetime import datetime
import dill
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=FutureWarning)


# 1. Автономная функция фильтрации колонок
def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет неинформативные колонки из датафрейма."""
    columns_to_drop = [
        'id', 'url', 'region', 'region_url', 'price',
        'manufacturer', 'image_url', 'description', 'posting_date',
        'lat', 'long'
    ]
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    return df.drop(columns=existing_columns)


# 2. Автономная функция удаления выбросов (IQR внутри функции)
def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Сглаживает выбросы в колонке 'year' по методу межквартильного размаха (IQR)."""
    df_copy = df.copy()

    def calculate_outliers(data):
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        iqr = q75 - q25
        return (q25 - 1.5 * iqr, q75 + 1.5 * iqr)

    if 'year' in df_copy.columns:
        boundaries = calculate_outliers(df_copy['year'])
        df_copy.loc[df_copy['year'] < boundaries[0], 'year'] = round(boundaries[0])
        df_copy.loc[df_copy['year'] > boundaries[1], 'year'] = round(boundaries[1])

    return df_copy


# 3. Автономная функция генeрации признаков
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Создает новые признаки: short_model и age_category."""
    df_copy = df.copy()

    # вложенная функция для безопасного выделения первого слова модели
    def extract_short_model(x):
        import pandas as pd_inner  # изолированный импорт для автономности
        if pd_inner.notnull(x):
            return str(x).lower().split(' ')[0]
        return x

    if 'model' in df_copy.columns:
        df_copy['short_model'] = df_copy['model'].apply(extract_short_model)
        df_copy = df_copy.drop(columns=['model'])

    if 'year' in df_copy.columns:
        df_copy['age_category'] = df_copy['year'].apply(
            lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average')
        )

    return df_copy


def main():
    print("Автоматический конвейер обучения и сериализации модели")

    # загрузка данных
    try:
        df = pd.read_csv('data/homework.csv')
    except FileNotFoundError:
        print("Ошибка: Файл 'data/homework.csv' не найден. Поместите его в папку 'data/'.")
        return

    X = df.drop(columns=['price_category'])
    y = df['price_category']

    # селекторы колонок по типам данных
    numerical_features = make_column_selector(dtype_include=['int64', 'float64'])
    categorical_features = make_column_selector(dtype_include=object)

    # преобразователи для числовых и категориальных признаков
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # объединение трансформеров
    column_transformer = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    # общий пайплайн предобработки
    preprocessor = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        ('outlier_remover', FunctionTransformer(remove_outliers)),
        ('feature_creator', FunctionTransformer(create_features)),
        ('column_transformer', column_transformer)
    ])

    # кандидаты моделей
    models = [
        LogisticRegression(solver='liblinear', max_iter=1000),
        RandomForestClassifier(random_state=42),
        SVC()
    ]

    best_score = 0.0
    best_pipe = None

    print("\nОценка моделей методом кросс-валидации (cv=4):")
    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        scores = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
        mean_acc = scores.mean()
        std_acc = scores.std()

        print(f"Модель: {type(model).__name__:<22} | Accuracy: {mean_acc:.4f} (std: {std_acc:.4f})")

        if mean_acc > best_score:
            best_score = mean_acc
            best_pipe = pipe

    # финальное обучение лучшей модели и сохранение в pickle (dill)
    if best_pipe is not None:
        best_model_name = type(best_pipe.named_steps['classifier']).__name__
        print(f"\nВыбрана лучшая модель: {best_model_name} (Accuracy: {best_score:.4f})")
        print("Обучение лучшей модели на полном датасете")

        best_pipe.fit(X, y)

        # формирование метаданных и сохранение
        model_payload = {
            'model': best_pipe,
            'metadata': {
                'name': 'Car price prediction model',
                'author': 'Peter Emelianov',
                'version': 1,
                'date': datetime.now().isoformat(),
                'type': best_model_name,
                'accuracy': round(best_score, 4)
            }
        }

        os.makedirs('model', exist_ok=True)
        output_filename = 'model/cars_pipe.pkl'
        with open(output_filename, 'wb') as file:
            dill.dump(model_payload, file)

        print(f"Пайплайн и метаданные успешно сохранены в файл '{output_filename}'")


if __name__ == '__main__':
    main()