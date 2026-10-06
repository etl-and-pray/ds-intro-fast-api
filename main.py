from contextlib import asynccontextmanager
from typing import Optional
import dill
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# глобальные переменные для хранения загруженной модели и метаданных
model_pipeline = None
model_metadata = {}


# 1. Обработчик жизненного цикла приложения (lifespan)
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pipeline, model_metadata
    print("Загрузка модели из model/cars_pipe.pkl...")
    try:
        # указываем путь к файлу с учётом папки model/
        with open("model/cars_pipe.pkl", "rb") as file:
            payload = dill.load(file)
            model_pipeline = payload["model"]
            model_metadata = payload["metadata"]
        print("Модель и метаданные успешно загружены!")
    except Exception as e:
        print(f"Ошибка при загрузке файла модели: {e}")

    # передача управления приложению
    yield

    # код, выполняемый при завершении работы сервера
    print("Завершение работы сервиса")


# 2. Создание экземпляра FastAPI
app = FastAPI(
    title="Car Price Prediction API",
    description="Сервис для предсказания ценовой категории автомобиля",
    lifespan=lifespan
)


# 3. Pydantic-модель для входящих данных об автомобиле
class CarForm(BaseModel):
    id: int
    url: Optional[str] = None
    region: Optional[str] = None
    region_url: Optional[str] = None
    price: int
    year: Optional[float] = None
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    condition: Optional[str] = None
    cylinders: Optional[str] = None
    fuel: Optional[str] = None
    odometer: Optional[float] = None
    title_status: Optional[str] = None
    transmission: Optional[str] = None
    vin: Optional[str] = None
    drive: Optional[str] = None
    size: Optional[str] = None
    type: Optional[str] = None
    paint_color: Optional[str] = None
    image_url: Optional[str] = None
    description: Optional[str] = None
    state: Optional[str] = None
    lat: Optional[float] = None
    long: Optional[float] = None


# 4. Pydantic-модель для формата ответа предсказания
class PredictionResponse(BaseModel):
    id: int
    pred: str
    price: int


# 5. Эндпоинт GET /status — проверка работоспособности
@app.get("/status", response_model=str)
def get_status():
    """Возвращает статус работы сервиса."""
    return "I’m OK"


# 6. Эндпоинт GET /version — получение информации о модели
@app.get("/version")
def get_version():
    """Возвращает метаданные обучающей модели."""
    if not model_metadata:
        raise HTTPException(status_code=500, detail="Метаданные модели недоступны")
    return model_metadata


# 7. Эндпоинт POST /prediction — выполнение предсказания
@app.post("/prediction", response_model=PredictionResponse)
def predict(car: CarForm):
    """Принимает данные авто и возвращает предсказанную категорию цены."""
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")

    # преобразование Pydantic-модели в DataFrame
    input_data = pd.DataFrame([car.model_dump()])

    # получение результата от пайплайна
    prediction = model_pipeline.predict(input_data)[0]

    return PredictionResponse(
        id=car.id,
        pred=str(prediction),
        price=car.price
    )