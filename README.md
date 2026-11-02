# Компьютерный практикум по анализу данных авиарейсов

## Работы № 1–4. Вариант 1

---

# Компьютерная работа № 1. Первичный аудит и составление паспорта набора данных

## 1. Цель работы

Изучить структуру набора данных о выполнении авиарейсов, определить смысл и типы переменных, исследовать полноту данных и составить формализованный паспорт набора.

## 2. Индивидуальное задание (Вариант 1. Перевозчики)

- определить число рейсов каждого перевозчика;
- рассчитать долю перевозчика в наборе;
- определить число используемых бортов;
- найти перевозчиков с наибольшей долей пропусков `TailNum`;
- сделать вывод о полноте идентификации воздушных судов.

## 3. Ход выполнения

### 3.1. Импорт библиотек

```python
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 140)
pd.set_option("display.max_rows", 100)

plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["font.size"] = 11
```

### 3.2. Загрузка файла

```python
DATA_PATH = Path("../data/flights_2008.csv")

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

print("Размер таблицы:", df.shape)
display(df.head())
```

Параметр `low_memory=False` уменьшает вероятность некорректного автоматического определения типов при чтении большого CSV-файла.

### 3.3. Проверка наличия обязательных полей

```python
required_columns = {
    "Year", "Month", "DayofMonth", "DayOfWeek",
    "DepTime", "CRSDepTime", "ArrTime", "CRSArrTime",
    "UniqueCarrier", "FlightNum", "TailNum",
    "ActualElapsedTime", "CRSElapsedTime", "AirTime",
    "ArrDelay", "DepDelay", "Origin", "Dest", "Distance",
    "TaxiIn", "TaxiOut", "Cancelled", "CancellationCode",
    "Diverted", "CarrierDelay", "WeatherDelay", "NASDelay",
    "SecurityDelay", "LateAircraftDelay"
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"В файле отсутствуют обязательные поля: {sorted(missing_columns)}"
    )

print("Все обязательные поля присутствуют.")
```

### 3.4. Сохранение копии исходных данных

```python
df_raw = df.copy()
```

### 3.5. Определение размера таблицы

```python
n_rows, n_columns = df.shape

print("Число строк:", n_rows)
print("Число столбцов:", n_columns)
```

**Результат:** анализируется полный набор данных за 2008 год.

### 3.6. Просмотр начала и конца таблицы

```python
display(df.head(10))
display(df.tail(10))
```

**Наблюдения:**

- строки имеют одинаковую структуру;
- пропуски обозначены как `NaN`;
- поля времени (`DepTime`, `ArrTime`) загружены как `float64` из-за наличия пропусков;
- встречаются отменённые (`Cancelled = 1`) и перенаправленные (`Diverted = 1`) рейсы.

### 3.7. Получение информации о типах данных

```python
df.info(show_counts=True)
```

```python
type_table = pd.DataFrame({
    "column": df.columns,
    "dtype": df.dtypes.astype(str).values
})

display(type_table)
```

**Объяснение:** целочисленные по смыслу признаки (`DepTime`, `ArrTime`, `ActualElapsedTime`, `AirTime`, `TaxiIn`, `TaxiOut`, `ArrDelay`, `DepDelay`, `CarrierDelay` и др.) загружены как `float64`, поскольку содержат пропуски. Pandas не может хранить `NaN` в целочисленном столбце без специального типа `Int64`.

### 3.8. Оценка используемой памяти

```python
memory_bytes = df.memory_usage(deep=True).sum()
memory_mb = memory_bytes / 1024**2

print(f"Объём памяти: {memory_mb:.2f} Мбайт")
```

```python
memory_by_column = (
    df.memory_usage(deep=True)
      .sort_values(ascending=False)
      .rename("bytes")
      .to_frame()
)

memory_by_column["megabytes"] = (
    memory_by_column["bytes"] / 1024**2
)

display(memory_by_column.head(10))
```

**Наибольший объём памяти занимают** строковые поля `TailNum`, `Origin`, `Dest`, `UniqueCarrier`, а также числовые поля с плавающей точкой (`ArrDelay`, `DepDelay`, `ActualElapsedTime`).

### 3.9. Формирование таблицы пропусков

```python
missing_table = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_share_percent": df.isna().mean() * 100
}).sort_values(
    "missing_share_percent",
    ascending=False
)

display(missing_table.round(2))
```

**Анализ пропусков:**

| Поле | Пропуски, % | Причина |
|---|---:|---|
| `CancellationCode` | ~98 | Структурный: код есть только у отменённых рейсов |
| `CarrierDelay` … `LateAircraftDelay` | ~80 | Публикуются только для задержанных рейсов |
| `TailNum` | ~0.3 | Технические пропуски регистрации |
| `ArrTime`, `ActualElapsedTime`, `AirTime`, `ArrDelay` | ~1.5–2 | Структурные: отсутствуют у отменённых рейсов |
| `DepTime`, `DepDelay`, `TaxiOut` | ~1.5 | Структурные: отсутствуют у отменённых рейсов |
| `TaxiIn` | ~1.7 | Структурные: отсутствуют у отменённых рейсов |

**Вывод:** большой процент пропусков в `CancellationCode` и причинах задержки закономерен и не является дефектом данных.

### 3.10. Определение числа уникальных значений

```python
unique_table = pd.DataFrame({
    "unique_count": df.nunique(dropna=True),
    "missing_count": df.isna().sum()
}).sort_values("unique_count")

display(unique_table)
```

```python
print("Авиакомпаний:", df["UniqueCarrier"].nunique())
print("Воздушных судов:", df["TailNum"].nunique())
print("Аэропортов отправления:", df["Origin"].nunique())
print("Аэропортов назначения:", df["Dest"].nunique())
```

```python
airports = set(df["Origin"].dropna()) | set(df["Dest"].dropna())
print("Всего различных аэропортов:", len(airports))
```

### 3.11. Создание признака маршрута

```python
df["Route"] = (
    df["Origin"].astype("string")
    + "-"
    + df["Dest"].astype("string")
)

print("Число направленных маршрутов:", df["Route"].nunique())
```

```python
origin_text = df["Origin"].astype("string")
dest_text = df["Dest"].astype("string")

df["AirportPair"] = np.where(
    origin_text <= dest_text,
    origin_text + "-" + dest_text,
    dest_text + "-" + origin_text
)
```

### 3.12. Частотный анализ категориальных признаков

```python
display(df["UniqueCarrier"].value_counts(dropna=False).head(20))
display(df["Origin"].value_counts(dropna=False).head(20))
display(df["Dest"].value_counts(dropna=False).head(20))
display(df["Route"].value_counts(dropna=False).head(20))
```

```python
carrier_share = (
    df["UniqueCarrier"]
    .value_counts(normalize=True, dropna=False)
    .mul(100)
    .round(2)
)

display(carrier_share)
```

### 3.13. Первичная описательная статистика

```python
numeric_summary = df.describe().T
display(numeric_summary)
```

**Критическая оценка:**

- среднее расстояние (`Distance`) имеет содержательный смысл;
- средний номер рейса (`FlightNum`) — нет, это идентификатор;
- среднее `Cancelled` равно доле отменённых рейсов (поле принимает 0 и 1);
- среднее `DayOfWeek` малоинформативно.

### 3.14. Анализ статусов рейсов

```python
status_table = pd.crosstab(
    df["Cancelled"],
    df["Diverted"],
    margins=True
)

display(status_table)
```

```python
cancelled_share = df["Cancelled"].mean() * 100
diverted_share = df["Diverted"].mean() * 100

print(f"Доля отменённых рейсов: {cancelled_share:.2f}%")
print(f"Доля перенаправленных рейсов: {diverted_share:.2f}%")
```

### 3.15. Исследование пропусков в зависимости от статуса

```python
fields_for_missing_analysis = [
    "DepTime", "ArrTime", "ActualElapsedTime", "AirTime",
    "ArrDelay", "DepDelay", "TaxiIn", "TaxiOut",
    "CancellationCode"
]

missing_by_status = (
    df.groupby(["Cancelled", "Diverted"])[fields_for_missing_analysis]
      .agg(lambda column: column.isna().mean() * 100)
      .round(2)
)

display(missing_by_status)
```

**Вывод:** у отменённых рейсов практически все фактические параметры отсутствуют — это структурные пропуски. У выполненных рейсов пропуски минимальны.

### 3.16. Проверка возможного составного ключа

```python
possible_key = [
    "Year", "Month", "DayofMonth", "UniqueCarrier",
    "FlightNum", "Origin", "Dest", "CRSDepTime"
]

duplicate_key_count = df.duplicated(
    subset=possible_key,
    keep=False
).sum()

print(
    "Строк, повторяющихся по возможному составному ключу:",
    duplicate_key_count
)
```

```python
duplicate_examples = (
    df.loc[df.duplicated(subset=possible_key, keep=False)]
      .sort_values(possible_key)
)

display(duplicate_examples.head(30))
```

**Вывод:** повторы по составному ключу требуют дополнительного анализа. Их наличие не означает, что строки нужно удалять — сначала необходимо сравнить остальные поля.

## 4. Выполнение индивидуального задания. Вариант 1. Перевозчики

### 4.1. Число рейсов каждого перевозчика

```python
carrier_flights = (
    df["UniqueCarrier"]
    .value_counts(dropna=False)
    .rename("flight_count")
    .to_frame()
)

carrier_flights["share_percent"] = (
    carrier_flights["flight_count"] / len(df) * 100
).round(2)

display(carrier_flights)
```

### 4.2. Доля перевозчика в наборе

```python
carrier_share = (
    df["UniqueCarrier"]
    .value_counts(normalize=True, dropna=False)
    .mul(100)
    .round(2)
    .rename("share_percent")
    .to_frame()
)

display(carrier_share)
```

### 4.3. Число используемых бортов каждым перевозчиком

```python
carrier_tails = (
    df.groupby("UniqueCarrier")["TailNum"]
      .nunique(dropna=True)
      .rename("unique_tails")
      .to_frame()
)

display(carrier_tails)
```

### 4.4. Перевозчики с наибольшей долей пропусков `TailNum`

```python
tail_missing_by_carrier = (
    df.groupby("UniqueCarrier")["TailNum"]
      .apply(lambda s: s.isna().mean() * 100)
      .rename("tail_missing_percent")
      .sort_values(ascending=False)
      .to_frame()
)

display(tail_missing_by_carrier.round(2))
```

### 4.5. Вывод о полноте идентификации воздушных судов

**Результаты:**

- В наборе присутствует 20 перевозчиков.
- Наибольшая доля рейсов приходится на `WN` (Southwest), `AA`, `UA`, `OO`, `DL`.
- Число уникальных бортов у перевозчиков различается: у крупных сетевых перевозчиков — сотни бортов, у региональных — десятки.
- Доля пропусков `TailNum` в целом невелика (около 0.3%), но у отдельных перевозчиков может быть выше.

**Ограничения:** пропуск `TailNum` не позволяет однозначно идентифицировать воздушное судно и затрудняет анализ оборота бортов. Отсутствие `TailNum` может быть связано с особенностями регистрации региональных перевозчиков или арендованными бортами.

## 5. Паспорт набора данных

### 5.1. Общая информация

| Параметр | Значение |
|---|---|
| Название набора | flights_2008 |
| Предполагаемый источник | U.S. Department of Transportation, Bureau of Transportation Statistics |
| Период наблюдений | 2008 год |
| Число строк | ~7 000 000 (полный набор) |
| Число столбцов | 29 (исходно) |
| Формат файла | CSV |
| Единица наблюдения | Один запланированный авиарейс в конкретную дату |

### 5.2. Словарь полей

| Поле | Содержательный тип | Программный тип | Единица измерения | Момент доступности | Пропуски, % | Комментарий |
|---|---|---|---|---|---:|---|
| `Year` | календарный | int64 | год | до рейса | 0 | Год выполнения |
| `Month` | календарный | int64 | месяц | до рейса | 0 | 1–12 |
| `DayofMonth` | календарный | int64 | день | до рейса | 0 | 1–31 |
| `DayOfWeek` | календарный | int64 | день недели | до рейса | 0 | 1 — понедельник … 7 — воскресенье |
| `DepTime` | фактическое время | float64 | HHMM | после отправления | ~1.5 | Не обычное число |
| `CRSDepTime` | плановое время | int64 | HHMM | до рейса | 0 | Плановое отправление |
| `ArrTime` | фактическое время | float64 | HHMM | после прибытия | ~1.5 | Не обычное число |
| `CRSArrTime` | плановое время | int64 | HHMM | до рейса | 0 | Плановое прибытие |
| `UniqueCarrier` | идентификатор | object | код | до рейса | 0 | Код перевозчика |
| `FlightNum` | идентификатор | int64 | номер | до рейса | 0 | Номер рейса, не количественная величина |
| `TailNum` | идентификатор | object | код | до рейса | ~0.3 | Бортовой номер ВС |
| `ActualElapsedTime` | фактическое | float64 | мин | после прибытия | ~1.5 | Фактическая продолжительность |
| `CRSElapsedTime` | плановое | float64 | мин | до рейса | 0 | Плановая продолжительность |
| `AirTime` | фактическое | float64 | мин | после прибытия | ~1.5 | Время в воздухе |
| `ArrDelay` | результативное | float64 | мин | после прибытия | ~1.5 | Может быть отрицательным |
| `DepDelay` | результативное | float64 | мин | после отправления | ~1.5 | Может быть отрицательным |
| `Origin` | идентификатор | object | код | до рейса | 0 | Аэропорт отправления |
| `Dest` | идентификатор | object | код | до рейса | 0 | Аэропорт назначения |
| `Distance` | плановое | int64 | мили | до рейса | 0 | Расстояние между аэропортами |
| `TaxiIn` | фактическое | float64 | мин | после прибытия | ~1.5 | Руление после посадки |
| `TaxiOut` | фактическое | float64 | мин | после отправления | ~1.5 | Руление перед взлётом |
| `Cancelled` | результативное | int64 | 0/1 | после рейса | 0 | Признак отмены |
| `CancellationCode` | результативное | object | код | после рейса | ~98 | Только у отменённых |
| `Diverted` | результативное | int64 | 0/1 | после рейса | 0 | Признак перенаправления |
| `CarrierDelay` | результативное | float64 | мин | после рейса | ~80 | Только для задержанных |
| `WeatherDelay` | результативное | float64 | мин | после рейса | ~80 | Только для задержанных |
| `NASDelay` | результативное | float64 | мин | после рейса | ~80 | Только для задержанных |
| `SecurityDelay` | результативное | float64 | мин | после рейса | ~80 | Только для задержанных |
| `LateAircraftDelay` | результативное | float64 | мин | после рейса | ~80 | Только для задержанных |

### 5.3. Ограничения

- отсутствие непосредственных метеорологических измерений;
- отсутствие пассажирской загрузки;
- отсутствие характеристик воздушного судна (тип, возраст, вместимость);
- отсутствие данных о техническом состоянии;
- использование локального времени аэропортов;
- структурные пропуски у отменённых рейсов;
- невозможность доказать причинность только по данному набору;
- данные только за 2008 год — ограниченная временная репрезентативность.

## 6. Выводы по работе № 1

1. Набор содержит один объект наблюдения — запланированный авиарейс в конкретную дату.
2. Исходная таблица включает 29 полей, которые делятся на календарные, плановые, фактические, результативные и идентификационные.
3. Наибольшая доля пропусков характерна для `CancellationCode` (~98%) и причин задержки (~80%) — это структурные пропуски, а не дефекты данных.
4. Пропуски фактических параметров (~1.5–2%) связаны преимущественно с отменёнными рейсами.
5. `FlightNum` является идентификатором, а не количественной переменной; среднее значение номера рейса не имеет эксплуатационного смысла.
6. `DepTime`, `ArrTime` и другие поля времени загружены как `float64` из-за наличия пропусков и не должны интерпретироваться как обычные числа.
7. В наборе 20 перевозчиков; распределение рейсов по перевозчикам неравномерно.
8. Пропуски `TailNum` (~0.3%) ограничивают анализ оборота воздушных судов.
9. Повторы по возможному составному ключу требуют дополнительной проверки; автоматическое удаление недопустимо.
10. Набор пригоден для анализа задержек, отмен и маршрутной структуры, но не позволяет делать причинные выводы о влиянии погоды, техники или загрузки.

---

# Компьютерная работа № 2. Преобразование календарных данных и времени

## 1. Цель работы

Освоить корректное преобразование календарных полей и времени авиационных событий, представленного в формате `HHMM`, сформировать производные временные признаки и исследовать особенности перехода через полночь.

## 2. Индивидуальное задание (Вариант 1. Внутрисуточная структура рейсов)

- рассчитать число плановых отправлений по часам;
- построить столбчатую диаграмму;
- определить часы минимальной и максимальной интенсивности;
- сравнить рабочие и выходные дни;
- сформулировать ограничения интерпретации.

## 3. Ход выполнения

### 3.1. Загрузка исходного файла

```python
df = pd.read_csv(
    "../data/flights_2008.csv",
    low_memory=False
)
```

### 3.2. Формирование календарной даты

```python
df["FlightDate"] = pd.to_datetime(
    {
        "year": pd.to_numeric(df["Year"], errors="coerce"),
        "month": pd.to_numeric(df["Month"], errors="coerce"),
        "day": pd.to_numeric(df["DayofMonth"], errors="coerce")
    },
    errors="coerce"
)

display(
    df[["Year", "Month", "DayofMonth", "FlightDate"]].head()
)
```

```python
invalid_date_count = df["FlightDate"].isna().sum()
print("Число некорректных дат:", invalid_date_count)
```

**Результат:** некорректных дат не обнаружено.

### 3.3. Проверка дня недели

```python
df["CalculatedDayOfWeek"] = (
    df["FlightDate"].dt.dayofweek + 1
)

df["DayOfWeekMismatch"] = (
    df["CalculatedDayOfWeek"] != df["DayOfWeek"]
)

mismatch_count = df["DayOfWeekMismatch"].sum()

print("Число несоответствий дня недели:", mismatch_count)
```

```python
display(
    df.loc[
        df["DayOfWeekMismatch"],
        ["Year", "Month", "DayofMonth", "DayOfWeek",
         "CalculatedDayOfWeek", "FlightDate"]
    ].head(20)
)
```

**Результат:** несоответствий дня недели не обнаружено — календарная согласованность данных подтверждена.

### 3.4. Разработка функции проверки HHMM

```python
def validate_hhmm(series):
    values = pd.to_numeric(series, errors="coerce")

    is_integer = (
        values.isna()
        | np.isclose(values, np.round(values))
    )

    integer_values = values.round()
    hours = integer_values // 100
    minutes = integer_values % 100

    standard_time = (
        hours.between(0, 23)
        & minutes.between(0, 59)
    )

    midnight_2400 = (
        (hours == 24)
        & (minutes == 0)
    )

    valid = (
        values.isna()
        | (is_integer & (standard_time | midnight_2400))
    )

    return valid
```

### 3.5. Проверка всех полей времени

```python
time_columns = [
    "CRSDepTime", "CRSArrTime", "DepTime", "ArrTime"
]

time_validation_results = []

for column in time_columns:
    valid_mask = validate_hhmm(df[column])

    time_validation_results.append({
        "column": column,
        "missing": int(df[column].isna().sum()),
        "invalid": int((~valid_mask).sum()),
        "value_2400": int((df[column] == 2400).sum())
    })

time_validation_table = pd.DataFrame(time_validation_results)
display(time_validation_table)
```

### 3.6. Просмотр недопустимых значений

```python
for column in time_columns:
    valid_mask = validate_hhmm(df[column])

    invalid_values = (
        df.loc[~valid_mask, column]
          .value_counts(dropna=False)
          .head(20)
    )

    print(f"\nНедопустимые значения в {column}:")
    display(invalid_values)
```

**Вывод:** недопустимые значения не обнаружены; значения `2400` встречаются и обрабатываются отдельно.

### 3.7. Преобразование HHMM в минуты от начала суток

```python
def hhmm_to_minutes(series):
    values = pd.to_numeric(series, errors="coerce")

    integer_values = values.round()
    hours = integer_values // 100
    minutes = integer_values % 100

    valid_standard = (
        hours.between(0, 23)
        & minutes.between(0, 59)
    )

    valid_2400 = (
        (hours == 24)
        & (minutes == 0)
    )

    result = pd.Series(
        np.nan, index=series.index, dtype="float64"
    )

    result.loc[valid_standard] = (
        hours.loc[valid_standard] * 60
        + minutes.loc[valid_standard]
    )

    result.loc[valid_2400] = 0

    return result


def hhmm_day_shift(series):
    values = pd.to_numeric(series, errors="coerce")
    return (values == 2400).astype("int8")
```

```python
for column in time_columns:
    df[column + "_Minutes"] = hhmm_to_minutes(df[column])
    df[column + "_DayShift"] = hhmm_day_shift(df[column])
```

```python
display(
    df[["CRSDepTime", "CRSDepTime_Minutes",
        "CRSDepTime_DayShift"]].head(20)
)
```

### 3.8. Создание плановой временной метки отправления

```python
df["ScheduledDepartureDateTime"] = (
    df["FlightDate"]
    + pd.to_timedelta(df["CRSDepTime_DayShift"], unit="D")
    + pd.to_timedelta(df["CRSDepTime_Minutes"], unit="m")
)

display(
    df[["FlightDate", "CRSDepTime",
        "ScheduledDepartureDateTime"]].head()
)
```

**Пояснение:** временная метка выражена в локальном времени аэропорта отправления.

### 3.9. Создание расчётной плановой временной метки прибытия

```python
df["ScheduledArrivalDateTimeCalculated"] = (
    df["ScheduledDepartureDateTime"]
    + pd.to_timedelta(df["CRSElapsedTime"], unit="m")
)
```

**Пояснение:** это расчётная метка на условной шкале аэропорта отправления; она не обязана совпадать по часам с `CRSArrTime` из-за разных часовых поясов.

### 3.10. Выделение часа и части суток

```python
df["ScheduledDepartureHour"] = (
    df["CRSDepTime_Minutes"] // 60
).astype("Int64")
```

```python
def get_day_period(hour):
    if pd.isna(hour):
        return pd.NA
    if 0 <= hour < 6:
        return "Ночь"
    if 6 <= hour < 12:
        return "Утро"
    if 12 <= hour < 18:
        return "День"
    return "Вечер"

df["ScheduledDeparturePeriod"] = (
    df["ScheduledDepartureHour"]
    .apply(get_day_period)
    .astype("category")
)

display(df["ScheduledDeparturePeriod"].value_counts(dropna=False))
```

**Границы частей суток:** ночь 00:00–05:59, утро 06:00–11:59, день 12:00–17:59, вечер 18:00–23:59. Границы заданы исследователем и не являются универсальными.

### 3.11. Формирование признака выходного дня

```python
df["IsWeekend"] = df["DayOfWeek"].isin([6, 7])

display(pd.crosstab(df["DayOfWeek"], df["IsWeekend"]))
```

### 3.12. Формирование циклических признаков

```python
departure_hour_fractional = df["CRSDepTime_Minutes"] / 60

df["DepartureHourSin"] = np.sin(
    2 * np.pi * departure_hour_fractional / 24
)

df["DepartureHourCos"] = np.cos(
    2 * np.pi * departure_hour_fractional / 24
)

df["DayOfWeekSin"] = np.sin(
    2 * np.pi * (df["DayOfWeek"] - 1) / 7
)

df["DayOfWeekCos"] = np.cos(
    2 * np.pi * (df["DayOfWeek"] - 1) / 7
)
```

```python
cyclical_columns = [
    "DepartureHourSin", "DepartureHourCos",
    "DayOfWeekSin", "DayOfWeekCos"
]

display(df[cyclical_columns].describe())
```

**Проверка:** значения синуса и косинуса находятся в диапазоне [−1; 1] с учётом вычислительной погрешности.

### 3.13. Восстановление фактического времени отправления

```python
df["ActualDepartureDateTimeCalculated"] = (
    df["ScheduledDepartureDateTime"]
    + pd.to_timedelta(df["DepDelay"], unit="m")
)
```

```python
calculated_actual_dep_minutes = (
    df["ActualDepartureDateTimeCalculated"].dt.hour * 60
    + df["ActualDepartureDateTimeCalculated"].dt.minute
)

df["DepTimeDifferenceModuloDay"] = np.minimum(
    (calculated_actual_dep_minutes - df["DepTime_Minutes"]).abs(),
    1440 - (calculated_actual_dep_minutes - df["DepTime_Minutes"]).abs()
)

display(df["DepTimeDifferenceModuloDay"].describe())
```

**Пояснение:** способ корректнее простого объединения даты рейса с `DepTime`, так как учитывает отправление после полуночи.

### 3.14. Формирование маршрута и сортировка

```python
df["Route"] = (
    df["Origin"].astype("string")
    + "-"
    + df["Dest"].astype("string")
)

df = df.sort_values(
    ["ScheduledDepartureDateTime", "UniqueCarrier", "FlightNum"]
)
```

### 3.15. Сохранение результата

```python
output_path = Path(
    "../data/processed/flights_2008_time_processed.csv"
)

output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)
```

При наличии библиотеки `pyarrow` рекомендуется Parquet:

```python
df.to_parquet(
    "../data/processed/flights_2008_time_processed.parquet",
    index=False
)
```

## 4. Выполнение индивидуального задания. Вариант 1. Внутрисуточная структура рейсов

### 4.1. Число плановых отправлений по часам

```python
hourly_departures = (
    df["ScheduledDepartureHour"]
    .value_counts()
    .sort_index()
    .rename("flight_count")
    .to_frame()
)

display(hourly_departures)
```

### 4.2. Столбчатая диаграмма

```python
plt.figure(figsize=(11, 5))

plt.bar(
    hourly_departures.index.astype(int),
    hourly_departures["flight_count"],
    color="steelblue",
    edgecolor="white"
)

plt.title("Число плановых отправлений по часам суток")
plt.xlabel("Час суток (локальное время аэропорта отправления)")
plt.ylabel("Число рейсов")
plt.xticks(range(0, 24))
plt.grid(axis="y", alpha=0.3)
plt.show()
```
https://github.com/alisabucko19/image-/blob/main/%D0%90%D0%BB%D0%B8%D1%81%D0%B0%204.2.png

### 4.3. Часы минимальной и максимальной интенсивности

```python
peak_hour = hourly_departures["flight_count"].idxmax()
min_hour = hourly_departures["flight_count"].idxmin()

print(f"Час максимальной интенсивности: {peak_hour}:00")
print(f"Час минимальной интенсивности: {min_hour}:00")
print(f"Рейсов в час пик: {hourly_departures.loc[peak_hour, 'flight_count']}")
print(f"Рейсов в час минимума: {hourly_departures.loc[min_hour, 'flight_count']}")
```
https://github.com/alisabucko19/image-/blob/main/%D0%90%D0%BB%D0%B8%D1%81%D0%B0%204.4.png

**Результат:** максимум отправлений приходится на утренние и дневные часы (6:00–9:00 и 15:00–18:00), минимум — на ночные часы (0:00–5:00).

### 4.4. Сравнение рабочих и выходных дней

```python
weekday_hourly = (
    df.loc[~df["IsWeekend"], "ScheduledDepartureHour"]
    .value_counts()
    .sort_index()
)

weekend_hourly = (
    df.loc[df["IsWeekend"], "ScheduledDepartureHour"]
    .value_counts()
    .sort_index()
)

hourly_comparison = pd.DataFrame({
    "weekday": weekday_hourly,
    "weekend": weekend_hourly
}).fillna(0)

display(hourly_comparison)
```

```python
plt.figure(figsize=(11, 5))

plt.plot(
    hourly_comparison.index.astype(int),
    hourly_comparison["weekday"],
    marker="o",
    label="Рабочие дни"
)

plt.plot(
    hourly_comparison.index.astype(int),
    hourly_comparison["weekend"],
    marker="s",
    label="Выходные"
)

plt.title("Распределение плановых отправлений по часам: рабочие и выходные дни")
plt.xlabel("Час суток")
plt.ylabel("Число рейсов")
plt.xticks(range(0, 24))
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

### 4.5. Ограничения интерпретации

- время указано в локальном времени аэропорта отправления, поэтому суммирование по часам смешивает разные часовые пояса;
- число рейсов в выходные меньше, чем в рабочие дни, что отражает деловую структуру перевозок;
- пиковые часы могут различаться по аэропортам и перевозчикам;
- анализ не учитывает сезонные изменения расписания.

## 5. Выводы по работе № 2

1. Календарная дата успешно сформирована из полей `Year`, `Month`, `DayofMonth`; некорректных дат нет.
2. День недели, рассчитанный по календарной дате, полностью совпадает с полем `DayOfWeek` — календарная согласованность подтверждена.
3. Все поля времени (`CRSDepTime`, `CRSArrTime`, `DepTime`, `ArrTime`) содержат только допустимые значения формата HHMM; значения `2400` корректно обрабатываются со сдвигом даты.
4. Время HHMM преобразовано в минуты от начала суток, что позволяет выполнять арифметические операции.
5. Плановая временная метка отправления построена с учётом возможного сдвига даты для `2400`.
6. Расчётная плановая временная метка прибытия построена через прибавление `CRSElapsedTime`; она не совпадает по часам с `CRSArrTime` из-за разных часовых поясов.
7. Сформированы признаки часа, части суток, выходного дня и циклические кодировки часа и дня недели.
8. Фактическое время отправления восстановлено через `DepDelay`; проверка по модулю суток подтвердила согласованность.
9. Внутрисуточная структура отправлений имеет выраженные пики в утренние и дневные часы; ночные часы характеризуются минимальной интенсивностью.
10. В выходные дни число рейсов меньше, чем в рабочие; структура распределения по часам сохраняется, но абсолютные значения ниже.

---

# Компьютерная работа № 3. Контроль качества и логической согласованности данных

## 1. Цель работы

Освоить методы предметно-ориентированного контроля качества авиационных данных, выявить недопустимые и противоречивые записи, классифицировать найденные проблемы и сформировать журнал качества данных.

## 2. Индивидуальное задание (Вариант 1. Баланс продолжительности)

- исследовать `ElapsedBalance`;
- определить долю точного равенства;
- определить 1-й, 5-й, 95-й и 99-й процентили;
- сравнить баланс по перевозчикам;
- предложить обоснованный порог предупреждения.

## 3. Ход выполнения

### 3.1. Загрузка преобразованных данных

```python
df = pd.read_parquet(
    "../data/processed/flights_2008_time_processed.parquet"
)
```

Если Parquet не использовался:

```python
df = pd.read_csv(
    "../data/processed/flights_2008_time_processed.csv",
    low_memory=False
)

df["FlightDate"] = pd.to_datetime(
    df["FlightDate"],
    errors="coerce"
)
```

### 3.2. Создание таблицы результатов проверок

```python
quality_results = []

def register_check(
    check_name,
    mask,
    severity,
    description
):
    quality_results.append({
        "check": check_name,
        "severity": severity,
        "violations": int(mask.sum()),
        "share_percent": float(mask.mean() * 100),
        "description": description
    })
```

Здесь `mask` должен принимать значение `True` для строки, нарушающей проверку.

### 3.3. Проверка календарных диапазонов

```python
invalid_month = ~df["Month"].between(1, 12)
invalid_day_of_week = ~df["DayOfWeek"].between(1, 7)
invalid_day_of_month = ~df["DayofMonth"].between(1, 31)

register_check(
    "invalid_month",
    invalid_month,
    "error",
    "Месяц вне диапазона 1–12"
)

register_check(
    "invalid_day_of_week",
    invalid_day_of_week,
    "error",
    "День недели вне диапазона 1–7"
)

register_check(
    "invalid_day_of_month",
    invalid_day_of_month,
    "error",
    "День месяца вне диапазона 1–31"
)
```

### 3.4. Проверка бинарных полей

```python
invalid_cancelled = ~df["Cancelled"].isin([0, 1])
invalid_diverted = ~df["Diverted"].isin([0, 1])

register_check(
    "invalid_cancelled",
    invalid_cancelled,
    "error",
    "Cancelled должно принимать 0 или 1"
)

register_check(
    "invalid_diverted",
    invalid_diverted,
    "error",
    "Diverted должно принимать 0 или 1"
)
```

### 3.5. Проверка расстояния и продолжительности

```python
invalid_distance = (
    df["Distance"].notna()
    & (df["Distance"] <= 0)
)

invalid_scheduled_elapsed = (
    df["CRSElapsedTime"].notna()
    & (df["CRSElapsedTime"] <= 0)
)

invalid_actual_elapsed = (
    df["ActualElapsedTime"].notna()
    & (df["ActualElapsedTime"] <= 0)
)

invalid_air_time = (
    df["AirTime"].notna()
    & (df["AirTime"] <= 0)
)

negative_taxi_in = (
    df["TaxiIn"].notna()
    & (df["TaxiIn"] < 0)
)

negative_taxi_out = (
    df["TaxiOut"].notna()
    & (df["TaxiOut"] < 0)
)

register_check(
    "invalid_distance",
    invalid_distance,
    "error",
    "Расстояние должно быть положительным"
)

register_check(
    "invalid_scheduled_elapsed",
    invalid_scheduled_elapsed,
    "error",
    "Плановая продолжительность должна быть положительной"
)

register_check(
    "invalid_actual_elapsed",
    invalid_actual_elapsed,
    "error",
    "Фактическая продолжительность должна быть положительной"
)

register_check(
    "invalid_air_time",
    invalid_air_time,
    "error",
    "Время в воздухе должно быть положительным для выполненного рейса"
)

register_check(
    "negative_taxi_in",
    negative_taxi_in,
    "error",
    "TaxiIn не может быть отрицательным"
)

register_check(
    "negative_taxi_out",
    negative_taxi_out,
    "error",
    "TaxiOut не может быть отрицательным"
)
```

### 3.6. Проверка аэропортов

```python
same_origin_destination = (
    df["Origin"].notna()
    & df["Dest"].notna()
    & (df["Origin"] == df["Dest"])
)

missing_origin = df["Origin"].isna()
missing_destination = df["Dest"].isna()

register_check(
    "same_origin_destination",
    same_origin_destination,
    "warning",
    "Аэропорт отправления совпадает с аэропортом назначения"
)

register_check(
    "missing_origin",
    missing_origin,
    "error",
    "Отсутствует аэропорт отправления"
)

register_check(
    "missing_destination",
    missing_destination,
    "error",
    "Отсутствует аэропорт назначения"
)
```

Совпадение аэропортов требует проверки, но не всегда автоматически является ошибкой: возможны специальные, возвратные или ошибочно закодированные рейсы.

### 3.7. Проверка согласованности продолжительности

```python
df["ElapsedBalance"] = (
    df["ActualElapsedTime"]
    - (
        df["TaxiOut"]
        + df["AirTime"]
        + df["TaxiIn"]
    )
)

completed_mask = (
    (df["Cancelled"] == 0)
    & (df["Diverted"] == 0)
)

elapsed_balance_violation = (
    completed_mask
    & df["ElapsedBalance"].notna()
    & (df["ElapsedBalance"].abs() > 2)
)

register_check(
    "elapsed_balance_violation",
    elapsed_balance_violation,
    "warning",
    "Фактическая продолжительность не согласуется с AirTime, TaxiOut и TaxiIn"
)
```

Порог в две минуты является учебным и должен быть обоснован. Следует построить распределение балансового остатка:

```python
display(
    df.loc[
        completed_mask,
        "ElapsedBalance"
    ].describe(percentiles=[0.01, 0.05, 0.95, 0.99])
)

df.loc[
    completed_mask,
    "ElapsedBalance"
].hist(bins=41)

plt.title(
    "Распределение балансового остатка продолжительности"
)
plt.xlabel("Остаток, мин")
plt.ylabel("Число рейсов")
plt.show()
```
https://github.com/alisabucko19/image-/blob/main/%D0%90%D0%BB%D0%B8%D1%81%D0%B0%20%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0%203%20%D0%BF%D1%83%D0%BD%D0%BA%D1%82%203%D1%8E7.png

### 3.8. Проверка связи задержек

```python
df["DelayBalance"] = (
    df["ArrDelay"]
    - (
        df["DepDelay"]
        + df["ActualElapsedTime"]
        - df["CRSElapsedTime"]
    )
)

delay_balance_violation = (
    completed_mask
    & df["DelayBalance"].notna()
    & (df["DelayBalance"].abs() > 2)
)

register_check(
    "delay_balance_violation",
    delay_balance_violation,
    "warning",
    "ArrDelay не согласуется с DepDelay и продолжительностью рейса"
)

display(
    df.loc[
        completed_mask,
        "DelayBalance"
    ].describe(percentiles=[0.01, 0.05, 0.95, 0.99])
)
```

Следует объяснить знак каждого слагаемого. Если фактическая продолжительность меньше плановой, часть задержки отправления может быть компенсирована в ходе рейса.

### 3.9. Создание показателя изменения задержки

```python
df["DelayChangeDuringFlight"] = (
    df["ArrDelay"] - df["DepDelay"]
)
```

Интерпретация:

- отрицательное значение — задержка к прибытию уменьшилась;
- положительное значение — задержка увеличилась;
- ноль — задержка не изменилась.

Это не означает, что изменение произошло только в воздухе, поскольку в фактическую продолжительность входят руление перед взлётом и после посадки.

### 3.10. Проверка отменённых рейсов

```python
cancelled_mask = df["Cancelled"] == 1

cancelled_with_arrival_time = (
    cancelled_mask
    & df["ArrTime"].notna()
)

cancelled_with_air_time = (
    cancelled_mask
    & df["AirTime"].notna()
)

cancelled_without_code = (
    cancelled_mask
    & df["CancellationCode"].isna()
)

noncancelled_with_code = (
    (df["Cancelled"] == 0)
    & df["CancellationCode"].notna()
)

register_check(
    "cancelled_with_arrival_time",
    cancelled_with_arrival_time,
    "warning",
    "У отменённого рейса указано фактическое время прибытия"
)

register_check(
    "cancelled_with_air_time",
    cancelled_with_air_time,
    "warning",
    "У отменённого рейса указано время в воздухе"
)

register_check(
    "cancelled_without_code",
    cancelled_without_code,
    "warning",
    "У отменённого рейса отсутствует код причины"
)

register_check(
    "noncancelled_with_code",
    noncancelled_with_code,
    "warning",
    "Код отмены указан у неотменённого рейса"
)
```

При интерпретации необходимо учитывать документацию источника. Необычное сочетание может быть результатом особенностей регистрации, а не обязательно ошибкой.

### 3.11. Проверка перенаправленных рейсов

```python
diverted_mask = df["Diverted"] == 1

diverted_with_normal_arrival_delay = (
    diverted_mask
    & df["ArrDelay"].notna()
)

cancelled_and_diverted = (
    (df["Cancelled"] == 1)
    & (df["Diverted"] == 1)
)

register_check(
    "diverted_with_arrival_delay",
    diverted_with_normal_arrival_delay,
    "information",
    "У перенаправленного рейса заполнена ArrDelay"
)

register_check(
    "cancelled_and_diverted",
    cancelled_and_diverted,
    "warning",
    "Рейс одновременно помечен отменённым и перенаправленным"
)
```

Здесь факт заполнения `ArrDelay` не следует автоматически считать ошибкой. Требуется посмотреть правила формирования набора.

### 3.12. Проверка причин задержки

```python
delay_reason_columns = [
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay"
]

negative_delay_reason = (
    df[delay_reason_columns] < 0
).any(axis=1)

register_check(
    "negative_delay_reason",
    negative_delay_reason,
    "error",
    "Компонента причины задержки не может быть отрицательной"
)

df["DelayReasonSum"] = (
    df[delay_reason_columns]
    .sum(axis=1, min_count=1)
)

df["DelayReasonBalance"] = (
    df["ArrDelay"] - df["DelayReasonSum"]
)
```

Необходимо исследовать только строки, где причины заполнены:

```python
reason_available = df[delay_reason_columns].notna().any(axis=1)

display(
    df.loc[
        reason_available,
        [
            "ArrDelay",
            "DelayReasonSum",
            "DelayReasonBalance"
        ]
    ].describe()
)
```

Сумма причин не обязана совпадать с отрицательной или малой задержкой, поскольку категории причин обычно публикуются только для определённого класса задержанных рейсов.

### 3.13. Проверка полных дубликатов

```python
full_duplicate_mask = df.duplicated(keep=False)

register_check(
    "full_duplicates",
    full_duplicate_mask,
    "warning",
    "Полностью повторяющиеся строки"
)

print(
    "Число строк, входящих в полные дубликаты:",
    full_duplicate_mask.sum()
)
```

### 3.14. Проверка частичных дубликатов

```python
key_columns = [
    "Year",
    "Month",
    "DayofMonth",
    "UniqueCarrier",
    "FlightNum",
    "Origin",
    "Dest",
    "CRSDepTime"
]

key_duplicate_mask = df.duplicated(
    subset=key_columns,
    keep=False
)

register_check(
    "possible_key_duplicates",
    key_duplicate_mask,
    "warning",
    "Повтор по предполагаемому составному ключу"
)

display(
    df.loc[key_duplicate_mask]
      .sort_values(key_columns)
      .head(50)
)
```

### 3.15. Проверка экстремальных значений

Экстремальные значения не следует сразу считать ошибочными.

```python
extreme_departure_delay = (
    df["DepDelay"].notna()
    & (df["DepDelay"].abs() > 24 * 60)
)

extreme_arrival_delay = (
    df["ArrDelay"].notna()
    & (df["ArrDelay"].abs() > 24 * 60)
)

extreme_taxi_out = (
    df["TaxiOut"].notna()
    & (df["TaxiOut"] > 180)
)

register_check(
    "extreme_departure_delay",
    extreme_departure_delay,
    "warning",
    "Модуль задержки отправления превышает 24 часа"
)

register_check(
    "extreme_arrival_delay",
    extreme_arrival_delay,
    "warning",
    "Модуль задержки прибытия превышает 24 часа"
)

register_check(
    "extreme_taxi_out",
    extreme_taxi_out,
    "warning",
    "Время TaxiOut превышает 180 минут"
)
```

Пороги являются исследовательскими. Студент должен пояснить, почему выбран конкретный порог.

### 3.16. Формирование итоговой таблицы качества

```python
quality_table = pd.DataFrame(
    quality_results
).sort_values(
    ["severity", "violations"],
    ascending=[True, False]
)

display(quality_table)

quality_table.to_csv(
    "../reports/work_03_quality_checks.csv",
    index=False
)
```

### 3.17. Формирование флагов качества строки

```python
df["HasCriticalError"] = (
    invalid_month
    | invalid_day_of_week
    | invalid_day_of_month
    | invalid_cancelled
    | invalid_diverted
    | invalid_distance
    | invalid_scheduled_elapsed
    | negative_taxi_in
    | negative_taxi_out
    | missing_origin
    | missing_destination
    | negative_delay_reason
)

df["HasQualityWarning"] = (
    same_origin_destination
    | elapsed_balance_violation
    | delay_balance_violation
    | cancelled_with_arrival_time
    | cancelled_with_air_time
    | cancelled_without_code
    | noncancelled_with_code
    | cancelled_and_diverted
    | full_duplicate_mask
    | key_duplicate_mask
    | extreme_departure_delay
    | extreme_arrival_delay
    | extreme_taxi_out
)

print(
    "Строк с критической ошибкой:",
    df["HasCriticalError"].sum()
)

print(
    "Строк с предупреждением:",
    df["HasQualityWarning"].sum()
)
```

### 3.18. Создание аналитической выборки

Для следующей работы необходимы выполненные и неперенаправленные рейсы с известной задержкой прибытия:

```python
analysis_df = df.loc[
    (df["Cancelled"] == 0)
    & (df["Diverted"] == 0)
    & df["ArrDelay"].notna()
    & df["DepDelay"].notna()
    & ~df["HasCriticalError"]
].copy()

print("Размер аналитической выборки:", analysis_df.shape)
```

Важно: строки с предупреждениями не обязательно удаляются. Решение зависит от цели анализа.

### 3.19. Сохранение результатов

```python
df.to_parquet(
    "../data/processed/flights_2008_quality_checked.parquet",
    index=False
)

analysis_df.to_parquet(
    "../data/processed/flights_2008_analysis_sample.parquet",
    index=False
)
```

## 4. Выполнение индивидуального задания. Вариант 1. Баланс продолжительности

### 4.1. Исследование `ElapsedBalance`

```python
elapsed_balance_stats = (
    df.loc[completed_mask, "ElapsedBalance"]
    .describe(
        percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    )
)

display(elapsed_balance_stats.round(3))
```

### 4.2. Доля точного равенства

```python
exact_balance_share = (
    (df.loc[completed_mask, "ElapsedBalance"] == 0)
    .mean() * 100
)

print(
    f"Доля рейсов с точным равенством "
    f"ActualElapsedTime = TaxiOut + AirTime + TaxiIn: "
    f"{exact_balance_share:.2f}%"
)
```

### 4.3. Процентили балансового остатка

```python
percentile_table = pd.DataFrame({
    "percentile": ["1%", "5%", "95%", "99%"],
    "value_minutes": [
        df.loc[completed_mask, "ElapsedBalance"].quantile(0.01),
        df.loc[completed_mask, "ElapsedBalance"].quantile(0.05),
        df.loc[completed_mask, "ElapsedBalance"].quantile(0.95),
        df.loc[completed_mask, "ElapsedBalance"].quantile(0.99)
    ]
})

display(percentile_table.round(2))
```

**Интерпретация:**

- 1-й и 5-й процентили показывают, насколько фактическая продолжительность может быть меньше суммы составляющих;
- 95-й и 99-й процентили показывают величину положительного остатка;
- если 5-й и 95-й процентили близки к нулю, балансовая согласованность высокая.


### 4.4. Сравнение баланса по перевозчикам

```python
balance_by_carrier = (
    df.loc[completed_mask]
    .groupby("UniqueCarrier")["ElapsedBalance"]
    .agg(
        flight_count="size",
        mean_balance="mean",
        median_balance="median",
        p05=lambda s: s.quantile(0.05),
        p95=lambda s: s.quantile(0.95),
        exact_share_percent=lambda s: (s == 0).mean() * 100
    )
    .sort_values("mean_balance", ascending=False)
)

display(balance_by_carrier.round(3))
```

**Интерпретация:** различия между перевозчиками могут отражать:

- особенности округления времени;
- разную практику регистрации `TaxiIn`, `TaxiOut`;
- разную структуру маршрутной сети;
- различия в программном обеспечении учёта.
- 
  

### 4.5. Обоснование порога предупреждения

На основании распределения предлагается порог **±2 минуты**:

- он соответствует точности округления временных полей до минуты;
- он охватывает подавляющее большинство рейсов (между 5-м и 95-м процентилями);
- он не является слишком жёстким, чтобы не помечать нормальные рейсы;
- он не является слишком мягким, чтобы не пропускать реальные расхождения.

При необходимости порог может быть уточнён по результатам анализа процентилей и сравнения по перевозчикам.

## 5. Журнал обработки

| Этап | Проверка или преобразование | Число затронутых строк | Принятое решение | Обоснование |
|---|---|---:|---|---|
| 1 | Недопустимый месяц | 0 | Исключить из аналитической выборки | Невозможно определить дату |
| 2 | Недопустимый день недели | 0 | Исключить | Нарушение календарной согласованности |
| 3 | Недопустимый день месяца | 0 | Исключить | Невозможно определить дату |
| 4 | Недопустимое `Cancelled`/`Diverted` | 0 | Исключить | Невозможно определить статус |
| 5 | Неположительное расстояние | 0 | Исключить | Физически недопустимо |
| 6 | Неположительная плановая продолжительность | 0 | Исключить | Физически недопустимо |
| 7 | Отрицательные `TaxiIn`/`TaxiOut` | 0 | Исключить | Физически недопустимо |
| 8 | Совпадение `Origin` и `Dest` | Единицы | Пометить, не удалять | Возможны специальные рейсы |
| 9 | Нарушение баланса продолжительности | Несколько % | Пометить | Возможно округление |
| 10 | Нарушение баланса задержек | Несколько % | Пометить | Возможно округление |
| 11 | Отменённый рейс без `CancellationCode` | Единицы | Пометить | Возможна ошибка регистрации |
| 12 | Код отмены у неотменённого рейса | Единицы | Пометить | Возможна ошибка регистрации |
| 13 | Полные дубликаты | Единицы | Пометить, не удалять | Требуется анализ |
| 14 | Повторы по составному ключу | Единицы | Пометить, не удалять | Требуется анализ |
| 15 | Задержка более 24 часов | Единицы | Пометить, не удалять | Возможное редкое событие |
| 16 | `TaxiOut` более 180 минут | Единицы | Пометить, не удалять | Возможное редкое событие |

## 6. Выводы по работе № 3

1. Сформирована система проверок качества, включающая синтаксические, диапазонные, логические и балансовые проверки.
2. Критические ошибки (недопустимые календарные значения, отрицательные продолжительности, отсутствие аэропортов) в наборе практически отсутствуют.
3. Наибольшее число предупреждений связано с балансовыми проверками продолжительности и задержек, что объясняется округлением временных полей до минуты.
4. Структурные пропуски у отменённых рейсов подтверждены и не являются дефектами данных.
5. Полные и частичные дубликаты требуют дополнительного анализа; автоматическое удаление недопустимо.
6. Экстремальные значения задержек и `TaxiOut` помечены как предупреждения, но не удалены, поскольку могут быть реальными эксплуатационными событиями.
7. Баланс продолжительности `ActualElapsedTime ≈ TaxiOut + AirTime + TaxiIn` выполняется с точностью до нескольких минут; отклонения связаны с округлением.
8. Баланс задержек `ArrDelay ≈ DepDelay + ActualElapsedTime − CRSElapsedTime` также выполняется с точностью до нескольких минут.
9. Предложен порог предупреждения ±2 минуты, обоснованный точностью округления и распределением остатка.
10. Сформирована аналитическая выборка выполненных неперенаправленных рейсов с известными задержками, пригодная для статистического анализа.

---

# Компьютерная работа № 4. Описательный статистический анализ задержек авиарейсов

## 1. Цель работы

Освоить методы описательного статистического анализа задержек отправления и прибытия авиарейсов, научиться исследовать форму распределения, рассчитывать показатели центра и разброса, определять долю задержанных рейсов, выявлять экстремальные наблюдения и корректно интерпретировать полученные результаты с учётом особенностей авиационных данных.

## 2. Индивидуальное задание (Вариант 1. Сравнение задержек по месяцам)

Для каждого месяца определить:

- число выполненных рейсов;
- среднюю задержку отправления;
- медианную задержку отправления;
- среднюю задержку прибытия;
- медианную задержку прибытия;
- 90-й процентиль `ArrDelay`;
- долю рейсов с `ArrDelay >= 15`;
- долю рейсов с `ArrDelay >= 60`.

Построить:

- график средней и медианной задержки;
- график доли задержек более 15 минут;
- boxplot по месяцам.

Ответить:

- в каком месяце задержки наиболее велики;
- сохраняется ли результат при использовании медианы;
- совпадает ли лидер по средней задержке с лидером по 90-му процентилю;
- достаточно ли этих данных для утверждения о влиянии погоды.

## 3. Ход выполнения

### 3.1. Загрузка аналитической выборки

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path(
    "../data/processed/flights_2008_analysis_sample.parquet"
)

analysis_df = pd.read_parquet(DATA_PATH)

print("Размер аналитической выборки:", analysis_df.shape)
display(analysis_df.head())
```

Если файл аналитической выборки не был сохранён, её можно сформировать повторно:

```python
DATA_PATH = Path(
    "../data/processed/flights_2008_quality_checked.parquet"
)

df = pd.read_parquet(DATA_PATH)

analysis_mask = (
    (df["Cancelled"] == 0)
    & (df["Diverted"] == 0)
    & df["ArrDelay"].notna()
    & df["DepDelay"].notna()
)

if "HasCriticalError" in df.columns:
    analysis_mask &= ~df["HasCriticalError"]

analysis_df = df.loc[analysis_mask].copy()

print("Размер исходной таблицы:", df.shape)
print("Размер аналитической выборки:", analysis_df.shape)
```

### 3.2. Проверка аналитической выборки

```python
status_check = analysis_df[
    ["Cancelled", "Diverted"]
].value_counts(dropna=False)

display(status_check)
```

```python
delay_missing = analysis_df[
    ["DepDelay", "ArrDelay"]
].isna().sum()

display(delay_missing)
```

```python
print(
    analysis_df[
        ["DepDelay", "ArrDelay"]
    ].dtypes
)
```

**Что отразить в отчёте:**

- размер исходной таблицы;
- размер аналитической выборки;
- число исключённых отменённых рейсов;
- число исключённых перенаправленных рейсов;
- число исключённых строк с пропусками;
- правило формирования аналитической выборки.

### 3.3. Расчёт основных описательных статистик

```python
delay_columns = [
    "DepDelay",
    "ArrDelay"
]

basic_statistics = analysis_df[
    delay_columns
].describe(
    percentiles=[
        0.01,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99
    ]
).T

display(basic_statistics.round(2))
```

Дополнительно рассчитать медиану, дисперсию, асимметрию и межквартильный размах:

```python
extended_statistics = pd.DataFrame(
    index=delay_columns
)

extended_statistics["count"] = (
    analysis_df[delay_columns].count()
)

extended_statistics["mean"] = (
    analysis_df[delay_columns].mean()
)

extended_statistics["median"] = (
    analysis_df[delay_columns].median()
)

extended_statistics["variance"] = (
    analysis_df[delay_columns].var()
)

extended_statistics["std"] = (
    analysis_df[delay_columns].std()
)

extended_statistics["minimum"] = (
    analysis_df[delay_columns].min()
)

extended_statistics["maximum"] = (
    analysis_df[delay_columns].max()
)

extended_statistics["q25"] = (
    analysis_df[delay_columns].quantile(0.25)
)

extended_statistics["q75"] = (
    analysis_df[delay_columns].quantile(0.75)
)

extended_statistics["iqr"] = (
    extended_statistics["q75"]
    - extended_statistics["q25"]
)

extended_statistics["skewness"] = (
    analysis_df[delay_columns].skew()
)

display(extended_statistics.round(2))
```

**Вопросы для анализа:**

1. Какая задержка больше в среднем: отправления или прибытия?
2. Насколько среднее отличается от медианы?
3. Что это говорит о форме распределения?
4. Насколько максимальное значение отличается от 95-го и 99-го процентилей?
5. Можно ли считать максимальное значение типичным?
6. Какой показатель лучше характеризует обычный рейс — среднее или медиана?

### 3.4. Расчёт доли ранних, своевременных и задержанных рейсов

```python
def classify_delay(value):
    if pd.isna(value):
        return pd.NA
    if value < 0:
        return "Раньше расписания"
    if value < 15:
        return "От 0 до 14 минут"
    if value < 60:
        return "От 15 до 59 минут"
    if value < 180:
        return "От 60 до 179 минут"
    return "180 минут и более"


analysis_df["DepartureStatus"] = (
    analysis_df["DepDelay"]
    .apply(classify_delay)
)

analysis_df["ArrivalStatus"] = (
    analysis_df["ArrDelay"]
    .apply(classify_delay)
)
```

```python
def category_summary(series):
    count = series.value_counts(dropna=False)
    share = series.value_counts(
        normalize=True,
        dropna=False
    ) * 100

    result = pd.DataFrame({
        "count": count,
        "share_percent": share
    })

    return result


departure_status_summary = category_summary(
    analysis_df["DepartureStatus"]
)

arrival_status_summary = category_summary(
    analysis_df["ArrivalStatus"]
)

display(departure_status_summary.round(2))
display(arrival_status_summary.round(2))
```

### 3.5. Расчёт долей для нескольких порогов

```python
thresholds = [
    0,
    15,
    30,
    60,
    120,
    180
]

threshold_results = []

for threshold in thresholds:
    threshold_results.append({
        "threshold_minutes": threshold,
        "departure_count": int(
            (analysis_df["DepDelay"] >= threshold).sum()
        ),
        "departure_share_percent": float(
            (analysis_df["DepDelay"] >= threshold).mean() * 100
        ),
        "arrival_count": int(
            (analysis_df["ArrDelay"] >= threshold).sum()
        ),
        "arrival_share_percent": float(
            (analysis_df["ArrDelay"] >= threshold).mean() * 100
        )
    })

threshold_table = pd.DataFrame(
    threshold_results
)

display(threshold_table.round(2))
```

**Методический вывод:** фраза «доля задержанных рейсов» неполна без указания показателя, порогового значения, правила включения границы и состава аналитической выборки.

### 3.6. Построение гистограммы задержки отправления

```python
plt.figure(figsize=(11, 5))

plt.hist(
    analysis_df["DepDelay"],
    bins=100,
    color="steelblue",
    edgecolor="white"
)

plt.title(
    "Распределение задержки отправления"
)
plt.xlabel("Задержка отправления, мин")
plt.ylabel("Число рейсов")
plt.grid(alpha=0.3)
plt.show()
```

Из-за длинного правого хвоста основная часть распределения может оказаться плохо различимой. Поэтому следует построить график центральной области.

```python
dep_low = analysis_df["DepDelay"].quantile(0.01)
dep_high = analysis_df["DepDelay"].quantile(0.99)

dep_central = analysis_df.loc[
    analysis_df["DepDelay"].between(
        dep_low,
        dep_high
    ),
    "DepDelay"
]

plt.figure(figsize=(11, 5))

plt.hist(
    dep_central,
    bins=80,
    color="steelblue",
    edgecolor="white"
)

plt.axvline(
    analysis_df["DepDelay"].mean(),
    color="red",
    linestyle="--",
    linewidth=2,
    label="Среднее"
)

plt.axvline(
    analysis_df["DepDelay"].median(),
    color="black",
    linestyle="-",
    linewidth=2,
    label="Медиана"
)

plt.title(
    "Центральные 98% распределения задержки отправления"
)
plt.xlabel("Задержка отправления, мин")
plt.ylabel("Число рейсов")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

**Обязательное замечание:** ограничение диапазона графика не означает удаления наблюдений из расчёта. В подписи необходимо указать, что показаны, например, центральные 98% значений.

### 3.7. Построение гистограммы задержки прибытия

```python
arr_low = analysis_df["ArrDelay"].quantile(0.01)
arr_high = analysis_df["ArrDelay"].quantile(0.99)

arr_central = analysis_df.loc[
    analysis_df["ArrDelay"].between(
        arr_low,
        arr_high
    ),
    "ArrDelay"
]

plt.figure(figsize=(11, 5))

plt.hist(
    arr_central,
    bins=80,
    color="darkorange",
    edgecolor="white"
)

plt.axvline(
    analysis_df["ArrDelay"].mean(),
    color="red",
    linestyle="--",
    linewidth=2,
    label="Среднее"
)

plt.axvline(
    analysis_df["ArrDelay"].median(),
    color="black",
    linestyle="-",
    linewidth=2,
    label="Медиана"
)

plt.axvline(
    15,
    color="darkgreen",
    linestyle=":",
    linewidth=2,
    label="Порог 15 минут"
)

plt.title(
    "Центральные 98% распределения задержки прибытия"
)
plt.xlabel("Задержка прибытия, мин")
plt.ylabel("Число рейсов")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

**Необходимо проанализировать:**

- положение основной массы значений;
- наличие отрицательных задержек;
- положение среднего и медианы;
- длину правого хвоста;
- долю значений за порогом 15 минут;
- отличие распределения от симметричного.

https://github.com/alisabucko19/image-/blob/main/%D0%90%D0%BB%D0%B8%D1%81%D0%B0%20%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0%204%20%D0%BF%D1%83%D0%BD%D0%BA%D1%82%204.3.png 

### 3.8. Сравнение задержек отправления и прибытия

```python
comparison_low = min(
    analysis_df["DepDelay"].quantile(0.01),
    analysis_df["ArrDelay"].quantile(0.01)
)

comparison_high = max(
    analysis_df["DepDelay"].quantile(0.99),
    analysis_df["ArrDelay"].quantile(0.99)
)

plot_sample = analysis_df.loc[
    analysis_df["DepDelay"].between(
        comparison_low,
        comparison_high
    )
    & analysis_df["ArrDelay"].between(
        comparison_low,
        comparison_high
    )
]

plt.figure(figsize=(11, 5))

plt.hist(
    plot_sample["DepDelay"],
    bins=80,
    alpha=0.55,
    label="Задержка отправления",
    color="steelblue"
)

plt.hist(
    plot_sample["ArrDelay"],
    bins=80,
    alpha=0.55,
    label="Задержка прибытия",
    color="darkorange"
)

plt.title(
    "Сравнение распределений задержки отправления и прибытия"
)
plt.xlabel("Задержка, мин")
plt.ylabel("Число рейсов")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

При наложении гистограмм необходимо использовать одинаковые границы диапазона, одинаковое число интервалов, одинаковую выборку и прозрачность.

https://github.com/alisabucko19/image-/blob/main/%D0%90%D0%BB%D0%B8%D1%81%D0%B0%20%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0%204%20%D0%BF%D1%83%D0%BD%D0%BA%D1%82%204.4.png

### 3.9. Построение диаграмм размаха

```python
plt.figure(figsize=(9, 5))

plt.boxplot(
    [
        analysis_df["DepDelay"].dropna(),
        analysis_df["ArrDelay"].dropna()
    ],
    tick_labels=[
        "Отправление",
        "Прибытие"
    ],
    showfliers=False
)

plt.title(
    "Задержки отправления и прибытия без отображения отдельных выбросов"
)
plt.ylabel("Задержка, мин")
plt.grid(axis="y", alpha=0.3)
plt.show()
```

Отдельно можно построить boxplot с отображением всех значений:

```python
plt.figure(figsize=(9, 5))

plt.boxplot(
    [
        analysis_df["DepDelay"].dropna(),
        analysis_df["ArrDelay"].dropna()
    ],
    tick_labels=[
        "Отправление",
        "Прибытие"
    ],
    showfliers=True
)

plt.title(
    "Полный boxplot задержек отправления и прибытия"
)
plt.ylabel("Задержка, мин")
plt.grid(axis="y", alpha=0.3)
plt.show()
```

Студент должен объяснить, почему при длинном правом хвосте центральная часть полного boxplot может выглядеть сжатой.

https://github.com/alisabucko19/image-/blob/main/%D0%90%D0%BB%D0%B8%D1%81%D0%B0%20%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0%204%20%D0%BF%D1%83%D0%BD%D0%BA%D1%82%204.4.png

### 3.10. Поиск статистических выбросов методом IQR

```python
q1 = analysis_df["ArrDelay"].quantile(0.25)
q3 = analysis_df["ArrDelay"].quantile(0.75)
iqr = q3 - q1

lower_iqr_bound = q1 - 1.5 * iqr
upper_iqr_bound = q3 + 1.5 * iqr

print("Первый квартиль:", q1)
print("Третий квартиль:", q3)
print("Межквартильный размах:", iqr)
print("Нижняя граница:", lower_iqr_bound)
print("Верхняя граница:", upper_iqr_bound)
```

```python
analysis_df["ArrDelayIQRFlag"] = (
    (analysis_df["ArrDelay"] < lower_iqr_bound)
    | (analysis_df["ArrDelay"] > upper_iqr_bound)
)

iqr_outlier_count = (
    analysis_df["ArrDelayIQRFlag"].sum()
)

iqr_outlier_share = (
    analysis_df["ArrDelayIQRFlag"].mean() * 100
)

print(
    "Число статистических выбросов:",
    iqr_outlier_count
)

print(
    f"Доля статистических выбросов: "
    f"{iqr_outlier_share:.2f}%"
)
```

**Принципиальное замечание:** флаг IQR означает только статистическую удалённость значения. Он не является доказательством ошибки. Необходимо сравнить число статистических выбросов, число значений, нарушающих предметные ограничения, и число экстремальных, но логически согласованных рейсов.

https://github.com/alisabucko19/image-/blob/main/%D0%90%D0%BB%D0%B8%D1%81%D0%B0%20%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0%204%20%D0%BF%D1%83%D0%BD%D0%BA%D1%82%204.5.png 

### 3.11. Исследование экстремальных задержек

```python
extreme_columns = [
    "FlightDate",
    "UniqueCarrier",
    "FlightNum",
    "TailNum",
    "Origin",
    "Dest",
    "CRSDepTime",
    "DepTime",
    "DepDelay",
    "CRSArrTime",
    "ArrTime",
    "ArrDelay",
    "ActualElapsedTime",
    "CRSElapsedTime",
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",[link](https://www.example.com)
    "SecurityDelay",
    "LateAircraftDelay"
]

available_extreme_columns = [
    column
    for column in extreme_columns
    if column in analysis_df.columns
]

largest_arrival_delays = (
    analysis_df
    .sort_values(
        "ArrDelay",
        ascending=False
    )
    [available_extreme_columns]
    .head(20)
)

display(largest_arrival_delays)
```

```python
earliest_arrivals = (
    analysis_df
    .sort_values(
        "ArrDelay",
        ascending=True
    )
    [available_extreme_columns]
    .head(20)
)

display(earliest_arrivals)
```
https://github.com/alisabucko19/image-/blob/main/%D0%90%D0%BB%D0%B8%D1%81%D0%B0%20%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0%204%20%D0%BF%D1%83%D0%BD%D0%BA%D1%82%204.6.png

**Для каждого экстремального случая проверить:**

1. Согласуется ли `ArrDelay` с `DepDelay`?
2. Согласуется ли фактическая продолжительность с плановой?
3. Заполнены ли причины задержки?
4. Не является ли рейс отменённым или перенаправленным?
5. Корректны ли плановые и фактические времена?
6. Имеются ли признаки ошибки данных?
7. Может ли значение быть реальным нерегулярным событием?

### 3.12. Анализ изменения задержки между отправлением и прибытием

```python
analysis_df["DelayChange"] = (
    analysis_df["ArrDelay"]
    - analysis_df["DepDelay"]
)
```

```python
delay_change_statistics = (
    analysis_df["DelayChange"]
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99
        ]
    )
)

display(delay_change_statistics.round(2))
```

```python
def classify_delay_change(value):
    if pd.isna(value):
        return pd.NA
    if value < 0:
        return "Задержка уменьшилась"
    if value > 0:
        return "Задержка увеличилась"
    return "Без изменения"


analysis_df["DelayChangeStatus"] = (
    analysis_df["DelayChange"]
    .apply(classify_delay_change)
)

delay_change_summary = category_summary(
    analysis_df["DelayChangeStatus"]
)

display(delay_change_summary.round(2))
```

```python
change_low = (
    analysis_df["DelayChange"].quantile(0.01)
)

change_high = (
    analysis_df["DelayChange"].quantile(0.99)
)

delay_change_central = analysis_df.loc[
    analysis_df["DelayChange"].between(
        change_low,
        change_high
    ),
    "DelayChange"
]

plt.figure(figsize=(11, 5))

plt.hist(
    delay_change_central,
    bins=80,
    color="mediumpurple",
    edgecolor="white"
)

plt.axvline(
    0,
    color="black",
    linestyle="--",
    linewidth=2
)

plt.title(
    "Изменение задержки между отправлением и прибытием"
)
plt.xlabel(
    "ArrDelay − DepDelay, мин"
)
plt.ylabel("Число рейсов")
plt.grid(alpha=0.3)
plt.show()
```
https://github.com/alisabucko19/image-/blob/main/%D0%90%D0%BB%D0%B8%D1%81%D0%B0%20%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0%204%20%D0%BF%D1%83%D0%BD%D0%BA%D1%82%204.7.png

### 3.13. Исследование связи задержки отправления и прибытия

```python
sample_size = min(
    50_000,
    len(analysis_df)
)

scatter_sample = analysis_df.sample(
    n=sample_size,
    random_state=42
)

plt.figure(figsize=(8, 7))

plt.scatter(
    scatter_sample["DepDelay"],
    scatter_sample["ArrDelay"],
    alpha=0.15,
    s=10,
    color="steelblue"
)

plot_min = min(
    scatter_sample["DepDelay"].quantile(0.01),
    scatter_sample["ArrDelay"].quantile(0.01)
)

plot_max = max(
    scatter_sample["DepDelay"].quantile(0.99),
    scatter_sample["ArrDelay"].quantile(0.99)
)

plt.plot(
    [plot_min, plot_max],
    [plot_min, plot_max],
    color="red",
    linestyle="--",
    label="ArrDelay = DepDelay"
)

plt.xlim(plot_min, plot_max)
plt.ylim(plot_min, plot_max)

plt.title(
    "Связь задержки отправления и прибытия"
)
plt.xlabel("Задержка отправления, мин")
plt.ylabel("Задержка прибытия, мин")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

```python
pearson_correlation = analysis_df[
    ["DepDelay", "ArrDelay"]
].corr(
    method="pearson"
).iloc[0, 1]

spearman_correlation = analysis_df[
    ["DepDelay", "ArrDelay"]
].corr(
    method="spearman"
).iloc[0, 1]

print(
    f"Корреляция Пирсона: "
    f"{pearson_correlation:.3f}"
)

print(
    f"Корреляция Спирмена: "
    f"{spearman_correlation:.3f}"
)
```

**Интерпретация:**

- корреляция Пирсона характеризует линейную связь;
- корреляция Спирмена характеризует монотонную связь по рангам;
- высокая корреляция не означает полного совпадения задержек;
- высокая корреляция не доказывает единственную причинную связь;
- наличие связи ожидаемо, поскольку задержка отправления часто переносится на прибытие.

https://github.com/alisabucko19/image-/blob/main/%D0%90%D0%BB%D0%B8%D1%81%D0%B0%20%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0%204%20%D0%BF%D1%83%D0%BD%D0%BA%D1%82%204.7.png

### 3.14. Анализ компенсации задержки отправления

```python
departure_delayed = analysis_df.loc[
    analysis_df["DepDelay"] >= 15
].copy()

departure_delayed["RecoveredMinutes"] = (
    departure_delayed["DepDelay"]
    - departure_delayed["ArrDelay"]
)
```

```python
recovery_summary = pd.Series({
    "Число рейсов с DepDelay >= 15": len(
        departure_delayed
    ),
    "Задержка уменьшилась, %": (
        departure_delayed["RecoveredMinutes"] > 0
    ).mean() * 100,
    "Задержка не изменилась, %": (
        departure_delayed["RecoveredMinutes"] == 0
    ).mean() * 100,
    "Задержка увеличилась, %": (
        departure_delayed["RecoveredMinutes"] < 0
    ).mean() * 100,
    "Среднее уменьшение задержки, мин": (
        departure_delayed["RecoveredMinutes"].mean()
    ),
    "Медианное уменьшение задержки, мин": (
        departure_delayed["RecoveredMinutes"].median()
    )
})

display(recovery_summary.round(2))
```

```python
recovered_to_below_15 = (
    (departure_delayed["ArrDelay"] < 15)
    .mean()
    * 100
)

print(
    "Доля рейсов, снизивших задержку прибытия "
    f"до уровня менее 15 минут: "
    f"{recovered_to_below_15:.2f}%"
)
```

### 3.15. Построение эмпирической функции распределения

```python
arr_delay_sorted = np.sort(
    analysis_df["ArrDelay"].dropna().to_numpy()
)

cumulative_probability = (
    np.arange(
        1,
        len(arr_delay_sorted) + 1
    )
    / len(arr_delay_sorted)
)

plt.figure(figsize=(11, 5))

plt.plot(
    arr_delay_sorted,
    cumulative_probability,
    color="darkorange"
)

plt.axvline(
    15,
    color="red",
    linestyle="--",
    label="15 минут"
)

plt.axhline(
    0.90,
    color="gray",
    linestyle=":",
    label="90% рейсов"
)

plt.xlim(
    analysis_df["ArrDelay"].quantile(0.01),
    analysis_df["ArrDelay"].quantile(0.99)
)

plt.title(
    "Эмпирическая функция распределения задержки прибытия"
)
plt.xlabel("Задержка прибытия, мин")
plt.ylabel("Доля рейсов с задержкой не более указанной")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

По графику можно оценить долю рейсов, прибывших не позднее заданного порога, и значение задержки, которое не превышают 90% или 95% рейсов.

https://github.com/alisabucko19/image-/blob/main/%D0%90%D0%BB%D0%B8%D1%81%D0%B0%20%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0%204%20%D0%BF%D1%83%D0%BD%D0%BA%D1%82%204.8.png

### 3.16. Сохранение результатов

```python
REPORTS_PATH = Path("../reports")
REPORTS_PATH.mkdir(
    parents=True,
    exist_ok=True
)

extended_statistics.to_csv(
    REPORTS_PATH / "work_04_delay_statistics.csv"
)

threshold_table.to_csv(
    REPORTS_PATH / "work_04_delay_thresholds.csv",
    index=False
)
```

```python
analysis_df.to_parquet(
    "../data/processed/"
    "flights_2008_delay_analysis.parquet",
    index=False
)
```

## 4. Выполнение индивидуального задания. Вариант 1. Сравнение задержек по месяцам

### 4.1. Сводная таблица по месяцам

```python
monthly_delay_summary = (
    analysis_df
    .groupby("Month")
    .agg(
        flight_count=("ArrDelay", "size"),
        mean_dep_delay=("DepDelay", "mean"),
        median_dep_delay=("DepDelay", "median"),
        mean_arr_delay=("ArrDelay", "mean"),
        median_arr_delay=("ArrDelay", "median"),
        p90_arr_delay=(
            "ArrDelay",
            lambda values: values.quantile(0.90)
        ),
        delayed_15_share=(
            "ArrDelay",
            lambda values: (values >= 15).mean() * 100
        ),
        delayed_60_share=(
            "ArrDelay",
            lambda values: (values >= 60).mean() * 100
        )
    )
    .reset_index()
)

display(monthly_delay_summary.round(2))
```

### 4.2. График средней и медианной задержки

```python
plt.figure(figsize=(11, 5))

plt.plot(
    monthly_delay_summary["Month"],
    monthly_delay_summary["mean_arr_delay"],
    marker="o",
    label="Средняя ArrDelay"
)

plt.plot(
    monthly_delay_summary["Month"],
    monthly_delay_summary["median_arr_delay"],
    marker="s",
    label="Медианная ArrDelay"
)

plt.plot(
    monthly_delay_summary["Month"],
    monthly_delay_summary["mean_dep_delay"],
    marker="^",
    linestyle="--",
    label="Средняя DepDelay"
)

plt.plot(
    monthly_delay_summary["Month"],
    monthly_delay_summary["median_dep_delay"],
    marker="v",
    linestyle="--",
    label="Медианная DepDelay"
)

plt.xticks(range(1, 13))
plt.title(
    "Средняя и медианная задержка по месяцам"
)
plt.xlabel("Месяц")
plt.ylabel("Задержка, мин")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```
https://github.com/alisabucko19/image-/blob/main/%D0%90%D0%BB%D0%B8%D1%81%D0%B0%20%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0%204%20%D0%BF%D1%83%D0%BD%D0%BA%D1%82%204.9.png

### 4.3. График доли задержек более 15 минут

```python
plt.figure(figsize=(11, 5))

plt.bar(
    monthly_delay_summary["Month"],
    monthly_delay_summary["delayed_15_share"],
    color="darkorange",
    edgecolor="white"
)

plt.xticks(range(1, 13))
plt.title(
    "Доля рейсов с задержкой прибытия не менее 15 минут по месяцам"
)
plt.xlabel("Месяц")
plt.ylabel("Доля рейсов, %")
plt.grid(axis="y", alpha=0.3)
plt.show()
```
https://github.com/alisabucko19/image-/blob/main/%D0%90%D0%BB%D0%B8%D1%81%D0%B0%204.10%20%D1%80%204.png

### 4.4. Boxplot по месяцам

```python
months_data = [
    analysis_df.loc[
        analysis_df["Month"] == month,
        "ArrDelay"
    ].dropna()
    for month in range(1, 13)
]

plt.figure(figsize=(11, 5))

plt.boxplot(
    months_data,
    tick_labels=list(range(1, 13)),
    showfliers=False
)

plt.title(
    "Распределение задержки прибытия по месяцам (без выбросов)"
)
plt.xlabel("Месяц")
plt.ylabel("Задержка прибытия, мин")
plt.grid(axis="y", alpha=0.3)
plt.show()
```
https://github.com/alisabucko19/image-/blob/main/%D0%90%D0%BB%D0%B8%D1%81%D0%B0%20%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0%204%20%D0%BF%D1%83%D0%BD%D0%BA%D1%82%204.11.png

### 4.5. Ответы на вопросы индивидуального задания

**В каком месяце задержки наиболее велики?**

По средней задержке прибытия наибольшие значения приходятся на летние месяцы (июнь–август) и декабрь. Это может быть связано с сезонным увеличением объёма перевозок, погодными условиями и праздничными периодами.

**Сохраняется ли результат при использовании медианы?**

При использовании медианы лидеры могут измениться: медиана менее чувствительна к экстремальным задержкам. Если месяц лидирует и по средней, и по медиане, это указывает на систематически повышенный уровень задержек, а не на единичные экстремальные события.

**Совпадает ли лидер по средней задержке с лидером по 90-му процентилю?**

90-й процентиль характеризует правый хвост распределения. Если лидер по средней не совпадает с лидером по 90-му процентилю, это означает, что в одном месяце преобладают умеренные задержки, а в другом — редкие, но очень большие.

**Достаточно ли этих данных для утверждения о влиянии погоды?**

Нет. Набор не содержит метеорологических измерений. Наблюдаемые различия по месяцам могут быть связаны с:

- погодой;
- объёмом перевозок;
- структурой маршрутов;
- составом перевозчиков;
- аэропортами;
- изменением расписания;
- нерегулярными событиями.

Для утверждения о влиянии погоды требуется дополнительный анализ с привлечением метеорологических данных.


## 5. Обязательные результаты работы

Студент должен получить и представить:

1. таблицу основных описательных статистик;
2. таблицу расширенных статистик;
3. таблицу долей задержек для нескольких порогов;
4. классификацию рейсов по величине задержки;
5. гистограмму задержки отправления;
6. гистограмму задержки прибытия;
7. сравнительный график двух распределений;
8. boxplot задержек;
9. результаты выявления выбросов методом IQR;
10. таблицу экстремальных рейсов;
11. распределение изменения задержки;
12. диаграмму связи `DepDelay` и `ArrDelay`;
13. коэффициенты корреляции;
14. анализ компенсации задержки;
15. результаты индивидуального задания;
16. содержательные выводы и ограничения анализа.

## 6. Выводы по работе № 4

1. Аналитическая выборка включает выполненные неперенаправленные рейсы с известными задержками отправления и прибытия.
2. Распределение задержек имеет выраженную положительную асимметрию: среднее значение превышает медиану, правый хвост длинный.
3. Средняя задержка прибытия больше медианной, что подтверждает влияние редких больших задержек.
4. Доля рейсов с задержкой прибытия не менее 15 минут составляет значительную часть выборки; доля задержек не менее 60 минут существенно ниже.
5. Высокие процентили (90-й, 95-й, 99-й) значительно превышают медиану, что важно для оценки эксплуатационной устойчивости.
6. Задержка отправления и задержка прибытия strongly correlated, но не совпадают: часть задержки компенсируется в ходе рейса.
7. У значительной доли поздно отправившихся рейсов задержка к прибытию уменьшается, что может быть связано с резервом расписания.
8. Экстремальные задержки существуют и не являются автоматически ошибками данных; их следует проверять, но не удалять без обоснования.
9. Метод IQR позволяет выявить статистические выбросы, но не доказывает их ошибочность.
10. Сравнение по месяцам показывает сезонные различия в уровне задержек, однако без метеорологических данных нельзя утверждать, что причина именно в погоде.
11. Для корректной интерпретации необходимо одновременно рассматривать среднее, медиану, квантили и доли превышения порогов.
12. Корреляционная связь между `DepDelay` и `ArrDelay` не доказывает причинно-следственную связь, хотя и ожидаема содержательно.
