import requests
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss
import scipy.stats as stats
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt


def stationary_tests(rates):
    rates_series = pd.Series(rates)

    # ADF-тест (тест Дикки-Фуллера)
    adf_result = adfuller(rates_series)
    print("ADF-тест:")
    print(f"ADF статистика: {adf_result[0]}")
    print(f"p-value: {adf_result[1]}")
    print(f"Критические значения:")
    for key, value in adf_result[4].items():
        print(f"\t{key}: {value}")

    # KPSS-тест
    kpss_result = kpss(rates_series, regression="c", nlags="auto")
    print("\nKPSS-тест:")
    print(f"KPSS статистика: {kpss_result[0]}")
    print(f"p-value: {kpss_result[1]}")
    print(f"Критические значения:")
    for key, value in kpss_result[3].items():
        print(f"\t{key}: {value}")


##### ПОЛУЧЕНИЕ ДАННЫХ #####

res = requests.get(
    "https://www.cbr.ru/scripts/XML_dynamic.asp?date_req1=24/09/2023&date_req2=24/09/2024&VAL_NM_RQ=R01717"
)

root = ET.fromstring(res.content)


rates = []

for item in root.findall("Record"):
    rate = float(item.find("VunitRate").text.replace(",", "."))
    rates.append(rate)

# print(len(rates))

stationary_tests(rates)

##### ОЧИСТКА ВЫБРОСОВ #####

# Вычисление квантилей и IQR
rates = np.array(rates)
Q1 = np.percentile(rates, 25)
Q3 = np.percentile(rates, 75)
IQR = Q3 - Q1

Q1, Q3, IQR

# Определение границ
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

cleaned_rates = np.copy(rates)

# Выявление индексов выбросов
outliers = np.where((rates < lower_bound) | (rates > upper_bound))

for index in outliers[0]:
    if rates[index] < lower_bound:
        # Замена на ближайшее значение, которое больше нижней границы
        valid_values = rates[rates >= lower_bound]
        cleaned_rates[index] = valid_values[
            np.argmin(np.abs(valid_values - rates[index]))
        ]
    elif rates[index] > upper_bound:
        # Замена на ближайшее значение, которое меньше верхней границы
        valid_values = rates[rates <= upper_bound]
        cleaned_rates[index] = valid_values[
            np.argmin(np.abs(valid_values - rates[index]))
        ]

print(f"Количество выбросов: {len(outliers[0])}")
# print(f"Очищенные данные: {cleaned_rates}")

##### ТЕСТ БОКСА-ПИРСА #####

# Шаг 1: Рассчитываем автокорреляции для h лагов
h = 1  # количество лагов, которые мы будем учитывать
acf_values = sm.tsa.acf(cleaned_rates, nlags=h, fft=False)

# Шаг 2: Рассчитываем статистику Бокса-Пирса
n = len(cleaned_rates)
Q = n * np.sum(
    np.square(acf_values[1 : h + 1])
)  # Пропускаем нулевой лаг (автокорреляция с собой)

print(f"Статистика Бокса-Пирса: {Q}")


# Применение теста Бокса-Пирса (или Ljung-Box, это усовершенствованная версия теста)
test_result = acorr_ljungbox(cleaned_rates, lags=[h], return_df=True)
print(test_result)


##### ФИЛЬТР БАТТЕРВОРТА #####

order = 4  # Порядок фильтра
cutoff_frequency = 0.1  # Частота среза (в долях частоты Найквиста)

# Проектирование фильтра Баттерворта
b, a = butter(order, cutoff_frequency, btype="low", analog=False)

# Применение фильтра к данным
filtered_series = filtfilt(b, a, cleaned_rates)

##### СГЛАЖИВАНИЕ #####

alpha = 0.2  # Сглаживающий коэффициент


def exponential_moving_average(data, alpha):
    ema = np.zeros_like(data)
    ema[0] = data[0]  # Начальное значение
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema


filtered_series = np.array(filtered_series)
smoothed_rates = exponential_moving_average(filtered_series, alpha)

stationary_tests(smoothed_rates)
######


def FosterStewart(data):
    data = np.array(data)

    # Шаг 1: Подсчет увеличений и уменьшений
    n = len(data)
    increments = np.sign(
        np.diff(data)
    )  # Получаем +1 для увеличений и -1 для уменьшений

    # Шаг 2: Подсчет статистики Z
    N_positive = np.sum(increments > 0)  # Количество положительных шагов (увеличений)
    N_negative = np.sum(increments < 0)  # Количество отрицательных шагов (уменьшений)

    # Шаг 3: Вычисление статистики
    mean = (n - 1) / 2
    variance = (n + 1) / 12
    z_statistic = (N_positive - mean) / np.sqrt(variance)

    # Шаг 4: p-value для нормального распределения
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_statistic)))

    # Вывод результатов
    print(f"Количество увеличений: {N_positive}")
    print(f"Количество уменьшений: {N_negative}")
    print(f"Статистика Z: {z_statistic}")
    print(f"p-value: {p_value}")

    # Интерпретация
    if p_value < 0.05:
        print("Ряд имеет значимый тренд.")
    else:
        print("Тренд не обнаружен.")


FosterStewart(smoothed_rates)


plt.figure(figsize=(12, 6))
plt.subplot(4, 1, 1)
plt.plot(rates, label="Оригинальный временной ряд", color="black")
plt.title("Оригинальный временной ряд")
plt.subplot(4, 1, 2)
plt.plot(cleaned_rates, label="Очищенный временной ряд")
plt.title("Очищенный временной ряд")
plt.subplot(4, 1, 3)
plt.plot(filtered_series, label="Фильтрованный временной ряд", color="orange")
plt.title("Фильтрованный временной ряд")
plt.subplot(4, 1, 4)
plt.plot(smoothed_rates, label="Сглаженный временной ряд", color="orange")
plt.title("Сглаженный временной ряд")
plt.tight_layout()
plt.show()
