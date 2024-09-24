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


def rates_request():
    """Запрос курса узбекских сум с сайта ЦБ"""
    res = requests.get(
        "https://www.cbr.ru/scripts/XML_dynamic.asp?date_req1=24/09/2023&date_req2=24/09/2024&VAL_NM_RQ=R01717"
    )
    root = ET.fromstring(res.content)
    rates = []
    for item in root.findall("Record"):
        rate = float(item.find("VunitRate").text.replace(",", "."))
        rates.append(rate)
    return rates


def ADF_test(data):
    """Тест Дикки-Фуллера, проверяет гипотезу о наличии единичного корня (нестационарности)."""
    rates_series = pd.Series(data)

    # ADF-тест (тест Дикки-Фуллера)
    adf_result = adfuller(rates_series)
    print("ADF-тест:")
    print(f"ADF статистика: {adf_result[0]}")
    print(f"p-value: {adf_result[1]}")
    print(f"Критические значения:")
    for key, value in adf_result[4].items():
        print(f"\t{key}: {value}")


def KPSS_test(data):
    """Тест Квятковского-Филлипса-Шмидта-Шина, проверяет ряд на стационарность"""
    rates_series = pd.Series(data)
    # KPSS-тест
    kpss_result = kpss(rates_series, regression="c", nlags="auto")
    print("\nKPSS-тест:")
    print(f"KPSS статистика: {kpss_result[0]}")
    print(f"p-value: {kpss_result[1]}")
    print(f"Критические значения:")
    for key, value in kpss_result[3].items():
        print(f"\t{key}: {value}")


def IQR_method(rates):
    """Межквартильный размах, удаляет выбросы"""
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
    return cleaned_rates


def Box_Pierce_test(data):
    """Статистика Бокса-Пирса, тестирование на наличие белого шума"""
    # Шаг 1: Рассчитываем автокорреляции для h лагов
    h = 1  # количество лагов, которые мы будем учитывать
    acf_values = sm.tsa.acf(data, nlags=h, fft=False)

    # Шаг 2: Рассчитываем статистику Бокса-Пирса
    n = len(data)
    Q = n * np.sum(
        np.square(acf_values[1 : h + 1])
    )  # Пропускаем нулевой лаг (автокорреляция с собой)

    print(f"Статистика Бокса-Пирса: {Q}")


def Butterworth_filter(data):
    """Фильтр Баттерворта, очистка от белого шума"""
    order = 4  # Порядок фильтра
    cutoff_frequency = 0.1  # Частота среза (в долях частоты Найквиста)

    # Проектирование фильтра Баттерворта
    b, a = butter(order, cutoff_frequency, btype="low", analog=False)

    # Применение фильтра к данным
    filtered_series = filtfilt(b, a, data)

    return filtered_series


def EMA(data, alpha=0.2):
    """Алгоритм экспоненциального скользящего среднего, сглаживает значения ряда"""
    data = np.array(data)
    ema = np.zeros_like(data)
    ema[0] = data[0]  # Начальное значение
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema


def FosterStewart(data):
    """Метод Фостера-Стьюарта, проверяет ряд на наличие тренда"""
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
