import requests
import xml.etree.ElementTree as ET
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt


res = requests.get(
    "https://www.cbr.ru/scripts/XML_dynamic.asp?date_req1=01/01/2019&date_req2=31/01/2023&VAL_NM_RQ=R01717"
)

root = ET.fromstring(res.content)


rates = []

for item in root.findall("Record"):
    rate = float(item.find("VunitRate").text.replace(",", "."))
    rates.append(rate)

print(len(rates))


# Вычисление квантилей и IQR
rates = np.array(rates)
Q1 = np.percentile(rates, 25)
Q3 = np.percentile(rates, 75)
IQR = Q3 - Q1

Q1, Q3, IQR


# Определение границ
lower_bound = Q1 - 2.5 * IQR
upper_bound = Q3 + 2.5 * IQR


cleaned_rates = np.copy(rates)

# Выявление индексов выбросов
outliers = np.where((rates < lower_bound) | (rates > upper_bound))

# Замена выбросов (например, на среднее между границами)
replacement_value = (lower_bound + upper_bound) / 2

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
print(f"Очищенные данные: {cleaned_rates}")


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

order = 4  # Порядок фильтра
cutoff_frequency = 0.1  # Частота среза (в долях частоты Найквиста)

# Проектирование фильтра Баттерворта
b, a = butter(order, cutoff_frequency, btype="low", analog=False)

# Применение фильтра к данным
filtered_series = filtfilt(b, a, cleaned_rates)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(cleaned_rates, label="Оригинальный временной ряд")
plt.title("Оригинальный временной ряд")
plt.subplot(2, 1, 2)
plt.plot(filtered_series, label="Фильтрованный временной ряд", color="orange")
plt.title("Фильтрованный временной ряд")
plt.tight_layout()
plt.show()
