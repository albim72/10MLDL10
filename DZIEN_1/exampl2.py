import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Generowanie danych
np.random.seed(42)  # Ustawienie ziarna dla powtarzalności
years = np.arange(2000, 2024)  # Lata od 2000 do 2023
cities = ['Warszawa', 'Kraków', 'Gdańsk', 'Wrocław', 'Poznań']

# Generowanie losowych danych temperatur w stopniach Celsjusza
# Średnie wartości: 10, 11, 9, 10.5, 10.2
temperature_data = {city: np.random.normal(loc=10 + np.random.rand() * 2, scale=1, size=len(years)) for city in cities}

# 2. Tworzenie DataFrame
temperature_df = pd.DataFrame(temperature_data, index=years)
temperature_df.index.name = 'Year'

# 3. Analiza danych
average_temperatures = temperature_df.mean()
temperature_trends = temperature_df.rolling(window=3).mean()  # Uśrednienie temperatur na podstawie 3 lat

# 4. Wizualizacja danych
plt.figure(figsize=(14, 7))

# Wykres temperatur
for city in cities:
    plt.plot(temperature_df.index, temperature_df[city], marker='o', linestyle='-', label=f'{city} (roczne)')
    plt.plot(temperature_trends.index, temperature_trends[city], linestyle='--', label=f'{city} (średnia 3-letnia)')

plt.title('Średnie roczne temperatury w miastach (2000-2023)')
plt.xlabel('Rok')
plt.ylabel('Temperatura (°C)')
plt.xticks(years, rotation=45)  # Rotacja etykiet osi X
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Wyświetlenie średnich temperatur
print("Średnie temperatury (2000-2023):")
print(average_temperatures)
