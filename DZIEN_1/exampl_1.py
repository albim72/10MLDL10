import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Generowanie danych
np.random.seed(0)  # Ustawienie ziarna dla powtarzalności
dates = pd.date_range('2023-01-01', periods=12, freq='M')  # Daty od stycznia do grudnia
sales = np.random.randint(100, 500, size=len(dates))  # Losowe wartości sprzedaży

# 2. Tworzenie DataFrame
sales_data = pd.DataFrame({
    'Date': dates,
    'Sales': sales
})

# Ustawienie dat jako indeks
sales_data.set_index('Date', inplace=True)

# 3. Wizualizacja danych
plt.figure(figsize=(10, 5))
plt.plot(sales_data.index, sales_data['Sales'], marker='o', linestyle='-', color='b')
plt.title('Sprzedaż produktów w 2023 roku')
plt.xlabel('Miesiąc')
plt.ylabel('Sprzedaż')
plt.xticks(rotation=45)  # Rotacja etykiet osi X
plt.grid()
plt.tight_layout()
plt.show()

# Wyświetlenie danych
print(sales_data)
