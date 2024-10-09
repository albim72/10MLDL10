import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Generowanie danych
np.random.seed(0)  # Ustawienie ziarna dla powtarzalności
months = pd.date_range('2023-01-01', periods=12, freq='ME').month_name()  # Nazwy miesięcy
categories = ['Żywność', 'Mieszkanie', 'Transport', 'Rozrywka', 'Inne']

# Generowanie losowych danych wydatków w przedziale 200-2000 PLN
expenses_data = {category: np.random.randint(200, 2001, size=len(months)) for category in categories}

# 2. Tworzenie DataFrame
expenses_df = pd.DataFrame(expenses_data, index=months)
expenses_df.index.name = 'Miesiąc'

# 3. Analiza danych
total_expenses = expenses_df.sum()  # Całkowite wydatki na każdą kategorię
average_expenses = expenses_df.mean()  # Średnie wydatki miesięczne na każdą kategorię
highest_month = expenses_df.sum(axis=1).idxmax()  # Miesiąc z najwyższymi wydatkami ogółem

# 4. Wizualizacja danych
plt.figure(figsize=(10, 6))
expenses_df.plot(kind='bar', stacked=True, color=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
plt.title('Wydatki domowe w 2023 roku')
plt.xlabel('Miesiąc')
plt.ylabel('Wydatki (PLN)')
plt.xticks(rotation=45)
plt.legend(title='Kategorie')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Wyświetlenie wyników analizy
print("Całkowite wydatki na każdą kategorię w 2023 roku:")
print(total_expenses)
print("\nŚrednie wydatki miesięczne na każdą kategorię:")
print(average_expenses)
print(f"\nMiesiąc z najwyższymi wydatkami: {highest_month}")
