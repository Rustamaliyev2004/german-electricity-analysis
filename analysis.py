import pandas as pd
import matplotlib.pyplot as plt

prices = pd.read_csv('Day-ahead_prices_202001010000_202501012000_Hour.csv',
                     sep=';', decimal=',')

prices = prices.rename(columns={
    'Germany/Luxembourg [€/MWh] Calculated resolutions': 'price_DE',
    'Start date': 'timestamp'
})

prices['timestamp'] = pd.to_datetime(prices['timestamp'], dayfirst=False)
prices = prices.set_index('timestamp')

# Convert price to numeric, force errors to NaN
prices['price_DE'] = pd.to_numeric(prices['price_DE'], errors='coerce')

print("Basic statistics for German electricity prices 2020-2024:")
print(prices['price_DE'].describe())
print()
print(f"Maximum price: {prices['price_DE'].max():.2f} EUR/MWh")
print(f"When: {prices['price_DE'].idxmax()}")
print()
print(f"Minimum price: {prices['price_DE'].min():.2f} EUR/MWh")
print(f"When: {prices['price_DE'].idxmin()}")

plt.figure(figsize=(14, 6))
prices['price_DE'].plot(color='steelblue', linewidth=0.5)
plt.title('German Day-Ahead Electricity Prices 2020-2024', fontsize=14)
plt.ylabel('Price [EUR/MWh]')
plt.xlabel('Date')
plt.axhline(y=0, color='red', linestyle='--', linewidth=0.8, label='Zero price')
plt.legend()
plt.tight_layout()
plt.savefig('price_history.png', dpi=150)
plt.show()
print("Chart saved as price_history.png")
