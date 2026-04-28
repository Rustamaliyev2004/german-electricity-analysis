import pandas as pd
import matplotlib.pyplot as plt

# Load prices
prices = pd.read_csv('Day-ahead_prices_202001010000_202501012000_Hour.csv',
                     sep=';', decimal=',')
prices = prices.rename(columns={
    'Germany/Luxembourg [€/MWh] Calculated resolutions': 'price_DE',
    'Start date': 'timestamp'
})
prices['timestamp'] = pd.to_datetime(prices['timestamp'], dayfirst=False)
prices = prices.set_index('timestamp')
prices['price_DE'] = pd.to_numeric(prices['price_DE'], errors='coerce')

# Load generation
generation = pd.read_csv('Actual_generation_202001010000_202501012000_Hour.csv',
                         sep=';', decimal=',')
generation['timestamp'] = pd.to_datetime(generation['Start date'], dayfirst=False)
generation = generation.set_index('timestamp')

# Rename key columns
generation = generation.rename(columns={
    'Wind offshore [MWh] Calculated resolutions': 'wind_offshore',
    'Wind onshore [MWh] Calculated resolutions': 'wind_onshore',
    'Photovoltaics [MWh] Calculated resolutions': 'solar',
    'Fossil gas [MWh] Calculated resolutions': 'gas',
    'Lignite [MWh] Calculated resolutions': 'lignite',
    'Hard coal [MWh] Calculated resolutions': 'hard_coal',
    'Nuclear [MWh] Calculated resolutions': 'nuclear',
})

# Convert to numeric
for col in ['wind_offshore','wind_onshore','solar','gas','lignite','hard_coal','nuclear']:
    generation[col] = pd.to_numeric(generation[col], errors='coerce')

# Create combined renewable column
generation['total_wind'] = generation['wind_offshore'] + generation['wind_onshore']
generation['total_renewable'] = generation['total_wind'] + generation['solar']

# Merge prices and generation
df = prices[['price_DE']].join(generation[['total_wind','solar','total_renewable','gas','lignite','hard_coal','nuclear']], how='inner')

print(f"Combined dataset shape: {df.shape}")
print(df.head())
print()

# PLOT 1 - Renewables vs Price scatter
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(df['total_wind'], df['price_DE'], alpha=0.1, s=1, color='steelblue')
axes[0].set_xlabel('Total Wind Generation [MWh]')
axes[0].set_ylabel('Price [EUR/MWh]')
axes[0].set_title('Wind Generation vs Electricity Price')
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=0.8)

axes[1].scatter(df['solar'], df['price_DE'], alpha=0.1, s=1, color='orange')
axes[1].set_xlabel('Solar Generation [MWh]')
axes[1].set_ylabel('Price [EUR/MWh]')
axes[1].set_title('Solar Generation vs Electricity Price')
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=0.8)

plt.tight_layout()
plt.savefig('renewables_vs_price.png', dpi=150)
plt.show()
print("Chart saved as renewables_vs_price.png")
