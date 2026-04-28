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

# Load generation without decimal parameter
generation = pd.read_csv('Actual_generation_202001010000_202501012000_Hour.csv',
                         sep=';')
generation['timestamp'] = pd.to_datetime(generation['Start date'], dayfirst=False)
generation = generation.set_index('timestamp')

generation = generation.rename(columns={
    'Wind offshore [MWh] Calculated resolutions': 'wind_offshore',
    'Wind onshore [MWh] Calculated resolutions': 'wind_onshore',
    'Photovoltaics [MWh] Calculated resolutions': 'solar',
    'Fossil gas [MWh] Calculated resolutions': 'gas',
    'Lignite [MWh] Calculated resolutions': 'lignite',
    'Hard coal [MWh] Calculated resolutions': 'hard_coal',
    'Nuclear [MWh] Calculated resolutions': 'nuclear',
})

# Convert each column handling both comma and dot decimals
for col in ['wind_offshore','wind_onshore','solar','gas','lignite','hard_coal','nuclear']:
    generation[col] = generation[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    generation[col] = pd.to_numeric(generation[col], errors='coerce')

generation['total_wind'] = generation['wind_offshore'] + generation['wind_onshore']

# Merge
df = prices[['price_DE']].join(
    generation[['total_wind','solar','gas','lignite','hard_coal','nuclear']], how='inner')

print(f"Dataset shape: {df.shape}")
print(df[['price_DE','gas','lignite','total_wind','solar']].describe())

# Daily averages
daily = df.resample('D').mean()

# PLOT
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

axes[0].plot(daily.index, daily['price_DE'], color='steelblue', linewidth=0.8)
axes[0].set_ylabel('Price [EUR/MWh]')
axes[0].set_title('German Electricity Market 2020-2024: The Crisis Story')
axes[0].axvspan('2022-02-24', '2023-06-01', alpha=0.15, color='red', label='Russia-Ukraine war period')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(daily.index, daily['gas'], color='orange', linewidth=0.8, label='Gas')
axes[1].plot(daily.index, daily['lignite'], color='brown', linewidth=0.8, label='Lignite')
axes[1].plot(daily.index, daily['nuclear'], color='green', linewidth=0.8, label='Nuclear')
axes[1].plot(daily.index, daily['hard_coal'], color='gray', linewidth=0.8, label='Hard coal')
axes[1].set_ylabel('Generation [MWh]')
axes[1].set_title('Fossil Fuel and Nuclear Generation')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(daily.index, daily['total_wind'], color='steelblue', linewidth=0.8, label='Wind')
axes[2].plot(daily.index, daily['solar'], color='gold', linewidth=0.8, label='Solar')
axes[2].set_ylabel('Generation [MWh]')
axes[2].set_title('Renewable Generation')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('crisis_story.png', dpi=150)
plt.show()
print("Chart saved as crisis_story.png")
