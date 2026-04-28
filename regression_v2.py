import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def to_num(s):
    return pd.to_numeric(s.astype(str).str.replace('.','',regex=False).str.replace(',','.',regex=False), errors='coerce')

prices = pd.read_csv('Day-ahead_prices_202001010000_202501012000_Hour.csv', sep=';', encoding='utf-8-sig', low_memory=False)
gen    = pd.read_csv('Actual_generation_202001010000_202501012000_Hour.csv', sep=';', encoding='utf-8-sig', low_memory=False)
cons   = pd.read_csv('Actual_consumption_202001010000_202501012000_Hour.csv', sep=';', encoding='utf-8-sig', low_memory=False)

df = pd.DataFrame({
    'price_DE':  pd.to_numeric(prices['Germany/Luxembourg [€/MWh] Calculated resolutions'], errors='coerce'),
    'wind_off':  to_num(gen['Wind offshore [MWh] Calculated resolutions']),
    'wind_on':   to_num(gen['Wind onshore [MWh] Calculated resolutions']),
    'solar':     to_num(gen['Photovoltaics [MWh] Calculated resolutions']),
    'gas':       to_num(gen['Fossil gas [MWh] Calculated resolutions']),
    'lignite':   to_num(gen['Lignite [MWh] Calculated resolutions']),
    'hard_coal': to_num(gen['Hard coal [MWh] Calculated resolutions']),
    'nuclear':   to_num(gen['Nuclear [MWh] Calculated resolutions']),
    'load':      to_num(cons['grid load [MWh] Calculated resolutions']),
    'dt':        pd.to_datetime(prices['Start date'], dayfirst=False),
})

df = df.set_index('dt').dropna()
df['total_wind'] = df['wind_off'] + df['wind_on']

# Time features
df['hour']    = df.index.hour
df['month']   = df.index.month
df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

# Hour dummies (drop hour 0 as baseline)
hour_dummies  = pd.get_dummies(df["hour"],  prefix="h", drop_first=True).astype(int).astype(int)
month_dummies = pd.get_dummies(df["month"], prefix="m", drop_first=True).astype(int).astype(int)
#(df['hour'],  prefix='h', drop_first=True)
month_dummies = pd.get_dummies(df["month"], prefix="m", drop_first=True).astype(int)

df2 = pd.concat([df[['price_DE','total_wind','solar','gas','lignite','hard_coal','nuclear','load','is_weekend']], 
                 hour_dummies, month_dummies], axis=1)

X = sm.add_constant(df2.drop('price_DE', axis=1))
y = df2['price_DE']
model = sm.OLS(y, X).fit()
print(f"R² = {model.rsquared:.3f}")
print(model.summary())

df2['predicted'] = model.predict(X)
df2['residual']  = model.resid

h1 = df2['2022-01-01':'2022-06-30']
fig, axes = plt.subplots(2,1,figsize=(14,9))
axes[0].plot(h1.index, h1['price_DE'],  label='Actual',    color='steelblue', linewidth=0.8)
axes[0].plot(h1.index, h1['predicted'], label='Predicted', color='tomato',    linewidth=0.8)
axes[0].set_title(f'Actual vs Predicted — H1 2022  (R²={model.rsquared:.3f})')
axes[0].set_ylabel('EUR/MWh')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].hist(df2['residual'], bins=120, color='steelblue', alpha=0.75)
axes[1].axvline(0, color='tomato', linestyle='--')
axes[1].set_title('Residuals')
axes[1].set_xlabel('Residual [EUR/MWh]')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('regression_v2_results.png', dpi=150)
plt.show()
print("Done.")
