import pandas as pd

generation = pd.read_csv('Actual_generation_202001010000_202501012000_Hour.csv',
                         sep=';', decimal=',')

print("All columns in generation file:")
for i, col in enumerate(generation.columns):
    print(f"{i}: '{col}'")

print()
print("First row of data:")
print(generation.iloc[0])
