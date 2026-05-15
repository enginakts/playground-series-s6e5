import pandas as pd

train = pd.read_csv('playground-series-s6e5/train.csv')
print(train[['Race', 'Year', 'Driver', 'LapNumber', 'PitNextLap', 'LapTime (s)']].head(20))

# Check if data is sequential
print("\nChecking if laps are sequential for a random driver in a race:")
sample = train[(train['Race'] == 'Canadian Grand Prix') & (train['Year'] == 2022) & (train['Driver'] == 'D109')].sort_values('LapNumber')
print(sample[['LapNumber', 'Stint', 'TyreLife', 'PitNextLap', 'LapTime (s)']])
