import pandas as pd

train = pd.read_csv('playground-series-s6e5/train.csv')
test = pd.read_csv('playground-series-s6e5/test.csv')
sub = pd.read_csv('playground-series-s6e5/sample_submission.csv')

print("--- Train Info ---")
print(train.info())
print("\n--- Missing Values in Train ---")
print(train.isnull().sum())
print("\n--- Train Target Distribution ---")
print(train['PitNextLap'].value_counts(normalize=True))

print("\n--- Test Info ---")
print(test.info())
print("\n--- Missing Values in Test ---")
print(test.isnull().sum())

print("\n--- Sample Submission ---")
print(sub.head())
