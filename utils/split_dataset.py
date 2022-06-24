import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('../data/go_emotions_dataset.csv')
df_train, df_test = train_test_split(df, test_size=0.1, shuffle=True, random_state=1247)

df_train.to_csv('../data/go_emotions_train.csv')
df_test.to_csv('../data/go_emotions_test.csv')