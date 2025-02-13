'''import data sets'''

from sklearn.datasets import load_breast_cancer
import pandas as pd

#load the dataset

def get_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

if __name__ == "__main__":
    df = get_data()
    print(df.head())  # Display the first few rows
