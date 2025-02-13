from sklearn.feature_selection import SelectKBest, f_classif
from breast_data import get_data

#load data

df = get_data()
X = df.drop(columns=['target'])
y = df['target']

# Select the best 10 features

selector = SelectKBest(score_func=f_classif, k=10)
X_new = selector.fit_transform(X, y)

# Get selected feature names

selected_features = X.columns[selector.get_support()]
print("Selected Features:", selected_features)
