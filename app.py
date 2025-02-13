import streamlit as st
from breast_data import get_data
from sklearn.neural_network import MLPClassifier

st.title("Breast Cancer Prediction App by Femi Adeyemo")

df = get_data()
X = df.drop(columns=['target'])
y = df['target']

model = MLPClassifier(hidden_layer_sizes=(50, 50), activation='relu', solver='adam', max_iter=500)
model.fit(X, y)

st.sidebar.header("Input Features")
user_input = [st.sidebar.slider(col, float(X[col].min()), float(X[col].max()), float(X[col].mean())) for col in X.columns]

if st.sidebar.button("Predict"):
    prediction = model.predict([user_input])[0]
    result = "Malignant" if prediction == 0 else "Benign"
    st.write(f"Prediction: {result}")
