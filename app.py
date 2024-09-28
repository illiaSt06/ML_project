from __future__ import annotations
import streamlit as st
import joblib
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt


def load_and_predict(X: ArrayLike, filename: str = "linear_regression_model.joblib") -> ArrayLike:
    model = joblib.load(filename)
    y = model.predict(X)
    return y[0]


def create_streamlit_app():
    st.title("Simple Regression Model Prediction")
    input_feature = st.slider("Input Feature for Prediction", -3.0, 3.0, 0.0)

    if st.button("Predict Value"):
        input_matrix = np.array([[input_feature]])
        prediction = load_and_predict(input_matrix)
        st.write(f"Prediction: {prediction:.2f}")
        visualize_difference(input_feature, prediction)


def visualize_difference(input_feature: float, prediction: float):
    X = joblib.load("X.joblib")
    y = joblib.load("y.joblib")

    closest_idx = _index_of_closest(X, input_feature)
    actual_target = y[closest_idx]
    difference = actual_target - prediction

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X, y, color='grey', label='Dataset', alpha=0.5)
    ax.scatter(input_feature, actual_target, color='blue', label='Actual Target', s=100)
    ax.scatter(input_feature, prediction, color='red', label='Predicted Target', s=100)

    ax.plot([input_feature, input_feature], [actual_target, prediction], 'k--')
    ax.annotate(f'Difference: {difference:.2f}', xy=(input_feature, (actual_target + prediction) / 2),
                xytext=(10, 10), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', lw=1.5))

    ax.set_title('Actual vs Predicted Target Value')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Target')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


def _index_of_closest(X: ArrayLike, k: float) -> int:
    X = np.asarray(X)
    idx = (np.abs(X - k)).argmin()
    return idx


if __name__ == '__main__':
    create_streamlit_app()
