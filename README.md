# 🐧 Penguin Species Classifier

A high-precision Machine Learning web application that predicts penguin species (**Adelie, Chinstrap, or Gentoo**) based on physical measurements. This project utilizes **PCA** to handle high multicollinearity and **Logistic Regression** for a robust, 99% accurate classification.

## 🚀 Live Demo
(https://penguin-classifier-app-model.streamlit.app/)

## 🛠️ Technical Highlights
- **Exploratory Data Analysis (EDA):** Identified extreme multicollinearity (VIF > 280) between flipper length and body mass.
- **Dimensionality Reduction:** Applied **PCA (Principal Component Analysis)** to reduce features from 6 to 4 while retaining 95.5% of variance.
- **Handling Imbalance:** Used **SMOTE** to balance the training set, ensuring perfect recall for the minority species (Chinstrap).
- **Performance:** Achieved **99.7% CV Accuracy** and **99% Test Accuracy**.

## 📂 Project Structure
```text
├── data/
│   └── penguins_size.csv           # Original dataset
├── EDA_Model/
│   └── penguin.py                  # Full research, EDA, and Model training code
├── streamlit_folder/
│   ├── app.py                      # Streamlit application script
│   ├── penguin_classification_v1.pkl # Pickled Model, Scaler, and PCA assets
│   └── requirements.txt            # Dependency list for deployment
└── .gitignore                      # Files excluded from GitHub
