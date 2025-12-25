# Rainfall Prediction System

This project implements a machine learning-based system to predict the occurrence of rainfall based on various atmospheric conditions. It is an end-to-end data science project, covering data preprocessing, model training, and deployment using a **Streamlit** web application.

---

## Overview

Accurate rainfall prediction is crucial for agriculture, disaster management, and daily planning. The goal of this project is to leverage historical weather data to predict whether it will rain on the following day. The prediction is based on features such as:

- **Pressure** (Atmospheric pressure in hPa)
- **Temperature** (Min, Max, and Current in °C)
- **Dew Point** (°C)
- **Humidity** (%)
- **Cloud Coverage** (%)
- **Sunshine Duration** (Hours)
- **Wind** (Direction and Speed)

The project follows a complete Data Science lifecycle and provides a **modern web interface** for users to get instant predictions.

---

##  Methodology & Workflow

This project was executed in the following detailed steps:

### 1. Data Cleaning & Preprocessing
The raw dataset required processing to be usable for modeling:
- **Missing Values:** Handled null values to ensure data consistency.
- **Encoding:** Categorical variables were encoded using Label Encoding.
- **Scaling:** Numerical features were standardized using `StandardScaler` to ensure all features contribute equally to the model.

### 2. Exploratory Data Analysis (EDA)
- **Distribution Analysis:** Analyzed the distribution of weather parameters to understand their spread and central tendencies.
- **Feature Relationships:** Explored correlations between humidity, cloud coverage, and rainfall occurrence.

### 3. Model Training
Various classification algorithms were trained to find the best performing model. The focus was on robust prediction capability given the input features.

---

##  Model Selection & Results

The **Gradient Boosting Classifier** was selected as the final model due to its balanced performance across all metrics. Below is the comparative analysis of various models trained:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest (Optimized)** | **0.811594** | **0.821256** | **0.811594** | **0.814836** |
| **Gradient Boosting (Optimized)** | **0.811594** | **0.806337** | **0.811594** | **0.807202** |
| Random Forest | 0.797101 | 0.811383 | 0.797101 | 0.801512 |
| Gradient Boosting | 0.797101 | 0.803403 | 0.797101 | 0.799554 |
| XGBoost | 0.797101 | 0.797101 | 0.797101 | 0.797101 |
| Logistic Regression | 0.768116 | 0.804498 | 0.768116 | 0.776126 |
| XGBoost (Optimized) | 0.768116 | 0.762654 | 0.768116 | 0.764685 |

---

## Project Structure

```text
rainfall_prediction/
│
├── app.py                         # Streamlit application
├── models/
│   ├── best_rainfall_model.pkl    # Trained Model
│   ├── scaler.pkl                 # Feature Scaler
│   ├── label_encoder.pkl          # Target Encoder
│   └── feature_names.pkl          # Feature Name List
│
├── notebooks/
│   └── Rainfall_Prediction.ipynb  # EDA, preprocessing, training & evaluation
│
├── dataset/
│   └── Rainfall.csv               # Dataset
│
├── .gitignore                     # Files & folders ignored by Git
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```
---

## Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/kindo-tk/rainfall_prediction.git
   cd rainfall_prediction
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment:**

   - On Windows:
     ```bash
     .\.venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Streamlit application:**

   ```bash
   streamlit run app.py
   ```
---

## Usage

1. **Launch the App:** Open the streamlit link in your browser.
2. **Input Parameters:** Adjust the sliders for various weather conditions (Pressure, Temperature, Humidity, etc.) in the sidebar.
3. **Get Prediction:** Click the **Predict Rainfall** button.
4. **View Result:** The app will display whether rainfall is expected along with the confidence score.

---

## Technologies Used

- **Python** (Programming Language)
- **Streamlit** (Web Framework)
- **Scikit-learn** (Machine Learning)
- **Pandas & NumPy** (Data Manipulation)
- **Joblib** (Model Serialization)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

For any inquiries or feedback, please contact:

- [Tufan Kundu (LinkedIn)](https://www.linkedin.com/in/tufan-kundu-577945221/)
- **Email**: tufan.kundu11@gmail.com
