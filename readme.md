# 🌫️ Delhi AQI Prediction using Machine Learning

This project focuses on predicting the **Air Quality Index (AQI)** of **Delhi** using historical weather and pollutant data. We built a machine learning model that can estimate AQI based on environmental conditions and deployed it as a live interactive web application using **Streamlit**.

---

## 📌 Project Overview

- **Objective**: Predict AQI values to help monitor and manage air pollution.
- **Location**: Delhi, India
- **Dataset Range**: 2015 – 2020
- **Target Variable**: AQI (numerical prediction)

---

## 🧰 Tools & Technologies

- **Language**: Python 3.x
- **Libraries**: pandas, numpy, scikit-learn, seaborn, matplotlib, streamlit, joblib
- **Frontend**: Streamlit (for interactive app)
- **Model Deployment**: Random Forest Regressor (best performer)

---

## 📁 Project Structure

Final AQI/
│
├── 🗂️ app/
│   └── 📄 app.py                  # Streamlit web application
│
├── 🗂️ data/
│   ├── 📁 raw/
│   │   └── 📄 delhi_aqi.csv       # Original raw dataset
│   └── 📁 processed/
│       └── 📄 processed_data.csv  # Cleaned dataset used for modeling
│
├── 🗂️ notebook/
│   └── 📄 01_data_exploration.ipynb   # EDA and model development notebook
│
├── 🗂️ saved_models/
│   └── 📄 random_forest_model.pkl    # Trained Random Forest model
│
├── 📄 requirements.txt           # Required libraries
└── 📄 README.md                  # Project documentation


---

## 📊 Exploratory Data Analysis (EDA)

- AQI trends over time
- Monthly and seasonal variations
- Correlation heatmap of pollutants
- AQI distribution and boxplots by month
- Rolling averages for trend smoothing

---

## 🤖 Model Evaluation

| Model              | MAE    | RMSE   | R² Score |
|-------------------|--------|--------|----------|
| Linear Regression | 29.61  | 40.31  | 0.876    |
| Decision Tree     | 31.00  | 43.23  | 0.858    |
| **Random Forest** | **23.49** | **32.73** | **0.918** |
| Gradient Boosting | 24.90  | 33.80  | 0.913    |
| XGBoost           | 23.44  | 32.89  | 0.918    |

✅ **Random Forest** was chosen as the final model due to its high accuracy and stability.

---

## ⚙️ Live Prediction Tool

Users can input real-time values for:
- Weather parameters (temperature, dew, wind, etc.)
- Pollutant levels (PM2.5, PM10, NO₂, CO, etc.)
- Date-based features (day, month, weekday)

And receive a predicted AQI instantly.

---

## 🚀 How to Run the Project

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/delhi-aqi-prediction.git
   cd delhi-aqi-prediction

2. Install dependencies:
   pip install -r requirements.txt

3. Run the app:
   streamlit run app/app.py

## 🙌 Acknowledgments

OpenAQ Platform for pollution data

Indian Meteorological Department (IMD)

Central Pollution Control Board (CPCB)

Scikit-learn, Streamlit, and open-source Python ecosystem

## 👨‍💻 Team
[Your Name]

[Collaborator 1]

[Collaborator 2]

## 📬 Contact

Let me know if you'd like this personalized more (with your name, links, or screenshot of the app). You’re ready to push this to GitHub now!


