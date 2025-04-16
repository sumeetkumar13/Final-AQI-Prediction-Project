import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load model
model = joblib.load("saved_models/random_forest_model.pkl")


# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/processed_data.csv", parse_dates=["datetime"])
    df.set_index("datetime", inplace=True)
    return df

# App Layout
st.set_page_config(page_title="Delhi AQI Prediction", layout="wide")

st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Project Overview", "EDA Visualizations", "Model Performance", "Live Prediction", "Team Info"])

# 1. Project Overview
if menu == "Project Overview":
    st.title("Delhi AQI Prediction 🌫️")
    st.markdown("""
    ### 📌 Goal:
    Predict the Air Quality Index (AQI) for Delhi using weather and pollutant data.

    ### 📁 Dataset:
    - Date Range: 2015 to 2020
    - Features: Temperature, Dew, Wind, Pollutants (PM2.5, PM10, NO₂, etc.)

    ### 🧰 Tools & Libraries:
    - Python, Pandas, Matplotlib, Seaborn, Scikit-learn, Streamlit

    ### 🎯 Target:
    - **AQI** (numerical prediction)

    """)

# 2. EDA Section
elif menu == "EDA Visualizations":
    st.title("Exploratory Data Analysis")
    data = load_data()

    st.subheader("1. AQI Trend Over Time")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(data.index, data["AQI"], label="AQI", color='blue', linewidth=1)
    ax1.set_title("AQI Over Time")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("AQI")
    st.pyplot(fig1)

    st.subheader("2. Average AQI by Month")
    data['Month'] = data.index.month
    monthly_avg = data.groupby("Month")["AQI"].mean()
    fig2, ax2 = plt.subplots()
    ax2.plot(monthly_avg.index, monthly_avg.values, marker='o', color='purple')
    ax2.set_title("Average AQI by Month")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Average AQI")
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    st.pyplot(fig2)

    st.subheader("3. Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    corr = data.drop(columns=["Month"]).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax3)
    ax3.set_title("Correlation Matrix")
    st.pyplot(fig3)

    st.subheader("4. AQI Distribution")
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    sns.histplot(data["AQI"], kde=True, ax=ax4, color="orange", bins=30)
    ax4.set_title("Distribution of AQI")
    st.pyplot(fig4)

    st.subheader("5. Boxplot of AQI by Month")
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    sns.boxplot(x="Month", y="AQI", data=data, ax=ax5, palette="Pastel1")
    ax5.set_title("AQI by Month")
    ax5.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    st.pyplot(fig5)

    st.subheader("6. 7-Day Rolling Average of AQI")
    fig6, ax6 = plt.subplots(figsize=(10, 4))
    data['Rolling_AQI'] = data["AQI"].rolling(window=7).mean()
    ax6.plot(data.index, data['Rolling_AQI'], color="green")
    ax6.set_title("7-Day Rolling Average of AQI")
    ax6.set_xlabel("Date")
    ax6.set_ylabel("AQI")
    st.pyplot(fig6)

    st.subheader("7. Scatter Plots: Top Pollutants vs. AQI")
    top_pollutants = ['PM2.5', 'PM10', 'NO2', 'O3']
    fig7, axs7 = plt.subplots(2, 2, figsize=(12, 10))
    for idx, pollutant in enumerate(top_pollutants):
        r, c = divmod(idx, 2)
        sns.scatterplot(x=data[pollutant], y=data["AQI"], ax=axs7[r][c], alpha=0.5)
        axs7[r][c].set_title(f"{pollutant} vs AQI")
        axs7[r][c].set_xlabel(pollutant)
        axs7[r][c].set_ylabel("AQI")
    plt.tight_layout()
    st.pyplot(fig7)

# 3. Model Performance
elif menu == "Model Performance":
    st.title("Model Performance 📊")

    st.markdown("""
    ### 🔍 Model Metrics

    | Model              | MAE    | RMSE   | R² Score |
    |-------------------|--------|--------|----------|
    | Linear Regression | 29.61  | 40.31  | 0.876    |
    | Decision Tree     | 31.00  | 43.23  | 0.858    |
    | **Random Forest** | **23.49** | **32.73** | **0.918** |
    | Gradient Boosting | 24.90  | 33.80  | 0.913    |
    | XGBoost           | 23.44  | 32.89  | 0.918    |

    ✅ **Random Forest** was selected for final deployment due to strong performance and interpretability.
    """)

# 4. Live Prediction
elif menu == "Live Prediction":
    st.title("Live AQI Prediction 🧪")

    st.markdown("Enter the weather and pollutant parameters below to get the predicted AQI:")

    # Full input set in the correct order
    tempmax = st.number_input("Max Temperature (°C)", value=30.0)
    tempmin = st.number_input("Min Temperature (°C)", value=20.0)
    temp = st.number_input("Temperature (°C)", value=25.0)
    dew = st.number_input("Dew Point (°C)", value=15.0)
    winddir = st.number_input("Wind Direction (°)", value=180)
    sealevelpressure = st.number_input("Sea Level Pressure (hPa)", value=1012)
    cloudcover = st.number_input("Cloud Cover (%)", value=50)
    visibility = st.number_input("Visibility (km)", value=4)

    pm25 = st.number_input("PM2.5 (µg/m³)", value=50.0)
    pm10 = st.number_input("PM10 (µg/m³)", value=70.0)
    no = st.number_input("NO (µg/m³)", value=20.0)
    no2 = st.number_input("NO2 (µg/m³)", value=30.0)
    nox = st.number_input("NOx (µg/m³)", value=50.0)
    nh3 = st.number_input("NH3 (µg/m³)", value=10.0)
    co = st.number_input("CO (mg/m³)", value=1.0)
    so2 = st.number_input("SO2 (µg/m³)", value=10.0)
    o3 = st.number_input("O3 (µg/m³)", value=40.0)
    benzene = st.number_input("Benzene (µg/m³)", value=1.5)
    toluene = st.number_input("Toluene (µg/m³)", value=5.0)

    # Date-based features
    year = st.selectbox("Year", [2015, 2016, 2017, 2018, 2019, 2020], index=5)
    month = st.selectbox("Month", list(range(1, 13)))
    day = st.selectbox("Day", list(range(1, 32)))
    dayofweek = st.selectbox("Day of Week (0=Mon, 6=Sun)", list(range(7)), format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])

    if st.button("Predict AQI"):
        features = np.array([[tempmax, tempmin, temp, dew, winddir, sealevelpressure,
                              cloudcover, visibility, pm25, pm10, no, no2, nox, nh3,
                              co, so2, o3, benzene, toluene,
                              year, month, day, dayofweek]])

        #st.write("Input shape:", features.shape)  # Optional for debugging
        prediction = model.predict(features)[0]
        st.success(f"🟩 Predicted AQI: {round(prediction, 2)}")


# 5. Team Info
elif menu == "Team Info":
    st.title("Team & Acknowledgments 🙌")
    st.markdown("""
    ### 👨‍💻 Team Members:
    - Your Name
    - Collaborator 1
    - Collaborator 2

    ### 🙏 Acknowledgments:
    - College Faculty
    - OpenAQ and Govt. of India Data Portals
    - Scikit-learn & Streamlit communities
    """)
