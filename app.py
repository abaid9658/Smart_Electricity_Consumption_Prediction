"""
app.py
------
Streamlit dashboard for the Smart Electricity Consumption Prediction &
Energy Optimization System.

Run with: streamlit run app.py

Pages: Home | Prediction | EDA | Analytics | Recommendations | Reports | About
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from predict import predict_consumption

st.set_page_config(page_title="Smart Electricity Predictor", page_icon="⚡", layout="wide")

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "electricity_features.csv")
EDA_DIR = os.path.join(BASE_DIR, "reports", "eda")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()

# ---------------- Sidebar Navigation ----------------
st.sidebar.title("⚡ Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Prediction", "EDA", "Analytics", "Recommendations", "Reports", "About Project"],
)

# =====================================================================
# HOME
# =====================================================================
if page == "Home":
    st.title("⚡ Smart Electricity Consumption Prediction & Energy Optimization")
    st.markdown(
        "A production-style Machine Learning system that predicts household daily "
        "electricity consumption and provides personalized energy-saving recommendations."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset Records", f"{len(df):,}")
    col2.metric("Features", f"{df.shape[1]-2}")
    col3.metric("Avg Daily Consumption", f"{df['Daily_Electricity_Consumption_kWh'].mean():.2f} kWh")
    col4.metric("Best Model R2", "0.9726")

    st.markdown("### How it works")
    st.markdown(
        "1. Go to **Prediction** and enter your household's appliance usage\n"
        "2. Get your predicted daily/monthly consumption and estimated bill\n"
        "3. View personalized recommendations to reduce energy usage\n"
        "4. Explore **EDA** and **Analytics** to understand consumption patterns"
    )

# =====================================================================
# PREDICTION
# =====================================================================
elif page == "Prediction":
    st.title("🔮 Predict Your Electricity Consumption")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Household")
            house_type = st.selectbox("House Type", ["Apartment", "Independent House", "Villa"])
            area_sqft = st.number_input("Area (sqft)", 300, 6000, 1200)
            num_rooms = st.number_input("Number of Rooms", 1, 12, 3)
            family_members = st.number_input("Family Members", 1, 12, 4)
            adults = st.number_input("Adults", 1, 8, 2)
            children = st.number_input("Children", 0, 6, 1)
            seniors = st.number_input("Senior Citizens", 0, 4, 0)
            work_from_home = st.checkbox("Work From Home")

        with c2:
            st.subheader("Appliance Usage (hrs/day)")
            ac_hours = st.slider("AC Usage Hours", 0.0, 16.0, 4.0)
            fan_hours = st.slider("Fan Usage Hours", 0.0, 20.0, 6.0)
            tv_hours = st.slider("TV Hours", 0.0, 12.0, 3.0)
            laptop_hours = st.slider("Laptop Hours", 0.0, 16.0, 4.0)
            lighting_hours = st.slider("Lighting Hours per Room", 0.0, 14.0, 6.0)
            water_motor_hours = st.slider("Water Motor Hours", 0.0, 4.0, 1.0)
            microwave_hours = st.slider("Microwave Hours", 0.0, 1.5, 0.3)
            geyser_hours = st.slider("Geyser Hours", 0.0, 3.0, 0.5)
            washing_machine = st.checkbox("Washing Machine Used Today")

        with c3:
            st.subheader("Environment & Extras")
            outdoor_temp = st.slider("Outdoor Temperature (°C)", 5.0, 48.0, 28.0)
            humidity = st.slider("Humidity (%)", 10.0, 95.0, 55.0)
            season = st.selectbox("Season", ["Summer", "Winter", "Spring", "Autumn"])
            day_of_week = st.selectbox("Day of Week",
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            is_holiday = st.checkbox("Is Holiday")
            peak_hour_usage = st.checkbox("Uses electricity during 6-10 PM peak", value=True)
            ev_charging = st.checkbox("EV Charging Today")
            solar_panel = st.checkbox("Has Solar Panel")
            battery_backup = st.checkbox("Has Battery Backup")
            electricity_rate = st.number_input("Electricity Rate (per kWh)", 5.0, 60.0, 22.0)

        submitted = st.form_submit_button("Predict Consumption", use_container_width=True)

    if submitted:
        is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0
        month_map = {"Summer": 5, "Winter": 12, "Spring": 3, "Autumn": 10}

        raw_input = {
            "House_Type": house_type, "Area_sqft": area_sqft, "Number_of_Rooms": num_rooms,
            "Family_Members": family_members, "Adults": adults, "Children": children,
            "Senior_Citizens": seniors, "Work_From_Home": int(work_from_home),
            "AC_Usage_Hours": ac_hours, "Fan_Usage_Hours": fan_hours, "TV_Hours": tv_hours,
            "Laptop_Hours": laptop_hours, "Lighting_Hours_Per_Room": lighting_hours,
            "Washing_Machine_Usage": int(washing_machine), "Water_Motor_Hours": water_motor_hours,
            "Microwave_Usage_Hours": microwave_hours, "Geyser_Usage_Hours": geyser_hours,
            "EV_Charging": int(ev_charging), "Solar_Panel": int(solar_panel),
            "Battery_Backup": int(battery_backup), "Outdoor_Temperature": outdoor_temp,
            "Humidity": humidity, "Season": season, "Month": month_map.get(season, 6),
            "Day_of_Week": day_of_week, "Is_Weekend": is_weekend, "Is_Holiday": int(is_holiday),
            "Peak_Hour_Usage": int(peak_hour_usage), "Electricity_Rate": electricity_rate,
        }

        result = predict_consumption(raw_input)
        summary = result["summary"]

        st.success(f"Predicted Daily Consumption: **{result['predicted_kwh']} kWh**")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Daily Consumption", f"{summary['predicted_daily_kwh']} kWh")
        m2.metric("Monthly Consumption", f"{summary['predicted_monthly_kwh']} kWh")
        m3.metric("Monthly Bill", f"{summary['predicted_monthly_bill']:.0f}")
        m4.metric("Efficiency Score", f"{summary['energy_efficiency_score']}/100")

        if summary["peak_usage_alert"]:
            st.warning("⚠️ Peak Hour Alert: Significant usage detected during 6-10 PM window.")

        st.markdown("### 💡 Personalized Recommendations")
        for r in result["recommendations"]:
            st.info(f"**{r['category']}**: {r['message']}")

        st.markdown("### 💰 Savings Potential")
        s1, s2, s3 = st.columns(3)
        s1.metric("Potential Monthly Savings", f"{summary['potential_monthly_savings_kwh']} kWh")
        s2.metric("Potential Monthly Savings (Cost)", f"{summary['potential_monthly_savings_cost']:.0f}")
        s3.metric("Yearly CO2 Reduction", f"{summary['carbon_reduction_kg_per_year']} kg")

        result_df = pd.DataFrame([summary])
        st.download_button("Download Prediction (CSV)", result_df.to_csv(index=False),
                            "prediction_result.csv", "text/csv")

# =====================================================================
# EDA
# =====================================================================
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")
    eda_images = {
        "Target Distribution": "01_target_distribution.png",
        "Correlation Heatmap": "02_correlation_heatmap.png",
        "Feature Correlation with Target": "03_target_correlation_bar.png",
        "Temperature vs Consumption": "04_temperature_vs_consumption.png",
        "Family Size vs Consumption": "05_family_size_vs_consumption.png",
        "Room Count vs Consumption": "06_rooms_vs_consumption.png",
        "Seasonal Usage": "07_seasonal_usage.png",
        "Weekend & Solar Impact": "08_weekend_solar_impact.png",
    }
    choice = st.selectbox("Select visualization", list(eda_images.keys()))
    img_path = os.path.join(EDA_DIR, eda_images[choice])
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)

# =====================================================================
# ANALYTICS (interactive Plotly)
# =====================================================================
elif page == "Analytics":
    st.title("📈 Interactive Analytics")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="Daily_Electricity_Consumption_kWh", nbins=30,
                            title="Consumption Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.scatter(df, x="Outdoor_Temperature", y="Daily_Electricity_Consumption_kWh",
                           color="AC_Usage_Hours", title="Temperature vs Consumption")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Filter & Explore")
    max_ac = st.slider("Filter by max AC hours", 0.0, float(df.AC_Usage_Hours.max()), float(df.AC_Usage_Hours.max()))
    filtered = df[df.AC_Usage_Hours <= max_ac]
    st.write(f"Showing {len(filtered)} of {len(df)} records")
    st.dataframe(filtered.head(50))

# =====================================================================
# RECOMMENDATIONS (general tips)
# =====================================================================
elif page == "Recommendations":
    st.title("💡 General Energy-Saving Tips")
    st.markdown("""
- **AC**: Set thermostat to 24-26°C; each degree lower increases consumption ~5-8%.
- **Peak Hours**: Avoid running heavy appliances between 6 PM-10 PM.
- **Solar**: Households with solar panels show ~27% lower average net consumption in our data.
- **EV Charging**: Charge overnight during off-peak hours.
- **Lighting**: LED bulbs use ~75% less energy than incandescent.
- **Geyser**: Use a timer; avoid running it more than 1-1.5 hrs/day.
- **Washing Machine**: Always run full loads; avoid peak hours.
    """)
    st.info("Go to the **Prediction** page for recommendations personalized to your household.")

# =====================================================================
# REPORTS
# =====================================================================
elif page == "Reports":
    st.title("📄 Reports")
    st.markdown("Download generated project reports:")

    reports = {
        "Model Comparison (CSV)": os.path.join(BASE_DIR, "reports", "model_comparison.csv"),
        "Feature Importance Ranking (CSV)": os.path.join(BASE_DIR, "reports", "feature_importance_ranking.csv"),
    }
    for name, path in reports.items():
        if os.path.exists(path):
            with open(path, "rb") as f:
                st.download_button(f"Download {name}", f, os.path.basename(path))

# =====================================================================
# ABOUT
# =====================================================================
elif page == "About Project":
    st.title("ℹ️ About This Project")
    st.markdown("""
**Smart Electricity Consumption Prediction & Energy Optimization System**

Built as a complete, production-style Machine Learning project covering:
- Original synthetic dataset generation (physics-informed, 2000 records, 30+ features)
- Data preprocessing (missing values, outliers, encoding, scaling)
- Exploratory Data Analysis with business insights
- Feature engineering (Power Usage Index, Occupancy Score, etc.)
- Model comparison across 7 algorithms (best: Gradient Boosting, R2 = 0.97)
- Hyperparameter tuning via RandomizedSearchCV
- Rule-based energy optimization recommendation engine
- This interactive Streamlit dashboard

See the project README for full documentation.
    """)
