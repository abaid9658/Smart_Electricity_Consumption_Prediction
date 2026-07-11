"""
recommendation_engine.py
--------------------------
Generates personalized energy-saving recommendations based on a household's
predicted consumption and appliance usage pattern. Logic is rule-based
(if-then), grounded in the same appliance power ratings used in data
generation -- so savings estimates are consistent with the rest of the project.
"""

POWER_RATING = {
    "AC": 1.5, "Fan": 0.075, "Geyser": 2.0, "WashingMachine": 0.5,
    "Microwave": 1.2, "WaterMotor": 0.75,
}

CARBON_FACTOR_KG_PER_KWH = 0.71  # approx grid emission factor (kg CO2 per kWh)


def generate_recommendations(user_input: dict, predicted_kwh: float, electricity_rate: float):
    """
    user_input: dict with keys like AC_Usage_Hours, Geyser_Usage_Hours,
                Washing_Machine_Usage, Peak_Hour_Usage, Solar_Panel, EV_Charging
    predicted_kwh: model's predicted daily consumption
    electricity_rate: currency units per kWh
    """
    recs = []
    potential_daily_savings_kwh = 0.0

    ac_hours = user_input.get("AC_Usage_Hours", 0)
    geyser_hours = user_input.get("Geyser_Usage_Hours", 0)
    washing_machine = user_input.get("Washing_Machine_Usage", 0)
    peak_usage = user_input.get("Peak_Hour_Usage", 0)
    solar = user_input.get("Solar_Panel", 0)
    ev = user_input.get("EV_Charging", 0)

    # --- AC optimization ---
    if ac_hours > 6:
        saving = (ac_hours - 6) * POWER_RATING["AC"] * 0.3  # 30% achievable via thermostat/timing
        potential_daily_savings_kwh += saving
        recs.append({
            "category": "AC Optimization",
            "message": f"Your AC usage ({ac_hours:.1f} hrs/day) is high. Setting the thermostat "
                       f"to 24-26°C and using a timer for the last 2 hours of sleep can save "
                       f"approximately {saving:.2f} kWh/day.",
        })

    # --- Peak hour shifting ---
    if peak_usage == 1:
        recs.append({
            "category": "Peak Hour Shifting",
            "message": "You're using significant electricity during peak hours (6 PM-10 PM). "
                       "Shifting high-power appliances (washing machine, geyser, water motor) "
                       "to off-peak hours (before 6 PM or after 10 PM) reduces strain on the grid "
                       "and may lower your bill under time-of-use tariffs.",
        })

    # --- Washing machine timing ---
    if washing_machine == 1:
        recs.append({
            "category": "Appliance Scheduling",
            "message": "Run the washing machine after 9 PM or before 7 AM (off-peak hours) "
                       "and use full loads only to maximize efficiency per cycle.",
        })

    # --- Geyser optimization ---
    if geyser_hours > 1.5:
        saving = (geyser_hours - 1.5) * POWER_RATING["Geyser"] * 0.4
        potential_daily_savings_kwh += saving
        recs.append({
            "category": "Water Heating",
            "message": f"Geyser usage ({geyser_hours:.1f} hrs/day) is above average. Installing a "
                       f"timer or lowering the thermostat by 5-10°C can save ~{saving:.2f} kWh/day.",
        })

    # --- Solar recommendation ---
    if solar == 0 and predicted_kwh > 12:
        recs.append({
            "category": "Solar Investment",
            "message": "Your household's consumption is above the dataset average. Based on similar "
                       "households with solar panels, you could reduce net grid consumption by "
                       "roughly 20-30% with a rooftop solar installation.",
        })

    # --- EV charging ---
    if ev == 1:
        recs.append({
            "category": "EV Charging",
            "message": "Charge your EV overnight (11 PM-6 AM) during off-peak hours instead of "
                       "evening hours -- this is typically the single largest reducible load in "
                       "your usage profile.",
        })

    # --- Lighting ---
    recs.append({
        "category": "Lighting Efficiency",
        "message": "Switching remaining incandescent/CFL bulbs to LED can cut lighting energy "
                    "use by up to 75% with no change in usage habits.",
    })

    # --- Savings & efficiency summary ---
    monthly_savings_kwh = potential_daily_savings_kwh * 30
    monthly_savings_cost = monthly_savings_kwh * electricity_rate
    yearly_savings_kwh = potential_daily_savings_kwh * 365
    carbon_reduction_kg = potential_daily_savings_kwh * 365 * CARBON_FACTOR_KG_PER_KWH

    # Efficiency score: lower consumption per "occupancy unit" = better.
    # Normalized against a rough dataset baseline (~11 kWh average).
    efficiency_score = max(0, min(100, 100 - ((predicted_kwh - 11) / 11) * 50))

    summary = {
        "predicted_daily_kwh": round(predicted_kwh, 2),
        "predicted_monthly_kwh": round(predicted_kwh * 30, 2),
        "predicted_monthly_bill": round(predicted_kwh * 30 * electricity_rate, 2),
        "potential_daily_savings_kwh": round(potential_daily_savings_kwh, 2),
        "potential_monthly_savings_kwh": round(monthly_savings_kwh, 2),
        "potential_monthly_savings_cost": round(monthly_savings_cost, 2),
        "potential_yearly_savings_kwh": round(yearly_savings_kwh, 2),
        "carbon_reduction_kg_per_year": round(carbon_reduction_kg, 2),
        "energy_efficiency_score": round(efficiency_score, 1),
        "peak_usage_alert": bool(peak_usage == 1),
    }

    return summary, recs


if __name__ == "__main__":
    # Quick manual test
    sample_input = {
        "AC_Usage_Hours": 9, "Geyser_Usage_Hours": 2.2, "Washing_Machine_Usage": 1,
        "Peak_Hour_Usage": 1, "Solar_Panel": 0, "EV_Charging": 1,
    }
    summary, recs = generate_recommendations(sample_input, predicted_kwh=18.5, electricity_rate=22.0)
    print("Summary:", summary)
    print("\nRecommendations:")
    for r in recs:
        print(f"- [{r['category']}] {r['message']}")
