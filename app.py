import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, time
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Household Energy Consumption Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom CSS for Premium Google Stitch / Dark Energy Aesthetics
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Global Styles & Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Background & Container Customization */
    .stApp {
        background: linear-gradient(135deg, #0B0F17 0%, #0F172A 50%, #0B132B 100%);
        color: #F8FAFC;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }

    /* Top Banner Header */
    .hero-header {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(6, 182, 212, 0.08) 100%);
        border: 1px solid rgba(16, 185, 129, 0.25);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #10B981 0%, #34D399 50%, #38BDF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 6px 0;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin: 0;
        font-weight: 400;
    }

    /* Custom Cards */
    .stitch-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .stitch-card:hover {
        border-color: rgba(16, 185, 129, 0.4);
    }

    /* Prediction Result Central Card */
    .prediction-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 2px solid rgba(16, 185, 129, 0.4);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 0 40px rgba(16, 185, 129, 0.15);
        margin-bottom: 24px;
    }

    .pred-value {
        font-size: 3.8rem;
        font-weight: 800;
        color: #34D399;
        margin: 10px 0;
        line-height: 1.1;
        text-shadow: 0 0 25px rgba(52, 211, 153, 0.3);
    }

    .pred-unit {
        font-size: 1.3rem;
        font-weight: 600;
        color: #94A3B8;
    }

    /* Status Badges */
    .badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    
    .badge-low {
        background: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.5);
    }
    
    .badge-medium {
        background: rgba(245, 158, 11, 0.2);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.5);
    }
    
    .badge-high {
        background: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.5);
    }

    /* KPI Mini Cards */
    .kpi-container {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .kpi-title {
        font-size: 0.82rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .kpi-val {
        font-size: 1.4rem;
        font-weight: 700;
        color: #F8FAFC;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0B1120 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.8);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 8px;
        color: #94A3B8;
        font-weight: 600;
        padding: 0 18px;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(16, 185, 129, 0.2) !important;
        color: #34D399 !important;
        border: 1px solid rgba(16, 185, 129, 0.4) !important;
    }
    
    /* Tooltip Helper Text */
    .help-text {
        font-size: 0.82rem;
        color: #64748B;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Load XGBoost Model & Features (Cached for Performance)
# ---------------------------------------------------------
MODEL_PATH = os.path.join("models", "revised_xgboost_model.pkl")
FEATURE_PATH = os.path.join("models", "feature_list.pkl")

# Exact required feature order expected by XGBoost model
REQUIRED_FEATURES = [
    'Global_reactive_power', 'Voltage', 'Global_intensity',
    'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
    'hour', 'day_of_week', 'month', 'day', 'is_weekend',
    'hour_sin', 'hour_cos'
]

@st.cache_resource
def load_model_and_features():
    """Load model and feature list with validation."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: `{MODEL_PATH}`")
        return None, None
    if not os.path.exists(FEATURE_PATH):
        st.error(f"❌ Feature list file not found at: `{FEATURE_PATH}`")
        return None, None

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    
    with open(FEATURE_PATH, "rb") as f:
        feature_list = pickle.load(f)
        
    return model, feature_list

model, loaded_features = load_model_and_features()

# Ensure model loaded successfully
if model is None or loaded_features is None:
    st.stop()

# Validate loaded features match exact required list
if loaded_features != REQUIRED_FEATURES:
    st.warning(f"⚠️ Note: `feature_list.pkl` contents deviate slightly. Enforcing standard 13-feature array structure.")


# ---------------------------------------------------------
# Sidebar Input Controls & Scenarios
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ Control Panel")
    st.markdown("Adjust parameters or select a scenario preset to predict household energy consumption.")

    # Scenario Presets
    preset = st.selectbox(
        "🎯 Preset Scenarios",
        options=["Custom Inputs", "🌙 Night Off-Peak (Low Load)", "☀️ Standard Daytime", "⚡ Peak Evening Usage", "🌱 Eco-Friendly Weekend"],
        index=0
    )

    # Preset configurations dictionary
    presets = {
        "🌙 Night Off-Peak (Low Load)": {
            "reactive": 0.08, "voltage": 242.0, "intensity": 1.4,
            "sub1": 0.0, "sub2": 0.0, "sub3": 1.0,
            "time": time(2, 30), "date": datetime(2026, 8, 18)
        },
        "☀️ Standard Daytime": {
            "reactive": 0.12, "voltage": 239.5, "intensity": 5.2,
            "sub1": 0.0, "sub2": 1.0, "sub3": 12.0,
            "time": time(14, 0), "date": datetime(2026, 8, 18)
        },
        "⚡ Peak Evening Usage": {
            "reactive": 0.35, "voltage": 235.0, "intensity": 15.8,
            "sub1": 15.0, "sub2": 20.0, "sub3": 18.0,
            "time": time(20, 15), "date": datetime(2026, 8, 18)
        },
        "🌱 Eco-Friendly Weekend": {
            "reactive": 0.05, "voltage": 241.0, "intensity": 3.0,
            "sub1": 0.0, "sub2": 0.0, "sub3": 0.0,
            "time": time(11, 30), "date": datetime(2026, 8, 23)  # Sunday
        }
    }

    # Set default values based on selected preset
    if preset in presets:
        p = presets[preset]
        def_reactive = p["reactive"]
        def_voltage = p["voltage"]
        def_intensity = p["intensity"]
        def_sub1 = p["sub1"]
        def_sub2 = p["sub2"]
        def_sub3 = p["sub3"]
        def_time = p["time"]
        def_date = p["date"]
    else:
        def_reactive = 0.10
        def_voltage = 240.0
        def_intensity = 4.5
        def_sub1 = 0.0
        def_sub2 = 1.0
        def_sub3 = 17.0
        def_time = time(19, 0)
        def_date = datetime.now()

    st.markdown("---")

    # 1. Date & Time Inputs
    st.markdown("#### 📅 Date & Time Parameters")
    input_date = st.date_input("Select Date", value=def_date)
    input_time = st.time_input("Select Time", value=def_time)
    
    # Automatic feature calculations from date/time
    selected_hour = input_time.hour
    day_of_week = input_date.weekday() # 0 = Monday, 6 = Sunday
    month = input_date.month
    day = input_date.day
    is_weekend = 1 if day_of_week in [5, 6] else 0

    # Cyclic hour transformation formulas
    hour_sin = float(np.sin(2 * np.pi * selected_hour / 24.0))
    hour_cos = float(np.cos(2 * np.pi * selected_hour / 24.0))

    st.caption(f"ℹ️ Auto-computed: Hour `{selected_hour}:00`, Day `{day}`, Month `{month}`, Weekend `{is_weekend}`")

    st.markdown("---")

    # 2. Electrical Measurements
    st.markdown("#### 🔌 Electrical Grid Metrics")
    voltage = st.slider("Voltage (Volts)", min_value=220.0, max_value=255.0, value=float(def_voltage), step=0.1, help="RMS AC voltage level of the household grid.")
    global_intensity = st.slider("Global Intensity (Amperes)", min_value=0.2, max_value=30.0, value=float(def_intensity), step=0.1, help="Total current intensity in Amps.")
    global_reactive = st.slider("Global Reactive Power (kW)", min_value=0.0, max_value=1.5, value=float(def_reactive), step=0.01, help="Unusable reactive power (VAR) in kW.")

    st.markdown("---")

    # 3. Sub-Metering Consumption Inputs
    st.markdown("#### 🧺 Sub-Metering Appliance Loads")
    st.caption("Active energy consumption (Watt-hours per minute)")
    sub_1 = st.number_input("Sub-Metering 1: Kitchen", min_value=0.0, max_value=50.0, value=float(def_sub1), step=1.0, help="Kitchen appliances (Dishwasher, Microwave, Oven).")
    sub_2 = st.number_input("Sub-Metering 2: Laundry", min_value=0.0, max_value=50.0, value=float(def_sub2), step=1.0, help="Laundry room (Washing Machine, Tumble Dryer, Refrigerator).")
    sub_3 = st.number_input("Sub-Metering 3: Climate/Water", min_value=0.0, max_value=50.0, value=float(def_sub3), step=1.0, help="Electric Water Heater & Air Conditioner.")

    st.markdown("---")
    st.markdown("💡 *Google Stitch Design System - Household Energy Predictor*")


# ---------------------------------------------------------
# Build Feature DataFrame for Prediction
# ---------------------------------------------------------
input_dict = {
    'Global_reactive_power': global_reactive,
    'Voltage': voltage,
    'Global_intensity': global_intensity,
    'Sub_metering_1': sub_1,
    'Sub_metering_2': sub_2,
    'Sub_metering_3': sub_3,
    'hour': selected_hour,
    'day_of_week': day_of_week,
    'month': month,
    'day': day,
    'is_weekend': is_weekend,
    'hour_sin': hour_sin,
    'hour_cos': hour_cos
}

# Construct DataFrame with EXACT 13 feature column ordering required by model
input_df = pd.DataFrame([input_dict])[REQUIRED_FEATURES]

# Predict Global Active Power in kW
predicted_power = float(model.predict(input_df)[0])
# Clip prediction to realistic non-negative floor
predicted_power = max(0.0, predicted_power)


# ---------------------------------------------------------
# Main Dashboard UI Layout
# ---------------------------------------------------------

# Top Hero Header Banner
st.markdown("""
<div class="hero-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h1 class="hero-title">⚡ Household Energy Predictor</h1>
            <p class="hero-subtitle">XGBoost-powered Machine Learning Forecast for Household Global Active Power Consumption</p>
        </div>
        <div style="margin-top: 10px;">
            <span class="badge badge-low">● Live ML Engine Active</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# Center Hero Prediction Card & Key Metrics
col_hero, col_metrics = st.columns([1.2, 1])

with col_hero:
    # Categorize consumption level
    if predicted_power < 1.2:
        status_label = "Low Energy Usage"
        badge_class = "badge-low"
        status_desc = "Optimal eco-friendly operating range."
    elif predicted_power < 3.0:
        status_label = "Moderate Energy Usage"
        badge_class = "badge-medium"
        status_desc = "Standard active household load."
    else:
        status_label = "High Demand Peak"
        badge_class = "badge-high"
        status_desc = "Heavy appliance activation detected."

    # Estimated hourly cost assuming average $0.16 per kWh rate
    est_cost_per_hr = predicted_power * 0.16
    est_daily = predicted_power * 24.0

    st.markdown(f"""
    <div class="prediction-card">
        <div style="font-size: 0.95rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.08em;">
            PREDICTED GLOBAL ACTIVE POWER
        </div>
        <div class="pred-value">{predicted_power:.3f} <span class="pred-unit">kW</span></div>
        <div style="margin-bottom: 16px;">
            <span class="badge {badge_class}">{status_label}</span>
        </div>
        <p style="color: #CBD5E1; font-size: 0.9rem; margin-bottom: 0;">
            {status_desc}
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_metrics:
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Est. Hourly Cost</div>
            <div class="kpi-val">${est_cost_per_hr:.3f} <span style="font-size:0.8rem; color:#64748B;">/hr</span></div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Est. 24h Output</div>
            <div class="kpi-val">{est_daily:.1f} <span style="font-size:0.8rem; color:#64748B;">kWh</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    m3, m4 = st.columns(2)
    with m3:
        sub_total_wh = sub_1 + sub_2 + sub_3
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Sub-metered Load</div>
            <div class="kpi-val">{sub_total_wh:.0f} <span style="font-size:0.8rem; color:#64748B;">Wh/min</span></div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        # Apparent power estimate S = V * I / 1000 kVA
        apparent_pwr = (voltage * global_intensity) / 1000.0
        pf = (predicted_power / apparent_pwr) if apparent_pwr > 0 else 0.0
        pf = min(1.0, pf)
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Power Factor (Est.)</div>
            <div class="kpi-val">{pf:.2f}</div>
        </div>
        """, unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------
# Interactive Analytics Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Sub-Metering Breakdown",
    "📈 24-Hour Projected Profile",
    "⚙️ Feature Engineering Inspector",
    "💡 Efficiency Recommendations"
])


# --- TAB 1: Sub-Metering Distribution ---
with tab1:
    st.markdown("### 📊 Sub-Metering & Appliance Load Distribution")
    st.caption("Detailed breakdown of active power consumption across sub-metered zones vs unmetered baseline circuits.")
    
    col_chart1, col_chart2 = st.columns([1, 1])
    
    # Active sub-metering values in Wh/min converted to equivalent kW (approx Wh/min * 60 / 1000)
    sub1_kw = (sub_1 * 60) / 1000.0
    sub2_kw = (sub_2 * 60) / 1000.0
    sub3_kw = (sub_3 * 60) / 1000.0
    sub_sum_kw = sub1_kw + sub2_kw + sub3_kw
    other_kw = max(0.0, predicted_power - sub_sum_kw)

    with col_chart1:
        # Donut Chart for Sub-metering proportion
        labels = ['Kitchen (Sub 1)', 'Laundry (Sub 2)', 'Climate/Water (Sub 3)', 'Other/Unmetered']
        values = [sub1_kw, sub2_kw, sub3_kw, other_kw]
        colors = ['#10B981', '#3B82F6', '#06B6D4', '#64748B']

        fig_pie = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.55,
            marker_colors=colors,
            textinfo='percent+label',
            hoverinfo='label+value+percent',
            hovertemplate="<b>%{label}</b><br>Power: %{value:.3f} kW<br>Share: %{percent}<extra></extra>"
        )])
        
        fig_pie.update_layout(
            title_text="Appliance Category Share (kW)",
            title_font_color="#F8FAFC",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#94A3B8"),
            showlegend=False,
            margin=dict(t=40, b=20, l=20, r=20),
            height=320
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_chart2:
        # Bar Chart comparison
        fig_bar = go.Figure(data=[
            go.Bar(
                x=['Kitchen', 'Laundry', 'Climate/Water', 'Unmetered'],
                y=[sub1_kw, sub2_kw, sub3_kw, other_kw],
                marker_color=colors,
                text=[f"{v:.3f} kW" for v in [sub1_kw, sub2_kw, sub3_kw, other_kw]],
                textposition='auto'
            )
        ])
        
        fig_bar.update_layout(
            title_text="Active Load Breakdown (kW)",
            title_font_color="#F8FAFC",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#94A3B8"),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="kW"),
            margin=dict(t=40, b=20, l=20, r=20),
            height=320
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# --- TAB 2: 24-Hour Projected Profile ---
with tab2:
    st.markdown("### 📈 24-Hour Daily Demand Forecast Profile")
    st.caption("Simulated 24-hour demand curve based on selected voltage, current intensity, and cyclic trigonometric hour features.")

    # Generate 24-hour curve keeping other parameters constant
    hours_range = list(range(24))
    curve_predictions = []

    for h in hours_range:
        h_sin = float(np.sin(2 * np.pi * h / 24.0))
        h_cos = float(np.cos(2 * np.pi * h / 24.0))
        
        row_dict = input_dict.copy()
        row_dict['hour'] = h
        row_dict['hour_sin'] = h_sin
        row_dict['hour_cos'] = h_cos
        
        row_df = pd.DataFrame([row_dict])[REQUIRED_FEATURES]
        pred_h = max(0.0, float(model.predict(row_df)[0]))
        curve_predictions.append(pred_h)

    # Plot 24h curve with Plotly
    fig_curve = go.Figure()

    fig_curve.add_trace(go.Scatter(
        x=hours_range,
        y=curve_predictions,
        mode='lines+markers',
        name='Predicted Demand',
        line=dict(color='#10B981', width=3, shape='spline'),
        marker=dict(size=6, color='#34D399'),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.1)'
    ))

    # Highlight current selected hour
    fig_curve.add_trace(go.Scatter(
        x=[selected_hour],
        y=[predicted_power],
        mode='markers',
        name=f'Selected Hour ({selected_hour}:00)',
        marker=dict(color='#F43F5E', size=14, symbol='diamond')
    ))

    fig_curve.update_layout(
        title="24-Hour Energy Consumption Simulation (kW)",
        title_font_color="#F8FAFC",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#94A3B8"),
        xaxis=dict(
            title="Hour of Day (0-23)",
            dtick=1,
            gridcolor='rgba(255,255,255,0.05)'
        ),
        yaxis=dict(
            title="Global Active Power (kW)",
            gridcolor='rgba(255,255,255,0.05)'
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=40, l=40, r=40),
        height=380
    )

    st.plotly_chart(fig_curve, use_container_width=True)


# --- TAB 3: Feature Engineering Inspector ---
with tab3:
    st.markdown("### ⚙️ Feature Engineering Inspector")
    st.markdown("Inspect the exact 13 features formatted into the XGBoost model pipeline in strict positional order.")

    features_info_df = pd.DataFrame([
        {"Feature Name": col, "Value": input_df[col].values[0], "Data Type": str(input_df[col].dtype), "Description": desc}
        for col, desc in [
            ('Global_reactive_power', 'Unusable reactive power (VAR) in kW'),
            ('Voltage', 'Grid AC RMS Voltage level'),
            ('Global_intensity', 'Current intensity in Amperes'),
            ('Sub_metering_1', 'Kitchen load (dishwasher, microwave) in Wh/min'),
            ('Sub_metering_2', 'Laundry load (washing machine, fridge) in Wh/min'),
            ('Sub_metering_3', 'Climate/Water heater load in Wh/min'),
            ('hour', 'Hour of the day (0 to 23)'),
            ('day_of_week', 'Day of week (0=Mon, 6=Sun)'),
            ('month', 'Month of year (1 to 12)'),
            ('day', 'Day of month (1 to 31)'),
            ('is_weekend', 'Binary flag: 1 for Sat/Sun, 0 for Mon-Fri'),
            ('hour_sin', 'Cyclic sine encoding: sin(2*pi*hour/24)'),
            ('hour_cos', 'Cyclic cosine encoding: cos(2*pi*hour/24)')
        ]
    ])

    st.dataframe(
        features_info_df,
        column_config={
            "Feature Name": st.column_config.TextColumn("Feature Name", width="medium"),
            "Value": st.column_config.NumberColumn("Input Value", format="%.4f"),
            "Data Type": st.column_config.TextColumn("Type", width="small"),
            "Description": st.column_config.TextColumn("Description", width="large")
        },
        hide_index=True,
        use_container_width=True
    )


# --- TAB 4: Smart Efficiency Recommendations ---
with tab4:
    st.markdown("### 💡 AI Energy Efficiency Insights")
    
    col_rec1, col_rec2 = st.columns(2)
    
    with col_rec1:
        st.markdown("#### 🎯 Load Optimization Tips")
        if is_weekend:
            st.info("🗓️ **Weekend Profile**: Off-peak tariffs may apply depending on your provider. Consider running high-power dishwashing cycles during early morning hours.")
        else:
            st.info("💼 **Weekday Work Routine**: Energy consumption typically spikes during evening hours (18:00 - 22:00). Shift washing machine cycles to non-peak windows.")
            
        if sub_3 > 20:
            st.warning("🔥 **High Climate/Water Heater Load**: Sub-metering 3 is drawing heavy power (>20 Wh/min). Setting thermostats 1°C closer to ambient temperature saves up to 7% energy.")
        else:
            st.success("✅ **Climate Control Balanced**: Water heating & cooling loads are operating efficiently.")

    with col_rec2:
        st.markdown("#### ⚡ Electrical Grid Health")
        if voltage < 230:
            st.warning("⚠️ **Low Line Voltage**: Grid voltage is below 230V. Low voltage causes inductive motors (refrigerators, AC compressors) to draw higher current.")
        elif voltage > 245:
            st.warning("⚠️ **High Line Voltage**: Voltage exceeds 245V. Check for solar feed-in surges or transformer tap adjustments.")
        else:
            st.success("✅ **Stable Grid Voltage**: Voltage is optimal within standard operating tolerances (230V - 245V).")

        if global_reactive > 0.3:
            st.warning("📉 **Power Factor Alert**: High reactive power detected (>0.3 kW). Inductive loads may be creating lagging phase angle. Ensure capacitors/filters are healthy.")
        else:
            st.success("✅ **Good Power Factor**: Minimal reactive power dissipation.")

# Footer Branding
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748B; font-size: 0.85rem; padding: 10px;'>"
    "Household Energy Consumption Predictor • Powered by XGBoost & Streamlit • Built with Google Stitch Design Aesthetics"
    "</div>",
    unsafe_allow_html=True
)
