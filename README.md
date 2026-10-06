# ⚡ Household Energy Predictor

A machine learning application that predicts Global Active Power consumption (kW) from household electricity measurements and time-based features, powered by a compact XGBoost regression model and an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-orange.svg)

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Input Features & Engineering](#input-features--engineering)
- [Machine Learning Workflow](#machine-learning-workflow)
- [Models Explored & Comparison](#models-explored--comparison)
- [Compact Model Selection](#compact-model-selection)
- [Validation & Testing](#validation--testing)
- [Streamlit Application](#streamlit-application)
- [Project Structure](#project-structure)
- [Model Artifacts](#model-artifacts)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Project Status](#project-status)
- [Future Improvements](#future-improvements)
- [Author](#author)

## 🔍 Project Overview
The **Household Energy Predictor** estimates household energy demand using machine learning. It leverages a trained XGBoost regression model deployed via an interactive, responsive Streamlit web application. Users can configure simulated electrical metrics, appliance loads, and time-of-day parameters to see real-time power predictions, cost estimates, and efficiency recommendations.

## 🎯 Problem Statement
Accurately forecasting energy consumption allows for smarter grid management, optimal appliance scheduling, and overall household energy efficiency. The challenge lies in accurately modeling non-linear energy spikes caused by human behavior, cyclical time patterns, and complex interactions between sub-metered appliances and total grid load.

## 💡 Objectives
- Develop an accurate regression model to predict Global Active Power.
- Engineer time-series features to capture cyclical and behavioral energy usage patterns.
- Compare multiple ML algorithms to balance predictive accuracy with model size.
- Deliver a lightweight, fast, and user-friendly interface for scenario testing and prediction.

## 📊 Dataset
The project utilizes the **Individual Household Electric Power Consumption** dataset.
- **Original Dataset Size**: Approximately 2,075,259 observations.
- **Key Variables**: Date, Time, Global Active Power, Global Reactive Power, Voltage, Global Intensity, Sub Metering 1, Sub Metering 2, Sub Metering 3.
- **Target Variable**: Global Active Power (measured in kW).

> **Note**: To maintain a lightweight application repository, the large raw dataset (originally millions of rows) is excluded from version control and is not present in this GitHub repository.

## 🛠 Input Features & Engineering
The Streamlit UI accepts **8 user inputs**: Date, Time, Voltage, Global Intensity, Global Reactive Power, Sub Metering 1, Sub Metering 2, and Sub Metering 3.

Through a dynamic prediction pipeline, these 8 inputs are expanded into the **13 exact features** required by the model:

1. `Global_reactive_power`
2. `Voltage`
3. `Global_intensity`
4. `Sub_metering_1`
5. `Sub_metering_2`
6. `Sub_metering_3`
7. `hour` *(extracted from Time)*
8. `day_of_week` *(extracted from Date)*
9. `month` *(extracted from Date)*
10. `day` *(extracted from Date)*
11. `is_weekend` *(binary flag from Date)*
12. `hour_sin` *(Cyclical time engineering)*
13. `hour_cos` *(Cyclical time engineering)*

**Cyclical Feature Engineering**: Time is cyclical (23:00 is close to 01:00). To help the model understand this, hours are transformed using trigonometric functions:
- `hour_sin = sin(2π × hour / 24)`
- `hour_cos = cos(2π × hour / 24)`

## ⚙️ Machine Learning Workflow
The project journey consisted of:
1. Data Cleaning & Exploratory Data Analysis (EDA)
2. Feature Engineering & Selection
3. Regression Model Experimentation
4. Model Comparison & Tuning
5. Compact Model Selection (XGBoost)
6. Model Serialization
7. End-to-End Validation
8. Streamlit Dashboard Integration & Testing

## 🔬 Models Explored & Comparison
Several regression architectures were evaluated during the experimentation phase:
- **Linear Regression**
- **Random Forest**
- **HistGradientBoosting**
- **XGBoost**

Based on evaluation results, the top two compact performing models were:

| Model | RMSE | R² Score |
|-------|------|----------|
| **XGBoost** (Selected) | 0.02887 | 0.99927 |
| HistGradientBoosting | 0.02981 | 0.99922 |

*Note: XGBoost is used as the selected deployment candidate based on this specific evaluation, practical deployment requirements, and optimal balance of size and speed.*

## 📦 Compact Model Selection
During early development, a larger, highly complex model was created (`final_energy_prediction_model.pkl`), which reached an approximate file size of **2.26 GB**. 

While mathematically viable, a 2.26 GB file is entirely impractical for lightweight local execution, memory-constrained web deployments, and standard GitHub version control. 

Consequently, the heavily optimized **XGBoost model** (~0.95 MB) was selected for the final application. This achieved near-identical predictive performance while remaining compact, fast to load, and perfectly suited for the Streamlit dashboard environment. The large legacy model is excluded from the application repository.

## ✅ Validation & Testing
The saved XGBoost model was thoroughly verified before deployment integration:
- **Feature Contract Validation**: Verified that input formats match training data constraints perfectly.
- **Row-Level Testing**: Tested across multiple real dataset rows, demonstrating an average absolute error of approximately **0.01197 kW**.
- **Fresh Load Verification**: Fresh model reloading exhibited a maximum prediction difference of **0.0**, proving consistent serialization.
- **End-to-End Test Case**: A known baseline test case produced an independently verified prediction of ~1.189593 kW, correctly reflected in the Streamlit UI (rounded to 1.189 kW).

*(Disclaimer: These metrics represent local validation subset results and do not guarantee universal, real-world accuracy across all edge cases.)*

## 💻 Streamlit Application
The frontend application (`app.py`) provides an interactive interface to:
- Test custom energy load parameters.
- View real-time active power predictions in kW.
- Inspect 24-hour demand projection curves.
- View active sub-metering appliance load breakdowns.
- Receive AI-generated efficiency and electrical grid health recommendations.
- Apply preset scenarios (e.g., Night Off-Peak, Standard Daytime, Peak Evening Usage).

## 📂 Project Structure

```text
Household_Energy_Predictor/
├── .streamlit/
│   └── config.toml                  # Streamlit theme/configuration
├── models/
│   ├── feature_engineering_info.pkl # Feature context/metadata
│   ├── feature_list.pkl             # Strict required feature order list
│   └── revised_xgboost_model.pkl    # Serialized compact XGBoost model (~0.95 MB)
├── app.py                           # Main Streamlit web application
├── README.md                        # Project documentation (this file)
├── requirements.txt                 # Python package dependencies
└── .gitignore                       # Ignored files (e.g., virtual environments, large data)
```

## 🧠 Model Artifacts
The application depends on three distinct serialized artifacts located in the `models/` directory:
- **`revised_xgboost_model.pkl`**: The core, trained machine learning regressor.
- **`feature_list.pkl`**: An enforced list of the 13 exact column names ensuring predictions never fail due to feature mismatch.
- **`feature_engineering_info.pkl`**: Metadata regarding the transformation parameters used during the model pipeline construction.

## 🛠 Technologies Used
- **Python 3.9+**
- **XGBoost**: Gradient boosting regression model.
- **Streamlit**: Web application framework.
- **Pandas & NumPy**: Data manipulation and numerical operations.
- **Scikit-Learn**: Validation and metrics.
- **Plotly**: Interactive charting (pie charts, line graphs).

## 🚀 Getting Started

### Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/cloudvaishnavi/Household_Energy_Predictor.git
   cd Household_Energy_Predictor
   ```
2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```
The application will automatically open in your default web browser at `http://localhost:8501`.

## 🔮 Future Improvements
- Expand time-series forecasting capability to predict future hours sequentially.
- Integrate cloud-based ML pipelines for continuous retraining on new data.
- Connect live IoT smart meter data feeds directly into the dashboard.
- Implement user authentication for personalized historical energy tracking.

## 📌 Project Status
**Local Streamlit application tested; public deployment pending.**
The local application runs smoothly, and the repository is established on GitHub. Cloud deployment steps are planned for future iterations.

## 👤 Author
**Vaishnavi K S**  
*Computer Science & Artificial Intelligence Engineering*  
GitHub: [@cloudvaishnavi](https://github.com/cloudvaishnavi)
