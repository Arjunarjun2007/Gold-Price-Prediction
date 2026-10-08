import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------------------------------------
# Page Configuration
# ---------------------------------------

st.set_page_config(
    page_title="Gold Price Prediction",
    page_icon="🥇",
    layout="wide"
)

# ---------------------------------------
# Load Dataset and Model
# ---------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("gold_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    return df


@st.cache_resource
def load_model():
    return joblib.load("gold_price_model.pkl")


df = load_data()
model = load_model()

# ---------------------------------------
# Title
# ---------------------------------------

st.title("🥇 Gold Price Prediction System")

st.write(
    "Machine Learning based gold price prediction using "
    "historical market data."
)

st.divider()

# ---------------------------------------
# Dashboard Metrics
# ---------------------------------------

latest_price = df["Close"].iloc[-1]
highest_price = df["Close"].max()
lowest_price = df["Close"].min()
average_price = df["Close"].mean()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Latest Gold Price",
        f"${latest_price:.2f}"
    )

with col2:
    st.metric(
        "Highest Price",
        f"${highest_price:.2f}"
    )

with col3:
    st.metric(
        "Lowest Price",
        f"${lowest_price:.2f}"
    )

with col4:
    st.metric(
        "Average Price",
        f"${average_price:.2f}"
    )

st.divider()

# ---------------------------------------
# Historical Price Chart
# ---------------------------------------

st.subheader("📈 Historical Gold Price")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    df["Date"],
    df["Close"]
)

ax.set_xlabel("Date")
ax.set_ylabel("Gold Price")
ax.set_title("Historical Gold Price")

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

st.divider()

# ---------------------------------------
# Recent Data
# ---------------------------------------

st.subheader("📊 Recent Gold Price Data")

st.dataframe(
    df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    .tail(20),
    use_container_width=True
)

st.divider()

# ---------------------------------------
# Prediction Section
# ---------------------------------------

st.subheader("🤖 Predict Gold Price")

st.write(
    "Enter the latest market values to predict the gold closing price."
)

col1, col2 = st.columns(2)

with col1:

    open_price = st.number_input(
        "Open Price",
        min_value=0.0,
        value=float(df["Open"].iloc[-1])
    )

    high_price = st.number_input(
        "High Price",
        min_value=0.0,
        value=float(df["High"].iloc[-1])
    )

    low_price = st.number_input(
        "Low Price",
        min_value=0.0,
        value=float(df["Low"].iloc[-1])
    )

    volume = st.number_input(
        "Volume",
        min_value=0.0,
        value=float(df["Volume"].iloc[-1])
    )

with col2:

    previous_close = st.number_input(
        "Previous Close",
        min_value=0.0,
        value=float(df["Close"].iloc[-1])
    )

    ma_7 = st.number_input(
        "7-Day Moving Average",
        min_value=0.0,
        value=float(df["Close"].tail(7).mean())
    )

    ma_14 = st.number_input(
        "14-Day Moving Average",
        min_value=0.0,
        value=float(df["Close"].tail(14).mean())
    )

    ma_30 = st.number_input(
        "30-Day Moving Average",
        min_value=0.0,
        value=float(df["Close"].tail(30).mean())
    )

# ---------------------------------------
# Prediction
# ---------------------------------------

if st.button("🔮 Predict Gold Price", use_container_width=True):

    input_data = pd.DataFrame({

        "Open": [open_price],

        "High": [high_price],

        "Low": [low_price],

        "Volume": [volume],

        "Previous_Close": [previous_close],

        "MA_7": [ma_7],

        "MA_14": [ma_14],

        "MA_30": [ma_30]
    })

    prediction = model.predict(input_data)

    predicted_price = prediction[0]

    st.success(
        f"Predicted Gold Price: ${predicted_price:.2f}"
    )

st.divider()

# ---------------------------------------
# Model Performance
# ---------------------------------------

st.subheader("📊 Model Performance")

st.info(
    "The model was evaluated using a chronological "
    "80% training and 20% testing split."
)

st.write("### Random Forest Regression")

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        "MAE",
        "0.20"
    )

with metric2:
    st.metric(
        "RMSE",
        "0.25"
    )

with metric3:
    st.metric(
        "R² Score",
        "0.9971"
    )

st.divider()

# ---------------------------------------
# About
# ---------------------------------------

st.subheader("ℹ️ About the Project")

st.write("""
This project predicts gold closing prices using a
Machine Learning regression model.

The system uses historical gold market information
and engineered features such as previous closing price
and moving averages.

Machine Learning Algorithm:

• Random Forest Regressor

Features used:

• Open Price
• High Price
• Low Price
• Volume
• Previous Close
• 7-Day Moving Average
• 14-Day Moving Average
• 30-Day Moving Average
""")