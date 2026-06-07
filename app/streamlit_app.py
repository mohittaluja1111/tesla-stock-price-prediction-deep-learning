import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Tesla Stock Price Prediction",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# PROFESSIONAL UI DESIGN
# =========================================================

st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #F5F7FB;
    color: #111827;
}

/* Remove extra top spacing */
.block-container {
    padding-top: 2rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E5E7EB;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: #111827 !important;
}

/* Main Title */
.title-style {
    font-size: 58px;
    font-weight: 800;
    text-align: center;
    color: #111827;
    margin-bottom: 10px;
}

/* Tesla Red */
.tesla-red {
    color: #EF4444;
}

/* Subtitle */
.subtitle-style {
    text-align: center;
    font-size: 24px;
    color: #6B7280;
    margin-bottom: 25px;
    font-weight: 500;
}

/* Metric Cards */
.metric-card {
    background: #FFFFFF;
    padding: 30px;
    border-radius: 20px;
    border: 1px solid #E5E7EB;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.05);
    transition: 0.3s ease;
    text-align: center;
}

.metric-card:hover {
    transform: translateY(-5px);
}

/* Card Titles */
.metric-card h2 {
    color: #374151;
    font-size: 28px;
}

/* Card Numbers */
.metric-card h1 {
    color: #111827;
    font-size: 52px;
    font-weight: 800;
}

/* Overview Box */
.overview-box {
    background: #FFFFFF;
    padding: 30px;
    border-radius: 20px;
    border: 1px solid #E5E7EB;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.05);
}

/* Section Titles */
.section-title {
    color: #111827;
    font-size: 34px;
    font-weight: 700;
    margin-bottom: 20px;
}

/* Paragraph Text */
p {
    color: #4B5563;
    font-size: 18px;
    line-height: 1.8;
}

/* Buttons */
.stButton>button {
    background-color: #EF4444;
    color: white;
    border-radius: 10px;
    border: none;
    height: 3em;
    width: 100%;
    font-size: 18px;
}

.stButton>button:hover {
    background-color: #DC2626;
    color: white;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 15px;
    overflow: hidden;
    border: 1px solid #E5E7EB;
}

/* Footer */
.footer {
    text-align: center;
    color: #6B7280;
    margin-top: 40px;
    font-size: 18px;
    padding-bottom: 20px;
}

/* Divider */
hr {
    border: none;
    height: 1px;
    background: #E5E7EB;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="title-style">
🚘 <span class="tesla-red">Tesla</span> Stock Price Prediction
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="subtitle-style">
Deep Learning Based Stock Forecasting using LSTM & SimpleRNN
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    df = pd.read_csv("../data/TSLA.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

df = load_data()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📌 Navigation")

menu = st.sidebar.radio(
    "Go To",
    [
        "🏠 Dashboard",
        "📂 Dataset",
        "📊 EDA Analysis",
        "🤖 AI Prediction",
        "🔮 Future Forecast"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info("""
### 🚀 Tech Stack

✅ TensorFlow  
✅ LSTM  
✅ Streamlit  
✅ Deep Learning  
✅ Time-Series Forecasting
""")

# =========================================================
# DASHBOARD
# =========================================================

if menu == "🏠 Dashboard":

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
        <h2>📈 Dataset Rows</h2>
        <h1>{df.shape[0]}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
        <h2>📊 Features</h2>
        <h1>{df.shape[1]}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
        <h2>🤖 Model</h2>
        <h1>LSTM</h1>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="overview-box">

    <div class="section-title">
    📌 Project Overview
    </div>

    <p>
    This project predicts Tesla stock prices using Deep Learning models 
    such as SimpleRNN and LSTM.
    </p>

    <p>
    The system analyzes historical stock market data and forecasts future 
    stock prices using Time-Series Forecasting techniques.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("📉 Tesla Stock Price Trend")

    fig = px.line(
        df,
        y='Adj Close',
        title='Tesla Adjusted Closing Price'
    )

    fig.update_layout(
        template='simple_white',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# DATASET
# =========================================================

elif menu == "📂 Dataset":

    st.subheader("📂 Tesla Stock Dataset")

    st.dataframe(df)

    st.markdown("---")

    st.subheader("📈 Statistical Summary")

    st.dataframe(df.describe())

# =========================================================
# EDA ANALYSIS
# =========================================================

elif menu == "📊 EDA Analysis":

    st.subheader("📊 Exploratory Data Analysis")

    # Adj Close Chart
    fig1 = px.line(
        df,
        y='Adj Close',
        title='Tesla Adjusted Close Price'
    )

    fig1.update_layout(
        template='simple_white',
        height=500
    )

    st.plotly_chart(fig1, use_container_width=True)

    # Volume Chart
    fig2 = px.line(
        df,
        y='Volume',
        title='Tesla Trading Volume'
    )

    fig2.update_layout(
        template='simple_white',
        height=500
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Moving Averages
    df['MA50'] = df['Adj Close'].rolling(50).mean()
    df['MA100'] = df['Adj Close'].rolling(100).mean()

    fig3 = go.Figure()

    fig3.add_trace(go.Scatter(
        x=df.index,
        y=df['Adj Close'],
        name='Adj Close'
    ))

    fig3.add_trace(go.Scatter(
        x=df.index,
        y=df['MA50'],
        name='50-Day MA'
    ))

    fig3.add_trace(go.Scatter(
        x=df.index,
        y=df['MA100'],
        name='100-Day MA'
    ))

    fig3.update_layout(
        title='Moving Average Analysis',
        template='simple_white',
        height=500
    )

    st.plotly_chart(fig3, use_container_width=True)

# =========================================================
# AI PREDICTION
# =========================================================

elif menu == "🤖 AI Prediction":

    st.subheader("🤖 Next Day Tesla Stock Prediction")

    data = df[['Adj Close']]

    scaler = MinMaxScaler(feature_range=(0,1))

    scaled_data = scaler.fit_transform(data)

    model = load_model("../models/tesla_lstm_model.keras")

    last_60_days = scaled_data[-60:]

    X_future = np.reshape(last_60_days, (1,60,1))

    future_price = model.predict(X_future)

    future_price = scaler.inverse_transform(future_price)

    predicted_price = future_price[0][0]

    st.markdown(f"""
    <div class="metric-card">
    <h2>📈 Predicted Tesla Price</h2>
    <h1>${predicted_price:.2f}</h1>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.success("Prediction generated successfully using trained LSTM model.")

# =========================================================
# FUTURE FORECAST
# =========================================================

elif menu == "🔮 Future Forecast":

    st.subheader("🔮 10-Day Tesla Stock Forecast")

    data = df[['Adj Close']]

    scaler = MinMaxScaler(feature_range=(0,1))

    scaled_data = scaler.fit_transform(data)

    model = load_model("../models/tesla_lstm_model.keras")

    last_60_days = scaled_data[-60:]

    temp_input = list(last_60_days.flatten())

    future_predictions = []

    for i in range(10):

        x_input = np.array(temp_input[-60:])

        x_input = x_input.reshape(1,60,1)

        pred = model.predict(x_input, verbose=0)

        temp_input.extend(pred[0].tolist())

        future_predictions.append(pred[0][0])

    future_predictions = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1,1)
    )

    forecast_df = pd.DataFrame({
        "Day": [f"Day {i+1}" for i in range(10)],
        "Predicted Price": future_predictions.flatten()
    })

    st.dataframe(forecast_df)

    fig4 = px.line(
        forecast_df,
        x='Day',
        y='Predicted Price',
        markers=True,
        title='10-Day Future Forecast'
    )

    fig4.update_layout(
        template='simple_white',
        height=500
    )

    st.plotly_chart(fig4, use_container_width=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
<div class="footer">

🚀 Developed by <b>Mohit Taluja</b><br><br>

Deep Learning • LSTM • Tesla Forecasting • Streamlit Dashboard

</div>
""", unsafe_allow_html=True)