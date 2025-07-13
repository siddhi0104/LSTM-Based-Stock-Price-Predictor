import numpy as np 
import pandas as pd
import yfinance as yf 
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta


st.set_page_config(page_title="Stock Predictor", layout="centered", page_icon="📈")


st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@500;700&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Montserrat', sans-serif;
            background-color: #0E1117;
            color: white;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        h1 {
            color: #00FFC6;
            font-size: 2.5rem;
            font-weight: 700;
            border-bottom: 2px solid #00FFC6;
            padding-bottom: 10px;
        }

        h2, h3 {
            color: #F1F1F1;
            margin-top: 2rem;
        }

        .stAlert {
            background-color: #1e2b33 !important;
            color: #39FF14 !important;
            font-weight: bold;
        }

        .stDataFrame {
            background-color: #111111;
            border: 1px solid #333333;
        }

        .css-1aumxhk {
            padding-top: 2rem;
        }

        .css-ffhzg2 {
            background-color: #0E1117 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Stock Selection")
stock = st.sidebar.text_input("Enter Stock Symbol (e.g. AAPL, GOOG)", "GOOG")

# Dates
start = '2012-01-01'
end = '2022-12-31'

# Load model
model = load_model('stock_predictions_model.keras')

# Header
st.title("AI-Powered Stock Market Predictor")

# Load data
data = yf.download(stock, start, end)


plt.rcParams.update({
    'axes.facecolor': '#0E1117',
    'figure.facecolor': '#0E1117',
    'axes.edgecolor': '#444444',
    'axes.labelcolor': '#FFFFFF',
    'xtick.color': '#AAAAAA',
    'ytick.color': '#AAAAAA',
    'grid.color': '#333333',
    'text.color': '#FFFFFF',
    'lines.linewidth': 2,
    'grid.linestyle': '--',
    'legend.facecolor': '#0E1117',
    'legend.edgecolor': '#444444',
})

# Moving average plot
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(data.Close, color='#32CD32', label='Close Price')   # Softer green
plt.plot(ma_50_days, color='#FF6EC7', label='MA50')          # Pink
plt.legend()
st.pyplot(fig1)

# Company Info
info = yf.Ticker(stock).info
st.subheader("Company Info")
st.markdown(f"- **Name:** {info.get('longName', 'N/A')}")
st.markdown(f"- **Sector:** {info.get('sector', 'N/A')}")
st.markdown(f"- **Country:** {info.get('country', 'N/A')}")
st.markdown(f"- **Market Cap:** {info.get('marketCap', 'N/A'):,}")

# Show Data
st.subheader("Historical Stock Data")
st.dataframe(data.tail(10), height=200)

# Prepare data
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

scaler = MinMaxScaler(feature_range=(0,1))
pas_100_days = data_train.tail(100)
data_test_full = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test_full)

# Moving Average Plots
def plot_ma(title, series_list, labels, colors):
    fig, ax = plt.subplots(figsize=(10,5))
    for s, l, c in zip(series_list, labels, colors):
        ax.plot(s, label=l, color=c)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    st.pyplot(fig)

ma_50 = data.Close.rolling(50).mean()
ma_100 = data.Close.rolling(100).mean()
ma_200 = data.Close.rolling(200).mean()

st.subheader("Moving Averages")
plot_ma("Price vs MA50", [data.Close, ma_50], ['Close', 'MA50'], ['#00FFAA', '#FF6EC7'])
plot_ma("Price vs MA50 vs MA100", [data.Close, ma_50, ma_100], ['Close', 'MA50', 'MA100'], ['#00FFAA', '#FF6EC7', '#1E90FF'])
plot_ma("Price vs MA100 vs MA200", [data.Close, ma_100, ma_200], ['Close', 'MA100', 'MA200'], ['#00FFAA', '#FFA500', '#9370DB'])

# Prepare model input
x, y = [], []
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])
x, y = np.array(x), np.array(y)

# Predict
predict = model.predict(x)
scale = 1 / scaler.scale_[0]
predict = predict * scale
y = y * scale

# Show Prediction
st.subheader("Model Predictions vs Actual")
fig5, ax5 = plt.subplots(figsize=(10,5))
ax5.plot(y, label="Actual", color="lime")
ax5.plot(predict, label="Predicted", color="#FF4500")
ax5.set_title("Stock Price Prediction")
ax5.set_xlabel("Time")
ax5.set_ylabel("Price")
ax5.legend()
st.pyplot(fig5)

# Next day forecast
st.subheader("Next Day Price Prediction")
future_input = np.array(data_test_scale[-100:]).reshape(1, 100, 1)
future_pred = model.predict(future_input)
future_pred = future_pred * scale
next_day = pd.to_datetime(data.index[-1]) + timedelta(days=1)
st.success(f"Predicted closing price for {stock} on {next_day.date()} is **${future_pred[0][0]:.2f}**")
