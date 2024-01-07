import streamlit as st
import yfinance as yf
import pandas as pd

from tensorflow.keras.models import load_model
import joblib
import numpy as np
import time

st.markdown(
    """
    <style>.rtl {direction: rtl;font-family: 'Roboto', sans-serif;font-size: 50px;font-weight: bold;}</style>
    <div class="rtl">الذكاء الاصطناعي في توقع سعر الاسهم</div>
    """,
    unsafe_allow_html=1,
)
st.title("Machine Learning Stock Prediction")


col1, col2 = st.columns(2)
t = col1.selectbox("Select Ticker:", ('TSLA', 'AAPL', 'MSFT', 'GOOGL', 'BABA', 'AMZN', 'NVDA', 'ADBE', 'ABNB', 'AMD'))

start="2010-6-29"
end = pd.to_datetime('today').strftime('%Y-%m-%d')

@st.cache_resource(ttl=60*60)
def load_data(t):
    data = yf.download(t, start, end)
    data.reset_index(inplace=True)
    return data
data = load_data(t)

# Display stock prices chart
st.write(f"### {t} Stock prices till today")
st.line_chart(data['Close'])


# Display first and last few rows of raw data
st.write("First & Last rows of raw data:")
col1, col2 = st.columns(2)

col1.dataframe(data.head().reset_index(drop=True))
col2.dataframe(data.tail().reset_index(drop=True))


# Load pre-trained model and scaler & Check if pre-trained files exist if not ask to upload them
try:
    model = load_model("saved_model.h5")
    scaler = joblib.load('scaler.pkl')
except:
    model = st.file_uploader("Upload the model")
    scaler = st.file_uploader("Upload the scaler")

st.markdown(
    """
    <style>.rtl {direction: rtl;font-family: 'Roboto', sans-serif;font-size: 50px;font-weight: bold;}</style>
    <div class="rtl">تنويه:</div>
    <h5>هذا المشروع ليس نصيحة مالية وانما لاغراض تعليمية فقط، لا تستخدم هذا المشروع لاتخاذ اي قرارات مالية </h5>
    """,
    unsafe_allow_html=1,)
st.write("This is not financial advice this is for educational purposes only, Do not use this to make any financial decisions\nYou are responsible for your own money")


# Prepare the data
new_df = data[['Date', 'Close']]
if len(new_df['Close']) < 60:
    st.write("Not enough data to make a prediction.")

last_60_days = new_df['Close'][-60:].values  # Use the last 60 trading days up to the day before the current day

last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))
X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

pred_price = model.predict(X_test)

pred_price = scaler.inverse_transform(pred_price)
# Make a prediction
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
st.write(f"Predicted closing price for the next trading day: {pred_price[0][0]}")


st.write(f"## Last day Price ({new_df['Date'].iloc[-1].date()}):")
if pred_price[0][0] > new_df['Close'].iloc[-1]:
    st.write(f"## {new_df['Close'].iloc[-1]:.2f} $ 📈")
else:
    st.write(f"## {new_df['Close'].iloc[-1]:.2f} $ 📉")
c1, c2 = st.columns(2)
c1.write("### Predicted price :")
c1.write(f"## {pred_price[0][0]:.2f} $")
c2.image("training.gif",use_column_width=True)

