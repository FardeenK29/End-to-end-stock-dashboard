import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import requests
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def get_stock_data_for_model(ticker_symbol, period="5y"):
    df = yf.download(ticker_symbol, period=period)
    return df[['Close']].dropna()

def preprocess_lstm_data(df):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i, 0])
        y.append(scaled[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_10_days(model, df, scaler):
    last_60 = df[-60:].values
    scaled_input = scaler.transform(last_60).reshape(1, 60, 1)
    predictions = []

    for _ in range(10):
        pred = model.predict(scaled_input, verbose=0)[0][0]
        predictions.append(pred)
        scaled_input = np.append(scaled_input[:,1:,:], [[[pred]]], axis=1)

    preds_rescaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=10)
    return pd.DataFrame(preds_rescaled, index=future_dates, columns=["Predicted Close"])


st.set_page_config(page_title="📊 Stock Dashboard", layout="wide")

# Custom Title
st.markdown(
    """
    <div style="background-color:#0e1117;padding:20px;border-radius:10px">
        <h1 style="color:white;text-align:center;">📈 Real-Time Stock Analytics Dashboard</h1>
        <p style="color:gray;text-align:center;">Live updates | Financial insights | News feed</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar Styling
st.sidebar.markdown("## ⚙️ Filters")
ticker = st.sidebar.text_input('Stock Ticker', 'INFY')
exchange_suffix = st.sidebar.selectbox('Exchange', ['NSE', 'BSE'])
ticker_yf = f"{ticker}.NS" if exchange_suffix == 'NSE' else f"{ticker}.BO"

# Fixed to YTD
today = datetime.now()
start_date = datetime(today.year, 1, 1)
interval = '1d'

# Fetch Stock Data
try:
    stock = yf.Ticker(ticker_yf)
    data = stock.history(start=start_date, end=today, interval=interval)
    info = stock.info
except Exception as e:
    st.error(f"Error fetching data: {e}")
    data = pd.DataFrame()
    info = {}

# Real-Time Ticker Ribbon Replacement
name = info.get('longName', 'N/A')
st.markdown(f"## 📊 {name}")

tv_link = f"https://in.tradingview.com/chart/?symbol=NSE%3A{ticker.upper()}" if exchange_suffix == "NSE" else f"https://in.tradingview.com/chart/?symbol=BSE%3A{ticker.upper()}"
st.markdown(
    f"""
    <div style="background-color:#e1e1e1;padding:8px 16px;border-radius:6px;margin-top:10px;margin-bottom:4px;">
        <a href="{tv_link}" target="_blank" style="text-decoration:none;color:#1a1a1a;font-size:16px;">
            📉 View on TradingView: <strong>{exchange_suffix}:{ticker.upper()}</strong>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

if info.get("regularMarketPrice"):
    price = f"₹{info['regularMarketPrice']:.2f} {info.get('currency', 'INR')}"
    st.markdown(
        f"""
        <div style="text-align:center;margin-bottom:10px;">
            <span style="color:#3c763d;font-size:18px;">Live Price: {price}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# YTD Line Chart and Key Metrics
if not data.empty:
    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        line=dict(color='deepskyblue'),
        name='Close Price'
    ))
    line_fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Close Price (₹)",
        template="plotly_dark",
        showlegend=False
    )
    st.plotly_chart(line_fig, use_container_width=True)

    # Key Metrics below chart
    market_cap = info.get('marketCap', 0) / 1e12
    pe = info.get('forwardPE', info.get('trailingPE', 'N/A'))
    volume = info.get('averageVolume', 0) / 1e6
    eps = info.get('trailingEps', 'N/A')
    roe = info.get('returnOnEquity', 'N/A')
    debt_eq = info.get('debtToEquity', 'N/A')
    div_yield = info.get('dividendYield', 'N/A')

    st.markdown(" ")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 P/E Ratio", f"{pe:.2f}" if isinstance(pe, (float, int)) else "N/A")
    with col2:
        st.metric("📈 ROE", f"{roe*100:.2f}%" if isinstance(roe, (float, int)) else "N/A")
    with col3:
        st.metric("💸 Avg Volume", f"{volume:.2f} M")
    with col4:
        st.metric("🏦 Market Cap", f"{market_cap:.2f} T INR")

st.markdown(" ")


# Tabs Layout
about, pricing_tab, fundamentals, news, financials, forecast, iv_tab, holding_tab = st.tabs(
    ['📘 About', '📊 Pricing', '📋 Fundamentals', '📰 News', '📈 Financials', '📅 10-Day Forecast','🧮 IV Calculator', '📊 Promoter Holding'])

with about:
    st.markdown("### 🏢 Company Overview")
    website = info.get('website', None)
    employees = info.get('fullTimeEmployees', 'N/A')
    sector = info.get('sector', None)
    headquarters = info.get('city', 'N/A') + ', ' + info.get('country', 'N/A')
    description = info.get('longBusinessSummary', 'No description available.')

    col1, col2 = st.columns(2)
    with col1:
        if sector:
            st.write(f"**🏷️ Sector:** {sector}")
        if headquarters:
            st.write(f"**🏭 Headquarters:** {headquarters}")
    with col2:
        if website:
            st.write(f"**🌐 Website:** [{website}]({website})")
        if employees:
            st.write(f"**👥 Employees:** {employees}")

    st.markdown("---")
    st.markdown(f"<div style='text-align: justify;'>{description}</div>", unsafe_allow_html=True)

with pricing_tab:
    st.markdown("### 📊 Historical Pricing Data")
    data2 = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    data2 = data2.sort_index(ascending=True)
    data2['% Change'] = data2['Close'].pct_change()
    data2.dropna(inplace=True)
    data2 = data2.sort_index(ascending=False)

    def highlight_change(row):
        color = 'green' if row['% Change'] > 0 else 'red'
        return [f'color: {color}' if col == '% Change' else '' for col in row.index]

    styled_df = data2.style.apply(highlight_change, axis=1)
    st.dataframe(styled_df.format({'% Change': '{:.2%}'}), use_container_width=True)

    # Pricing Stats
    annual_return = data2['% Change'].mean() * 252
    volatility = np.std(data2['% Change']) * np.sqrt(252)
    sharpe = annual_return / volatility if volatility else None

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        col1.metric("📈 Annual Return", f"{annual_return:.2%}")
        st.caption("Average return the stock has given annually based on recent trends.")

    with col2:
        col2.metric("📉 Volatility", f"{volatility:.2f}")
        st.caption("Measures how much the stock price fluctuates annually.")


    with col3:
        col3.metric("📌 Sharpe Ratio", f"{sharpe:.2f}" if sharpe else "N/A")
        st.caption("Risk-adjusted return (higher is better).")


with fundamentals:
    st.markdown("### 📋 Key Financial Fundamentals")

    if info:
        # Define the list of important financial metrics to show
        important_metrics = [
            "marketCap",
            "trailingPE",
            "forwardPE",
            "priceToBook",
            "enterpriseValue",
            "enterpriseToRevenue",
            "enterpriseToEbitda",
            "pegRatio",
            "returnOnEquity",
            "returnOnAssets",
            "debtToEquity",
            "currentRatio",
            "quickRatio",
            "totalRevenue",
            "grossMargins",
            "operatingMargins",
            "profitMargins",
            "ebitdaMargins",
            "earningsGrowth",
            "revenueGrowth",
            "freeCashflow",
            "operatingCashflow"
        ]

        # Convert info to DataFrame and filter by selected metrics
        df_info = pd.DataFrame(info.items(), columns=["Metric", "Value"])
        df_info = df_info[df_info["Metric"].isin(important_metrics)]
        df_info = df_info[df_info["Value"].notnull()].reset_index(drop=True)

        # Format numbers nicely
        df_info["Value"] = df_info["Value"].apply(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else str(x))

        # Display as clean HTML table
        st.markdown(
            df_info.to_html(index=False, escape=False, justify="left"),
            unsafe_allow_html=True
        )
    else:
        st.warning("Fundamentals data not available.")


with news:
    st.markdown("### 📰 Latest News")
    api_key = "adf161ec70a44feb9227ea97492b40a3"
    query = info.get("longName", ticker)
    response = requests.get(f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&apiKey={api_key}")
    if response.status_code == 200:
        articles = response.json().get('articles', [])[:10]
        for article in articles:
            st.markdown(f"#### [{article['title']}]({article['url']})")
            st.caption(f"🕒 {article['publishedAt']} | 🗞 {article['source']['name']}")
            st.write(article.get('description', ''))
            st.markdown("---")
    else:
        st.warning("Could not fetch news at the moment.")

def quarterly_financials(ticker):
    stock = yf.Ticker(ticker)
    return stock.quarterly_financials.T

def annual_financials(ticker):
    stock = yf.Ticker(ticker)
    return stock.financials.T

with financials:
    st.markdown("### 📈 Income Statements")
    quarterly_data = quarterly_financials(ticker_yf)
    annual_data = annual_financials(ticker_yf)
    selection = st.segmented_control(label="", options=["Quarterly", "Annual"], default="Quarterly")
    st.write("")

    if selection == "Quarterly":
        quarterly_financials = quarterly_data.rename_axis("Quarter").reset_index()
        quarterly_financials['Quarter'] = quarterly_financials['Quarter'].astype(str)
        revenue_chart = alt.Chart(quarterly_financials).mark_bar().encode(
            x=alt.X('Quarter:O', title='Quarter'),
            y=alt.Y('Total Revenue:Q', title='Total Revenue (INR)'),
        )
        net_income_chart = alt.Chart(quarterly_financials).mark_bar(color='#FFFFC5').encode(
            x=alt.X('Quarter:O', title='Quarter'),
            y=alt.Y('Net Income:Q', title='Net Income (INR)'),
        )
        st.altair_chart(revenue_chart, use_container_width=True)
        st.altair_chart(net_income_chart, use_container_width=True)

    if selection == "Annual":
        annual_financials = annual_data.rename_axis("Year").reset_index()
        annual_financials['Year'] = annual_financials['Year'].astype(str).transform(lambda year: year.str.split('-').str[0])
        revenue_chart = alt.Chart(annual_financials).mark_bar().encode(
            x=alt.X('Year:O', title='Year'),
            y=alt.Y('Total Revenue:Q', title='Total Revenue (INR)'),
        )
        net_income_chart = alt.Chart(annual_financials).mark_bar(color='#FFFFC5').encode(
            x=alt.X('Year:O', title='Year'),
            y=alt.Y('Net Income:Q', title='Net Income (INR)'),
        )
        st.altair_chart(revenue_chart, use_container_width=True)
        st.altair_chart(net_income_chart, use_container_width=True)

with forecast:
    st.markdown("### 📅 Next 10 Days Price Forecast")
    try:
        df_model = get_stock_data_for_model(ticker_yf)
        X, y, scaler = preprocess_lstm_data(df_model)
        model = build_lstm_model((X.shape[1], 1))
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)
        forecast_df = predict_10_days(model, df_model, scaler)

        # Plotting
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_model.index[-100:], df_model['Close'].values[-100:], label="Past 100 Days")
        ax.plot(forecast_df.index, forecast_df['Predicted Close'], label="Forecast", color='orange')
        ax.set_title(f"{ticker.upper()} 10-Day Closing Price Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Add this to the end of your Streamlit app

import scipy
from scipy.stats import norm
from scipy.optimize import brentq

# IV Calculator Functions
def black_scholes_call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_volatility_call(S, K, T, r, market_price):
    objective = lambda sigma: black_scholes_call_price(S, K, T, r, sigma) - market_price
    try:
        iv = brentq(objective, 1e-5, 5)
        return iv
    except Exception:
        return None

# Promoter Holding Sample (Static data or API-driven if available)
promoter_data = pd.DataFrame({
    'Date': pd.to_datetime([
        '2021-03-31', '2021-06-30', '2021-09-30',
        '2021-12-31', '2022-03-31', '2022-06-30',
        '2022-09-30', '2022-12-31', '2023-03-31'
    ]),
    'Promoter Holding (%)': [54.3, 54.1, 53.9, 53.7, 53.6, 53.4, 53.5, 53.3, 53.2]
})

with iv_tab:
    st.markdown("### 🧮 Implied Volatility Calculator (Black-Scholes)")
    S = st.number_input("Current Stock Price (S)", value=100.0)
    K = st.number_input("Strike Price (K)", value=100.0)
    T = st.number_input("Time to Expiry (in years, e.g., 0.5 for 6 months)", value=0.5)
    r = st.number_input("Risk-free Rate (%)", value=6.5) / 100
    market_price = st.number_input("Option Market Price", value=10.0)

    if st.button("Calculate IV"):
        iv = implied_volatility_call(S, K, T, r, market_price)
        if iv:
            st.success(f"🧮 Implied Volatility: {iv * 100:.2f}%")
        else:
            st.error("❌ Could not compute IV. Check inputs.")

with holding_tab:
    st.markdown("### 📊 Promoter Holding Over Time")
    if not promoter_data.empty:
        import plotly.express as px

        fig = px.line(
            promoter_data,
            x='Date',
            y='Promoter Holding (%)',
            markers=True,
            title=f"Promoter Holding Trend - {ticker.upper()}"
        )
        fig.update_layout(yaxis_title="Holding (%)", xaxis_title="Date")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Promoter holding data is unavailable.")

