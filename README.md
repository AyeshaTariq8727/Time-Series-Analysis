# Time Series Analysis: Methods and Use Cases in Forecasting

## Introduction
Time Series Analysis is used to analyze sequential data points collected over time. It helps in understanding trends, seasonal patterns, and making future predictions. This is widely used in finance, sales forecasting, weather prediction, and stock market analysis.

### Topics Covered:
- ✅ Time Series Components (Trend, Seasonality, Cyclicity, Noise)
- ✅ Key Methods (Moving Average, Exponential Smoothing, ARIMA, LSTM)
- ✅ Real-World Example using Python

---

## 1. Understanding Time Series Components

### 1.1 Trend
A long-term increase or decrease in data over time.
Example: Increase in stock prices over a decade.

### 1.2 Seasonality
Repeating patterns at regular intervals (e.g., monthly, yearly).
Example: Retail sales increase in December due to Christmas.

### 1.3 Cyclic Patterns
Fluctuations that occur over long, irregular intervals.
Example: Economic cycles (recession & boom) lasting several years.

### 1.4 Noise (Irregularity)
Random variations with no pattern.
Example: Sudden stock market crashes due to unforeseen events.

### 🔹 Visualizing Time Series Data
```python
import pandas as pd  
import matplotlib.pyplot as plt  

# Load dataset (example: airline passenger data)
df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv", parse_dates=['Month'], index_col='Month')  

# Plot time series
plt.figure(figsize=(10,5))
plt.plot(df, label="Passengers")
plt.title("Airline Passengers Over Time")
plt.xlabel("Year")
plt.ylabel("Number of Passengers")
plt.legend()
plt.show()
```
✅ Observation: There is a clear increasing trend and seasonal patterns.

---

## 2. Common Time Series Analysis Methods

### 2.1 Moving Average (MA) Smoothing
Moving Average helps smooth out fluctuations and highlights trends.
```python
df["MA_12"] = df["Passengers"].rolling(window=12).mean()

plt.figure(figsize=(10,5))
plt.plot(df["Passengers"], label="Original Data")
plt.plot(df["MA_12"], label="12-Month Moving Average", color="red")
plt.legend()
plt.show()
```
✅ Observation: The red line shows the smoothed trend.

---

### 2.2 Exponential Smoothing
Gives more weight to recent data points for better trend detection.
```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing  

model = ExponentialSmoothing(df["Passengers"], trend="add", seasonal="add", seasonal_periods=12).fit()  
df["Smoothed"] = model.fittedvalues  

plt.figure(figsize=(10,5))
plt.plot(df["Passengers"], label="Original Data")
plt.plot(df["Smoothed"], label="Exponential Smoothing", color="red")
plt.legend()
plt.show()
```
✅ Useful for sales forecasting where recent data is more relevant.

---

### 2.3 ARIMA (AutoRegressive Integrated Moving Average)
ARIMA is a powerful model for forecasting non-seasonal time series.

#### Steps for ARIMA Modeling:
1. Check Stationarity (data should have a constant mean/variance).
2. Make Data Stationary (by differencing or transformation).
3. Find ARIMA Parameters (p, d, q).

```python
from statsmodels.tsa.stattools import adfuller  

# Check stationarity using Augmented Dickey-Fuller test
result = adfuller(df["Passengers"])
print(f"ADF Statistic: {result[0]}")
print(f"P-value: {result[1]}")
```
✅ If p-value > 0.05, the data is non-stationary and needs differencing.

#### Fitting ARIMA Model
```python
from statsmodels.tsa.arima.model import ARIMA  

# Fit ARIMA model (p=2, d=1, q=2 as an example)
model = ARIMA(df["Passengers"], order=(2,1,2))
model_fit = model.fit()
df["ARIMA_Pred"] = model_fit.fittedvalues  

# Plot results
plt.figure(figsize=(10,5))
plt.plot(df["Passengers"], label="Original Data")
plt.plot(df["ARIMA_Pred"], label="ARIMA Prediction", color="red")
plt.legend()
plt.show()
```
✅ ARIMA is widely used in finance and demand forecasting.

---

### 2.4 LSTM (Long Short-Term Memory Neural Network)
LSTM is a deep learning model ideal for complex and long-term dependencies in time series.
```python
import tensorflow as tf  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import LSTM, Dense  

# Prepare data for LSTM (reshape for neural network input)
X = df["Passengers"].values.reshape(-1,1)

# Define LSTM model
model = Sequential([
    LSTM(50, activation="relu", return_sequences=True, input_shape=(1,1)),
    LSTM(50, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")  
model.fit(X, df["Passengers"].values, epochs=50, batch_size=8, verbose=0)
```
✅ LSTM is useful for complex time series like weather forecasting.

---

## 3. Real-World Use Cases of Time Series Analysis

### 3.1 Stock Market Prediction
Stock price movement can be analyzed using ARIMA and LSTM.
Example: Predicting Apple (AAPL) stock price trends based on historical data.

### 3.2 Sales Forecasting in Retail
Companies use Exponential Smoothing to predict sales spikes (e.g., Black Friday).

### 3.3 Weather Forecasting
LSTMs are used for predicting temperature and rainfall patterns.

### 3.4 Demand Forecasting in Supply Chain
Moving Average & ARIMA help optimize inventory levels.

---

## Conclusion
✔ Explored key time series components (trend, seasonality, cyclic patterns, noise).
✔ Applied forecasting techniques like Moving Average, ARIMA, and LSTM.
✔ Covered real-world use cases in business, finance, and weather.

---

## 📌 How to Use This Code
1. Clone this repository.
2. Install dependencies: `pip install pandas matplotlib statsmodels tensorflow`
3. Run each section of the notebook to analyze and forecast time series data.

📢 **Contributions & Feedback Welcome!** 🚀
