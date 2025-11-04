# KO–PEP Cointegration Pairs Trading Strategy

This project was originally developed inside a custom Python environment (`quantra_py`)

## Overview
This project implements a **statistical arbitrage strategy** based on **cointegration** between  
**The Coca-Cola Company (KO)** and **PepsiCo (PEP)** stocks.

The goal is to exploit **mean-reverting relationships** by identifying deviations from equilibrium  
and trading when the spread between both assets becomes statistically extreme.

The model uses:
- **OLS hedge ratio (β)** estimation without intercept  
- **Rolling z-score (90 days)** to measure deviation from the mean  
- **Rolling percentile thresholds (250 days)** for adaptive entry and exit zones  
- **State-based position logic** (exit → long → short priority)  
- **Cumulative return backtesting** for performance evaluation  

---

##  Methodology

### 1️⃣ Hedge Ratio (β)
Estimated via Ordinary Least Squares (OLS) regression **without intercept**:
\[
KO_t = \beta \cdot PEP_t + \varepsilon_t
\]

The hedge ratio represents how many units of PEP are required to hedge one unit of KO.  
The **spread** is defined as:
\[
spread_t = KO_t - \beta \cdot PEP_t
\]

---

### 2️⃣ Rolling Z-Score (90 Days)
Measures how many standard deviations the current spread deviates from its mean:
\[
z_t = \frac{spread_t - \mu_{90}}{\sigma_{90}}
\]

- A shorter window → more sensitive to recent changes  
- A longer window → smoother but slower response  

---

### 3️⃣ Rolling Thresholds (250 Days)
Dynamic entry and exit levels are calculated from rolling percentiles:
- Upper threshold = 97.5th percentile → short entry  
- Lower threshold = 2.5th percentile → long entry  
- Exit band = 20% of the range between upper and lower thresholds  

This allows the strategy to adapt to changing market volatility.

---

### 4️⃣ Trading Logic

| Condition | Signal | Action |
|------------|---------|--------|
| z < Lower | Long | Buy KO, Sell PEP |
| z > Upper | Short | Sell KO, Buy PEP |
| |z| < Exit Band | Exit | Close position |

- **Exit** has the highest priority (it always closes existing trades).  
- Long/short signals open new positions when no position is active.  

---

### 5️⃣ Returns & Backtest

Dollar-neutral daily return:
\[
r_t = w_{KO,t-1} \cdot r_{KO,t} + w_{PEP,t-1} \cdot r_{PEP,t}
\]

where:
- \( w_{KO} = position \)
- \( w_{PEP} = -\beta \times position \)

Cumulative return:
\[
cum\_ret_t = \prod_{i=1}^{t}(1 + r_i)
\]

No look-ahead bias: positions use a **1-day shift** before returns are applied.

---

## Results
The script outputs:
1. **Z-Score & Threshold Chart** → Entry/exit zones over time  
2. **Cumulative Return Curve** → Performance evolution  
3. **ADF p-value** → Confirms spread stationarity  

---

##Requirements
Create a `requirements.txt` file in the project folder containing:

yfinance==0.2.44
pandas==2.2.2
numpy==1.26.4
statsmodels==0.14.2
matplotlib==3.8.4

## How to Run

1️⃣ **Clone this repository**
```bash
git clone https://github.com/diegochm/KO-PEP-Cointegration-Strategy.git


2️⃣ Install dependencies

pip install -r requirements.txt


3️⃣ Run the script

python KO_and_Pepsi.py



--Next Improvements

    Rolling hedge ratio (β) recalculated every 90–120 days

    Dynamic cointegration testing (ADF) to detect structural breaks

    Multi-pair portfolio and capital allocation

    Performance metrics: Sharpe, Sortino, and Max Drawdown


Author

Diego Choquesillo
Quantitative Finance Enthusiast | Portfolio & Algorithmic Trading Developer
📍 Melbourne, Australia
