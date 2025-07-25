# 📈 Direction Forecasting in Financial Markets using Machine Learning

This project focuses on building a machine learning pipeline to predict next-day stock direction (up or down) using OHLCV data and technical indicators, enhanced with statistical testing, clustering, and Granger-based lag augmentation.

---

## 🗂️ Data Collection
- Collected **OHLCV** (Open, High, Low, Close, Volume) data 
- Multiple sectors across 1-year period (📅 **July 3, 2024 – July 3, 2025**)

---

## 🛠️ Feature Engineering
### Technical Indicators
Engineered **40+ technical indicators** including:
- `RSI_14`, `MACD`, `lag_volume_3`, `EMA_20`, `vol_rolling_mean_10`, etc.

### Data Processing
- Outlier treatment using **IQR method**
- Checked normality using **Shapiro-Wilk test** (stat, p-value)
- Computed **Spearman correlation coefficients**

### Statistical Validation
- Performed **Mann-Whitney U test** to:
  - Validate statistically significant correlations
  - Drop features with insignificant relationships

---

## 🎯 Target Variable
Created binary target:
```python
close_next_day = 1 if next_day_close > today_close else 0
```

---

## 🤖 Modeling and Evaluation

**Algorithms Used:**
- RandomForest
- LogisticRegression
- SVM
- XGBoost
- GradientBoosting
- AdaBoost
- NaiveBayes

**Optimization:**
- Hyperparameter tuning via **GridSearchCV**
- **Best performance**: XGBoost (~85% accuracy)

**Validation:**
- ✅ Confirmed no overfitting using **K-Fold Cross-Validation**

---

## 🧠 Market Regime Clustering
### Why Clustering?
Clustering helped uncover hidden patterns in market behavior and improve both model interpretability and predictive performance. Here's why it was essential:

#### 1. Understand Market Regimes
Financial markets don't behave uniformly.

There are periods of high volatility, bullish trends, panic sell-offs, or flat movement.

By clustering historical data based on technical indicators, we were able to segment the market into regimes like:

📈 High-activity (e.g., earnings days, news shocks)

 Calm/sideways (e.g., accumulation, low sentiment)

#### 2. Improve Model Specialization
A single global model may underperform due to mixed patterns from different market conditions.

By training separate models per cluster, we allowed each model to:

Specialize in predicting within its regime.

Capture localized patterns that would be averaged out in a global model.

#### 3. Generate Actionable Technical Insights
Clusters revealed interpretable traits:

Ex: Cluster 1 → high volume, high RSI, breakout setups → 🟢 Momentum trading

Ex: Cluster 0 → low volatility, stable EMA → 🟡 Swing or mean-reversion trading

This allowed us to bridge ML results with trading strategies, making the system usable for:

Traders

Financial analysts

Algorithmic trading systems

#### 4. Data-Driven Regime Shifts
Markets are non-stationary; clusters adapt to different time-based behaviors.

This avoids static modeling and builds toward a dynamic, regime-aware forecasting system.

### 1️⃣ KMeans Clustering

**Methodology:**
- Determined optimal clusters using:
  - Elbow Method
  - Silhouette Score

**Cluster Insights:**

| Cluster | Characteristics               | Market Interpretation                     |
|---------|-------------------------------|-------------------------------------------|
| 1       | 🔺 High volume, volatility    | High-activity regime (bull runs, earnings)|
| 0       | 🔻 Low volume, low volatility | Calm/Accumulation phase (sideways market) |

### 2️⃣ Agglomerative Clustering

**Cluster Characteristics:**

| Cluster | Technical Profile            | Market Regime            | Trading Scenarios         | Strategies               |
|---------|------------------------------|--------------------------|---------------------------|--------------------------|
| 1       | 🔺 High volatility           | Volatile/Active Phase    | Bull runs, earnings       | Momentum trading         |
| 0       | 🔻 Low volatility            | Sideways Market          | Consolidation periods     | Mean-reversion strategies|

---

## 🔁 Lag Based Feature Augmentation

In financial markets, the behavior of one stock or asset can influence another with a time lag. We implemented **Granger Causality-based lagged feature augmentation** to capture these cross-stock temporal dependencies and enhance predictive power.

---

### 📌 What is Granger Causality?

**Granger causality** is a statistical hypothesis test that determines whether past values of one time series help predict future values of another.

**Mathematical Formulation:**

For two stationary time series 𝑋ₜ and 𝑌ₜ:

1. **Univariate Model:**
   𝑌ₜ = Σ(αᵢ𝑌ₜ₋ᵢ) + εₜ (i=1 to p)

2. **Multivariate Model:**
   𝑌ₜ = Σ(αᵢ𝑌ₜ₋ᵢ) + Σ(βⱼ𝑋ₜ₋ⱼ) + εₜ' (i,j=1 to p)

If the multivariate model shows **statistically significant improvement** (via F-test), we conclude 𝑋 Granger-causes 𝑌.

---

###  Why Granger Causality Matters in Finance?

- Stock movements exhibit **interdependencies** (e.g., indices → stocks, sector leaders → peers)
- Captures **lead-lag relationships** in markets
- Models **information flow** between assets

---

###  Implementation

1. **Stock Selection:**
- Used sector leaders and index components as potential influencers

2. **Testing Procedure:**
- Applied Granger causality tests for all relevant stock pairs
- Determined optimal lag 𝑝 using **AIC/BIC** criteria
- Set significance threshold at **p-value < 0.05**

3. **Feature Creation:**
- Generated lagged features like:
  - `lag_close_stock_A_t-1`
  - `lag_rsi_stock_A_t-2`
- Included only **statistically significant** lags

---

### 📊 Example Results

**Case:** Stock A (Banking) → Stock B (NBFC)  
**Added Features:**
- `lag_close_A_t-1`
- `lag_volume_A_t-2`

**Impact:**  
✅ **3-5% improvement** in prediction AUC for Stock B

---

### ✅ Key Benefits

- Captures **cross-asset influences**
- Models **market information flow**
- Enhances **feature richness**
- Improves **prediction stability**
- Provides **interpretable relationships** between assets


---

## 📊 Final Takeaways

✅ **End-to-end forecasting system** for stock direction prediction 

✅ Advanced techniques applied:
   - Statistical validation
   - Market regime clustering  
   - Lag based Feature augmentation using Granger Causality
   
✅ **Domain knowledge** combined with ML for robust modeling
