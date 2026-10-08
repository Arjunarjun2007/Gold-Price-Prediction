import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# -----------------------------------
# 1. Load Dataset
# -----------------------------------

print("Loading dataset...")

df = pd.read_csv("gold_data.csv")

print("\nDataset loaded successfully!")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# -----------------------------------
# 2. Convert Date
# -----------------------------------

df["Date"] = pd.to_datetime(df["Date"])

# Sort by date
df = df.sort_values("Date").reset_index(drop=True)


# -----------------------------------
# 3. Create Features
# -----------------------------------

# Previous day's closing price
df["Previous_Close"] = df["Close"].shift(1)

# 7-day moving average
df["MA_7"] = df["Close"].rolling(window=7).mean()

# 14-day moving average
df["MA_14"] = df["Close"].rolling(window=14).mean()

# 30-day moving average
df["MA_30"] = df["Close"].rolling(window=30).mean()


# -----------------------------------
# 4. Remove Missing Values
# -----------------------------------

df = df.dropna().reset_index(drop=True)

print("\nAfter preprocessing:")
print("Shape:", df.shape)


# -----------------------------------
# 5. Select Features and Target
# -----------------------------------

features = [
    "Open",
    "High",
    "Low",
    "Volume",
    "Previous_Close",
    "MA_7",
    "MA_14",
    "MA_30"
]

X = df[features]
y = df["Close"]


# -----------------------------------
# 6. Time-Based Train/Test Split
# -----------------------------------

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# -----------------------------------
# 7. Train Random Forest Model
# -----------------------------------

print("\nTraining Random Forest model...")

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# -----------------------------------
# 8. Make Predictions
# -----------------------------------

y_pred = model.predict(X_test)


# -----------------------------------
# 9. Evaluate Model
# -----------------------------------

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)


print("\n===================================")
print("MODEL PERFORMANCE")
print("===================================")

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")


# -----------------------------------
# 10. Save Model
# -----------------------------------

joblib.dump(model, "gold_price_model.pkl")

print("\nModel saved successfully!")

print("\nCreated file:")
print("gold_price_model.pkl")