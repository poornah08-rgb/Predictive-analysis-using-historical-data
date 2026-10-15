"""
Project: Predictive Analytics Using Historical Data
Description: Cleans historical sales data, trains a regression model,
evaluates performance, visualizes predictions, and forecasts the next 30 days.

Run:
    pip install pandas numpy matplotlib scikit-learn
    python predictive_analytics.py
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)

OUTPUT_DIR = Path("project_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def create_sample_data(days=365, seed=42):
    """Create realistic daily sales history when no CSV is supplied."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1),
                          periods=days, freq="D")
    day_number = np.arange(days)
    weekly_pattern = 90 * np.sin(2 * np.pi * day_number / 7)
    yearly_pattern = 160 * np.sin(2 * np.pi * day_number / 365)
    trend = 1.2 * day_number
    noise = rng.normal(0, 75, days)
    sales = 1200 + trend + weekly_pattern + yearly_pattern + noise
    sales = np.maximum(sales, 100).round(2)

    data = pd.DataFrame({"Date": dates, "Sales": sales})
    # Add a few missing values to demonstrate cleaning.
    if days > 60:
        missing_indices = rng.choice(np.arange(days), size=3, replace=False)
        data.loc[missing_indices, "Sales"] = np.nan
    return data


def load_and_clean_data(csv_path="sales_history.csv"):
    """Load a CSV if present; otherwise generate sample data."""
    path = Path(csv_path)
    if path.exists():
        df = pd.read_csv(path)
        # Accept common column-name variants.
        lower_to_original = {str(c).strip().lower(): c for c in df.columns}
        date_col = next((lower_to_original[c] for c in ("date", "order date", "sales date")
                         if c in lower_to_original), None)
        sales_col = next((lower_to_original[c] for c in ("sales", "revenue", "amount")
                          if c in lower_to_original), None)
        if date_col is None or sales_col is None:
            raise ValueError(
                "CSV must contain a date column (Date) and a numeric sales column (Sales)."
            )
        df = df.rename(columns={date_col: "Date", sales_col: "Sales"})
    else:
        print("sales_history.csv not found. Creating sample historical sales data.")
        df = create_sample_data()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    df = df.groupby("Date", as_index=False)["Sales"].sum()
    df = df.set_index("Date").asfreq("D")
    # Fill missing sales values using time interpolation, then nearby values if needed.
    df["Sales"] = df["Sales"].interpolate(method="time").ffill().bfill()
    df = df.reset_index()
    if len(df) < 45:
        raise ValueError("Please provide at least 45 days of historical data.")
    df.to_csv(OUTPUT_DIR / "cleaned_sales_data.csv", index=False)
    return df


def add_features(df):
    """Create calendar and lag features using only information available by each date."""
    result = df.copy()
    result["time_index"] = (result["Date"] - result["Date"].min()).dt.days
    result["week_sin"] = np.sin(2 * np.pi * result["Date"].dt.dayofweek / 7)
    result["week_cos"] = np.cos(2 * np.pi * result["Date"].dt.dayofweek / 7)
    result["year_sin"] = np.sin(2 * np.pi * result["Date"].dt.dayofyear / 365.25)
    result["year_cos"] = np.cos(2 * np.pi * result["Date"].dt.dayofyear / 365.25)
    result["day_of_week"] = result["Date"].dt.dayofweek
    result["day_of_month"] = result["Date"].dt.day
    result["month"] = result["Date"].dt.month
    result["day_of_year"] = result["Date"].dt.dayofyear
    result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)
    result["lag_1"] = result["Sales"].shift(1)
    result["lag_7"] = result["Sales"].shift(7)
    result["rolling_mean_7"] = result["Sales"].shift(1).rolling(7).mean()
    result["rolling_mean_30"] = result["Sales"].shift(1).rolling(30).mean()
    return result.dropna().reset_index(drop=True)


def main():
    df = load_and_clean_data()
    featured = add_features(df)

    feature_columns = [
        "time_index", "week_sin", "week_cos", "year_sin", "year_cos",
        "day_of_week", "day_of_month", "month", "day_of_year", "is_weekend"
    ]
    X = featured[feature_columns]
    y = featured["Sales"]

    # Time-ordered split avoids randomly mixing future and past observations.
    split_index = int(len(featured) * 0.80)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # Linear regression with calendar cycles can extrapolate the time trend.
    model = LinearRegression()
    model.fit(X_train, y_train)
    test_predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, test_predictions)
    rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    r2 = r2_score(y_test, test_predictions)

    metrics = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R2 score"],
        "Value": [mae, rmse, r2]
    })
    metrics.to_csv(OUTPUT_DIR / "model_evaluation.csv", index=False)

    comparison = pd.DataFrame({
        "Date": featured.loc[y_test.index, "Date"].values,
        "Actual_Sales": y_test.values,
        "Predicted_Sales": test_predictions.round(2)
    })
    comparison.to_csv(OUTPUT_DIR / "actual_vs_predicted.csv", index=False)

    # Plot actual and predicted sales for the test period.
    plt.figure(figsize=(12, 5))
    plt.plot(comparison["Date"], comparison["Actual_Sales"], label="Actual sales")
    plt.plot(comparison["Date"], comparison["Predicted_Sales"], label="Predicted sales")
    plt.title("Actual vs Predicted Sales")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()

    # Forecast 30 future days recursively using the latest available sales values.
    history = df.set_index("Date")["Sales"].copy()
    future_rows = []
    for _ in range(30):
        next_date = history.index.max() + pd.Timedelta(days=1)
        recent = history
        row = {
            "time_index": int((next_date - df["Date"].min()).days),
            "week_sin": float(np.sin(2 * np.pi * next_date.dayofweek / 7)),
            "week_cos": float(np.cos(2 * np.pi * next_date.dayofweek / 7)),
            "year_sin": float(np.sin(2 * np.pi * next_date.dayofyear / 365.25)),
            "year_cos": float(np.cos(2 * np.pi * next_date.dayofyear / 365.25)),
            "day_of_week": next_date.dayofweek,
            "day_of_month": next_date.day,
            "month": next_date.month,
            "day_of_year": next_date.dayofyear,
            "is_weekend": int(next_date.dayofweek >= 5),
            "lag_1": float(recent.iloc[-1]),
            "lag_7": float(recent.iloc[-7]),
            "rolling_mean_7": float(recent.iloc[-7:].mean()),
            "rolling_mean_30": float(recent.iloc[-30:].mean()),
        }
        predicted = float(model.predict(pd.DataFrame([row])[feature_columns])[0])
        predicted = max(0, predicted)
        future_rows.append({"Date": next_date, "Predicted_Sales": round(predicted, 2)})
        history.loc[next_date] = predicted

    forecast = pd.DataFrame(future_rows)
    forecast.to_csv(OUTPUT_DIR / "next_30_days_forecast.csv", index=False)

    plt.figure(figsize=(12, 5))
    recent_history = df.tail(60)
    plt.plot(recent_history["Date"], recent_history["Sales"], label="Historical sales")
    plt.plot(forecast["Date"], forecast["Predicted_Sales"], marker="o",
             markersize=3, label="30-day forecast")
    plt.title("Historical Sales and 30-Day Forecast")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sales_forecast.png", dpi=150)
    plt.close()

    # Coefficients help explain the direction/size of each feature effect.
    importance = pd.DataFrame({
        "Feature": feature_columns,
        "Coefficient": model.coef_
    })
    importance["Absolute_Coefficient"] = importance["Coefficient"].abs()
    importance = importance.sort_values("Absolute_Coefficient", ascending=False)
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    print("\n=== PREDICTIVE ANALYTICS RESULTS ===")
    print(f"Historical records: {len(df)}")
    print(f"Training records:   {len(X_train)}")
    print(f"Testing records:    {len(X_test)}")
    print(f"MAE:                {mae:.2f}")
    print(f"RMSE:               {rmse:.2f}")
    print(f"R2 score:           {r2:.3f}")
    print("\nNext 5 days forecast:")
    print(forecast.head().to_string(index=False))
    print(f"\nAll output files saved in: {OUTPUT_DIR.resolve()}")
    print("Open sales_forecast.png and actual_vs_predicted.png to view the graphs.")


if __name__ == "__main__":
    main()
