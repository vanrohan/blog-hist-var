from datetime import datetime
import math
import numpy as np
import polars as pl

import seaborn as sns
import matplotlib.pyplot as plt
import ccxt

# CSV Columns: Timestamp,Open,High,Low,Close,Volume,TradeCount
btc_df = pl.read_csv(
    "XBTUSD_60.csv",
    has_header=False,
    new_columns=[
        "Timestamp",
        "BTC_Open",
        "BTC_High",
        "BTC_Low",
        "BTC_Close",
        "BTC_Volume",
        "BTC_TradeCount",
    ],
)
# CSV Columns: Timestamp,Open,High,Low,Close,Volume,TradeCount
eth_df = pl.read_csv(
    "ETHUSD_60.csv",
    has_header=False,
    new_columns=[
        "Timestamp",
        "ETH_Open",
        "ETH_High",
        "ETH_Low",
        "ETH_Close",
        "ETH_Volume",
        "ETH_TradeCount",
    ],
)

# Ensure both dataframes have Timestamp as Int64
btc_df = btc_df.with_columns(pl.col("Timestamp").cast(pl.Int64))
eth_df = eth_df.with_columns(pl.col("Timestamp").cast(pl.Int64))

# Sort both dataframes by Timestamp
btc_df = btc_df.sort("Timestamp")
eth_df = eth_df.sort("Timestamp")

# Merge the dataframes on Timestamp
merged_df = btc_df.join(eth_df, on="Timestamp", how="outer")
# Remove rows where Timestamp is null
merged_df = merged_df.filter(pl.col("Timestamp").is_not_null())


# Sort the merged dataframe by Timestamp
merged_df = merged_df.sort("Timestamp")

# Fill missing values with the previous value for all columns except Timestamp
columns_to_fill = [col for col in merged_df.columns if col != "Timestamp"]
merged_df = merged_df.with_columns([pl.col(col).forward_fill().backward_fill() for col in columns_to_fill])

# Filter data after 2017 for both BTC and ETH
start_timestamp = int(datetime(2018, 1, 1).timestamp())
merged_df = merged_df.filter(pl.col("Timestamp") >= start_timestamp)

# Calculate the natural log of the ratio of previous Close to current Close
# Looking at a 1-day period (24 hours)
merged_df = merged_df.with_columns(
    [
        (pl.col("BTC_Close").shift(24) / pl.col("BTC_Close")).log().alias("BTC_LogRatio"),
        (pl.col("ETH_Close").shift(24) / pl.col("ETH_Close")).log().alias("ETH_LogRatio"),
    ]
)
# Remove records where BTC_LogRatio or ETH_LogRatio is null
merged_df = merged_df.filter((pl.col("BTC_LogRatio").is_not_null()) & (pl.col("ETH_LogRatio").is_not_null()))

# Convert Timestamp to datetime for easier handling
merged_df = merged_df.with_columns([pl.col("Timestamp").cast(pl.Datetime).alias("DateTime")])

# Print summary statistics
print(merged_df.describe())

# Calculate correlation between BTC and ETH percentage changes
correlation = merged_df.select([pl.corr("BTC_LogRatio", "ETH_LogRatio").alias("Correlation")])
print("Correlation between BTC and ETH percentage changes:")
print(correlation)


# Print the first few rows of the merged dataframe to verify
print(merged_df.head())
print(merged_df.tail())


# Filter out NaN values from BTC_LogRatio and ETH_LogRatio
btc_log_ratio = merged_df.filter(pl.col("BTC_LogRatio").is_not_null())["BTC_LogRatio"].to_numpy()
eth_log_ratio = merged_df.filter(pl.col("ETH_LogRatio").is_not_null())["ETH_LogRatio"].to_numpy()

# Convert log ratios to returns
btc_returns = np.exp(btc_log_ratio) - 1
eth_returns = np.exp(eth_log_ratio) - 1

plt.figure(figsize=(12, 6))
plt.hist(btc_returns, bins=50, alpha=0.5, label="BTC", edgecolor="black")
plt.hist(eth_returns, bins=50, alpha=0.5, label="ETH", edgecolor="black")
plt.title("Histogram of BTC and ETH Returns (1-day period)")
plt.xlabel("Returns")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)

# Add vertical lines for means
btc_mean_return = np.mean(btc_returns)
eth_mean_return = np.mean(eth_returns)
plt.axvline(
    btc_mean_return,
    color="blue",
    linestyle="dashed",
    linewidth=2,
    label=f"BTC Mean: {btc_mean_return:.4f}",
)
plt.axvline(
    eth_mean_return,
    color="orange",
    linestyle="dashed",
    linewidth=2,
    label=f"ETH Mean: {eth_mean_return:.4f}",
)

plt.legend()
plt.tight_layout()
plt.show()

# Print some statistics about both log ratios
print("\nBTC and ETH Log Ratio Statistics:")
print(merged_df.select(pl.col("BTC_LogRatio"), pl.col("ETH_LogRatio")).describe())


# Calculate and print the correlation between BTC and ETH log ratios
log_ratio_correlation = merged_df.select([pl.corr("BTC_LogRatio", "ETH_LogRatio").alias("Log_Ratio_Correlation")])
print("\nCorrelation between BTC and ETH log ratios:")
print(log_ratio_correlation)

# Create a scatterplot to visualize the correlation
plt.figure(figsize=(10, 6))
plt.scatter(btc_returns, eth_returns, alpha=0.5)
plt.title("Correlation between BTC and ETH Returns")
plt.xlabel("BTC Return")
plt.ylabel("ETH Return")
plt.grid(True, alpha=0.3)

# Add a line of best fit
z = np.polyfit(merged_df["BTC_LogRatio"], merged_df["ETH_LogRatio"], 1)
p = np.poly1d(z)
plt.plot(merged_df["BTC_LogRatio"], p(merged_df["BTC_LogRatio"]), "r--", alpha=0.8)

# Add correlation coefficient to the plot
correlation_value = log_ratio_correlation[0, 0]
plt.text(
    0.05,
    0.95,
    f"Correlation: {correlation_value:.4f}",
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

plt.tight_layout()
plt.show()

# Print some statistics about both log ratios
print("\nBTC and ETH Log Ratio Statistics:")
print(merged_df.select(pl.col("BTC_LogRatio"), pl.col("ETH_LogRatio")).describe())

# Calculate and print the correlation between BTC and ETH log ratios
log_ratio_correlation = merged_df.select([pl.corr("BTC_LogRatio", "ETH_LogRatio").alias("Log_Ratio_Correlation")])
print("\nCorrelation between BTC and ETH log ratios:")
print(log_ratio_correlation)


TOTAL_CASH = 100000  # USD

# Fetch current market prices for BTC and ETH using ccxt
exchange = ccxt.binance()

try:
    btc_ticker = exchange.fetch_ticker("BTC/USDT")
    eth_ticker = exchange.fetch_ticker("ETH/USDT")

    current_btc_price = btc_ticker["last"]
    current_eth_price = eth_ticker["last"]

    print(f"Current BTC price: ${current_btc_price:.2f}")
    print(f"Current ETH price: ${current_eth_price:.2f}")

    # Allocate 50% of the portfolio to BTC and 50% to ETH by value
    BALANCED_PORTFOLIO = {
        "BTC": TOTAL_CASH / 2 / current_btc_price,
        "ETH": TOTAL_CASH / 2 / current_eth_price,
    }

    # Calculate balanced portfolio value
    portfolio_value = (BALANCED_PORTFOLIO["BTC"] * current_btc_price) + (BALANCED_PORTFOLIO["ETH"] * current_eth_price)
    print(f"Portfolio value: ${portfolio_value:.2f}")

except ccxt.NetworkError as e:
    print(f"Network error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Calculate and print the current portfolio value
btc_value = BALANCED_PORTFOLIO["BTC"] * current_btc_price
eth_value = BALANCED_PORTFOLIO["ETH"] * current_eth_price
total_value = btc_value + eth_value

print("\nBalanced Portfolio Breakdown:")
print(f"BTC: {BALANCED_PORTFOLIO['BTC']} BTC = ${btc_value:.2f}")
print(f"ETH: {BALANCED_PORTFOLIO['ETH']} ETH = ${eth_value:.2f}")
print(f"Total portfolio value: ${total_value:.2f}")


# Update portfolio allocation to 30% BTC and 70% ETH by value
REWEIGHTED_PORTFOLIO = {
    "BTC": 0.3 * TOTAL_CASH / current_btc_price,
    "ETH": 0.7 * TOTAL_CASH / current_eth_price,
}

print("\nReweighted Portfolio Breakdown (30% BTC, 70% ETH by value):")
print(f"BTC: {REWEIGHTED_PORTFOLIO['BTC']:.6f} BTC = ${btc_value:.2f}")
print(f"ETH: {REWEIGHTED_PORTFOLIO['ETH']:.6f} ETH = ${eth_value:.2f}")
print(f"Total portfolio value: ${total_value:.2f}")


# Function to calculate VaR for a given portfolio allocation
def calculate_var(btc_value, eth_value):
    portfolio_values = []
    for row in merged_df.iter_rows(named=True):
        btc_return = math.exp(row["BTC_LogRatio"]) - 1
        eth_return = math.exp(row["ETH_LogRatio"]) - 1

        new_btc_value = btc_value * (1 + btc_return)
        new_eth_value = eth_value * (1 + eth_return)
        new_total_value = new_btc_value + new_eth_value

        portfolio_values.append(new_total_value)

    portfolio_values_series = pl.Series("Portfolio_Values", portfolio_values)
    var_95 = portfolio_values_series.quantile(0.05)
    var_95_percentage = ((total_value - var_95) / total_value) * 100
    return var_95, var_95_percentage, portfolio_values_series


# Calculate VaR for the balanced portfolio (50% BTC, 50% ETH)
balanced_var_95, balanced_var_95_percentage, balanced_portfolio_values = calculate_var(btc_value, eth_value)
print(
    f"\n95% VaR (Balanced Portfolio): ${total_value - balanced_var_95:.2f} ({balanced_var_95_percentage:.2f}% of current portfolio value)"
)

# Calculate VaR for the original portfolio (higher ETH allocation)
reweighted_btc_value = REWEIGHTED_PORTFOLIO["BTC"] * current_btc_price
reweighted_eth_value = REWEIGHTED_PORTFOLIO["ETH"] * current_eth_price
reweighted_var_95, reweighted_var_95_percentage, reweighted_portfolio_values = calculate_var(reweighted_btc_value, reweighted_eth_value)
print(
    f"\n95% VaR (Reweighted Portfolio): ${(reweighted_btc_value + reweighted_eth_value) - reweighted_var_95:.2f} ({reweighted_var_95_percentage:.2f}% of original portfolio value)"
)

# Compare the two VaR results
print("\nComparison:")
print(f"Balanced Portfolio (50% BTC, 50% ETH) VaR: {balanced_var_95_percentage:.2f}%")
print(f"Reweighted Portfolio (30% BTC, 70% ETH) VaR: {reweighted_var_95_percentage:.2f}%")
print(f"Difference: {abs(balanced_var_95_percentage - reweighted_var_95_percentage):.2f}%")

# Plot the distribution of portfolio values for both allocations
plt.figure(figsize=(12, 6))
sns.histplot(
    balanced_portfolio_values,
    kde=True,
    color="blue",
    alpha=0.5,
    label="Balanced Portfolio",
)
sns.histplot(
    reweighted_portfolio_values,
    kde=True,
    color="orange",
    alpha=0.5,
    label="ETH Weighted Portfolio",
)
plt.title("Distribution of Simulated Portfolio Values")
plt.xlabel("Portfolio Value ($)")
plt.ylabel("Frequency")
plt.axvline(balanced_var_95, color="g", linestyle="--", label="Balanced Portfolio 95% VaR")
plt.axvline(
    reweighted_var_95,
    color="orange",
    linestyle="--",
    label="ETH Weighted Portfolio 95% VaR",
)
plt.legend()
plt.show()
