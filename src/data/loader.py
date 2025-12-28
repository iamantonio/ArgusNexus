"""
V4 Data Loader - Clean Fuel for the Truth Engine

A Truth Engine cannot run on dirty fuel.
This module ensures all historical data is:
- Sorted chronologically
- De-duplicated
- Gap-filled (forward-fill for missing bars)
- Validated for OHLC consistency

Supported formats:
- Yahoo Finance CSV
- Generic OHLCV CSV
- Coinbase/Exchange exports

Supported data sources:
- Yahoo Finance (fetch_yahoo_data) - Good for backtesting
- Coinbase API (fetch_coinbase_data) - Use for live/paper trading

Usage:
    from data.loader import load_ohlcv, fetch_yahoo_data, fetch_coinbase_data

    # Load from CSV
    df = load_ohlcv("data/btc_history.csv")

    # Fetch from Yahoo Finance (backtesting)
    df = fetch_yahoo_data("BTC-USD", start="2024-01-01", end="2024-12-31")

    # Fetch from Coinbase API (live/paper trading)
    df = fetch_coinbase_data("BTC-USD", lookback_days=120, interval="4h")
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple
from decimal import Decimal

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# Standard column mappings for different data sources
COLUMN_MAPPINGS = {
    "yahoo": {
        "Date": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume"
    },
    "coinbase": {
        "time": "timestamp",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume"
    },
    "generic": {
        "timestamp": "timestamp",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume"
    }
}


class DataQualityReport:
    """Report on data quality after loading and sanitization."""

    def __init__(self):
        self.original_rows = 0
        self.final_rows = 0
        self.duplicates_removed = 0
        self.gaps_filled = 0
        self.ohlc_violations_fixed = 0
        self.date_range: Tuple[datetime, datetime] = (None, None)
        self.warnings: List[str] = []

    def __str__(self) -> str:
        return (
            f"Data Quality Report:\n"
            f"  Original rows: {self.original_rows}\n"
            f"  Final rows: {self.final_rows}\n"
            f"  Duplicates removed: {self.duplicates_removed}\n"
            f"  Gaps filled: {self.gaps_filled}\n"
            f"  OHLC violations fixed: {self.ohlc_violations_fixed}\n"
            f"  Date range: {self.date_range[0]} to {self.date_range[1]}\n"
            f"  Warnings: {len(self.warnings)}"
        )


def detect_format(df: pd.DataFrame) -> str:
    """Detect the data format based on column names."""
    columns = set(df.columns.str.lower())

    if "date" in columns and "adj close" in [c.lower() for c in df.columns]:
        return "yahoo"
    elif "time" in columns:
        return "coinbase"
    else:
        return "generic"


def normalize_columns(df: pd.DataFrame, format_type: str) -> pd.DataFrame:
    """Normalize column names to standard format."""
    mapping = COLUMN_MAPPINGS.get(format_type, COLUMN_MAPPINGS["generic"])

    # Create case-insensitive mapping
    df_columns = {c.lower(): c for c in df.columns}
    rename_map = {}

    for source_col, target_col in mapping.items():
        source_lower = source_col.lower()
        if source_lower in df_columns:
            rename_map[df_columns[source_lower]] = target_col

    df = df.rename(columns=rename_map)

    # Ensure required columns exist
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    return df


def parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamp column to datetime."""
    if df["timestamp"].dtype == "object":
        # Try multiple date formats
        for fmt in [None, "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y"]:
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"], format=fmt)
                break
            except (ValueError, TypeError):
                continue

    # Ensure timezone-naive for consistency
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    return df


def remove_duplicates(df: pd.DataFrame, report: DataQualityReport) -> pd.DataFrame:
    """Remove duplicate timestamps, keeping the last occurrence."""
    original_len = len(df)
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    report.duplicates_removed = original_len - len(df)

    if report.duplicates_removed > 0:
        report.warnings.append(f"Removed {report.duplicates_removed} duplicate timestamps")

    return df


def sort_by_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Sort data chronologically."""
    return df.sort_values("timestamp").reset_index(drop=True)


def detect_timeframe(df: pd.DataFrame) -> timedelta:
    """Detect the data timeframe (1m, 1h, 1d, etc.)."""
    if len(df) < 2:
        return timedelta(days=1)  # Default to daily

    # Calculate median time difference
    time_diffs = df["timestamp"].diff().dropna()
    median_diff = time_diffs.median()

    return median_diff


def fill_gaps(df: pd.DataFrame, report: DataQualityReport) -> pd.DataFrame:
    """Fill gaps in the data using forward-fill."""
    timeframe = detect_timeframe(df)

    # Only fill gaps for intraday data
    if timeframe >= timedelta(days=1):
        return df  # Daily data may have legitimate gaps (weekends, holidays)

    # Create complete date range
    start = df["timestamp"].min()
    end = df["timestamp"].max()
    full_range = pd.date_range(start=start, end=end, freq=timeframe)

    # Reindex and forward-fill
    df = df.set_index("timestamp")
    original_len = len(df)
    df = df.reindex(full_range)
    df = df.ffill()  # Forward fill prices
    df = df.reset_index().rename(columns={"index": "timestamp"})

    report.gaps_filled = len(df) - original_len

    if report.gaps_filled > 0:
        report.warnings.append(f"Filled {report.gaps_filled} missing bars via forward-fill")

    return df


def fix_ohlc_consistency(df: pd.DataFrame, report: DataQualityReport) -> pd.DataFrame:
    """Ensure OHLC values are consistent (high >= all, low <= all)."""
    violations = 0

    # High should be >= open, close
    mask = (df["high"] < df["open"]) | (df["high"] < df["close"])
    if mask.any():
        df.loc[mask, "high"] = df.loc[mask, ["open", "high", "close"]].max(axis=1)
        violations += mask.sum()

    # Low should be <= open, close
    mask = (df["low"] > df["open"]) | (df["low"] > df["close"])
    if mask.any():
        df.loc[mask, "low"] = df.loc[mask, ["open", "low", "close"]].min(axis=1)
        violations += mask.sum()

    report.ohlc_violations_fixed = violations

    if violations > 0:
        report.warnings.append(f"Fixed {violations} OHLC consistency violations")

    return df


def validate_data(df: pd.DataFrame, report: DataQualityReport) -> bool:
    """Final validation of data quality."""
    is_valid = True

    # Check for NaN values in critical columns
    critical_cols = ["timestamp", "open", "high", "low", "close"]
    for col in critical_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            report.warnings.append(f"Column '{col}' has {nan_count} NaN values")
            is_valid = False

    # Check for negative prices
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            report.warnings.append(f"Column '{col}' has {neg_count} negative values")
            is_valid = False

    # Check for zero volume (warning only)
    zero_vol = (df["volume"] == 0).sum()
    if zero_vol > len(df) * 0.1:  # More than 10% zero volume
        report.warnings.append(f"{zero_vol} bars ({zero_vol/len(df)*100:.1f}%) have zero volume")

    return is_valid


def load_ohlcv(
    filepath: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, DataQualityReport]:
    """
    Load and sanitize OHLCV data from a CSV file.

    This is THE function for loading historical data into V4.
    It ensures data quality before the Truth Engine consumes it.

    Args:
        filepath: Path to the CSV file
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        verbose: Print quality report

    Returns:
        Tuple of (sanitized DataFrame, DataQualityReport)
    """
    report = DataQualityReport()

    # Load raw data
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)
    report.original_rows = len(df)

    if verbose:
        print(f"Loading {filepath}...")
        print(f"  Raw rows: {len(df)}")

    # Detect format and normalize columns
    format_type = detect_format(df)
    if verbose:
        print(f"  Detected format: {format_type}")

    df = normalize_columns(df, format_type)

    # Parse timestamps
    df = parse_timestamp(df)

    # Sort chronologically
    df = sort_by_timestamp(df)

    # Remove duplicates
    df = remove_duplicates(df, report)

    # Fill gaps (for intraday data)
    df = fill_gaps(df, report)

    # Fix OHLC consistency
    df = fix_ohlc_consistency(df, report)

    # Apply date filters
    if start_date:
        start_dt = pd.to_datetime(start_date)
        df = df[df["timestamp"] >= start_dt]

    if end_date:
        end_dt = pd.to_datetime(end_date)
        df = df[df["timestamp"] <= end_dt]

    # Final validation
    validate_data(df, report)

    # Update report
    report.final_rows = len(df)
    if len(df) > 0:
        report.date_range = (df["timestamp"].iloc[0], df["timestamp"].iloc[-1])

    if verbose:
        print(f"  Final rows: {len(df)}")
        print(f"  Date range: {report.date_range[0]} to {report.date_range[1]}")
        if report.warnings:
            print(f"  Warnings: {len(report.warnings)}")
            for w in report.warnings:
                print(f"    - {w}")

    return df, report


def fetch_yahoo_data(
    symbol: str,
    start: str,
    end: str,
    interval: str = "1h",
    verbose: bool = True
) -> Tuple[pd.DataFrame, DataQualityReport]:
    """
    Fetch historical data from Yahoo Finance.

    Args:
        symbol: Ticker symbol (e.g., "BTC-USD")
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        interval: Data interval ("1m", "5m", "15m", "1h", "1d")
        verbose: Print progress

    Returns:
        Tuple of (sanitized DataFrame, DataQualityReport)
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required. Install with: pip install yfinance")

    report = DataQualityReport()

    if verbose:
        print(f"Fetching {symbol} from Yahoo Finance...")
        print(f"  Period: {start} to {end}")
        print(f"  Interval: {interval}")

    # Fetch data
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval=interval)

    if df.empty:
        raise ValueError(f"No data returned for {symbol}")

    report.original_rows = len(df)

    # Reset index to get timestamp as column
    df = df.reset_index()

    # Rename columns to standard format
    df = df.rename(columns={
        "Date": "timestamp",
        "Datetime": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })

    # Keep only required columns
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    # Ensure timezone-naive
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    # Sort and clean
    df = sort_by_timestamp(df)
    df = remove_duplicates(df, report)
    df = fix_ohlc_consistency(df, report)

    # Validate
    validate_data(df, report)

    # Update report
    report.final_rows = len(df)
    if len(df) > 0:
        report.date_range = (df["timestamp"].iloc[0], df["timestamp"].iloc[-1])

    if verbose:
        print(f"  Downloaded: {report.original_rows} bars")
        print(f"  Final: {report.final_rows} bars")
        print(f"  Date range: {report.date_range[0]} to {report.date_range[1]}")

    return df, report


def save_ohlcv(df: pd.DataFrame, filepath: str) -> None:
    """Save OHLCV data to CSV for caching."""
    df.to_csv(filepath, index=False)
    logger.info(f"Saved {len(df)} bars to {filepath}")


def resample_ohlcv(
    df: pd.DataFrame,
    target_interval: str,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Resample OHLCV data to a larger timeframe.

    This is THE function for zooming out.
    Converts 1H to 4H, 1H to 1D, etc.

    Args:
        df: DataFrame with timestamp, open, high, low, close, volume
        target_interval: Target interval ("4H", "1D", "1W", etc.)
        verbose: Print progress

    Returns:
        Resampled DataFrame
    """
    if verbose:
        print(f"Resampling from {len(df)} bars to {target_interval}...")

    # Ensure timestamp is index for resampling
    df = df.copy()
    df = df.set_index("timestamp")

    # OHLCV resampling rules:
    # - Open: first value
    # - High: max value
    # - Low: min value
    # - Close: last value
    # - Volume: sum
    resampled = df.resample(target_interval).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    })

    # Drop NaN rows (incomplete bars at the end)
    resampled = resampled.dropna()

    # Reset index to get timestamp as column
    resampled = resampled.reset_index()

    if verbose:
        print(f"  Resampled to {len(resampled)} bars")
        if len(resampled) > 0:
            print(f"  Date range: {resampled['timestamp'].iloc[0]} to {resampled['timestamp'].iloc[-1]}")

    return resampled


# =============================================================================
# Convenience function for the backtester
# =============================================================================

def load_or_fetch_btc(
    start: str = "2024-01-01",
    end: str = "2024-12-31",
    interval: str = "1h",
    cache_dir: str = "data"
) -> pd.DataFrame:
    """
    Load BTC-USD data from cache or fetch from Yahoo Finance.

    This is the convenience function for backtesting.

    Args:
        start: Start date
        end: End date
        interval: Data interval
        cache_dir: Directory to cache downloaded data

    Returns:
        Sanitized DataFrame ready for backtesting
    """
    cache_path = Path(cache_dir) / f"btc_usd_{start}_{end}_{interval}.csv"

    if cache_path.exists():
        print(f"Loading cached data from {cache_path}")
        df, report = load_ohlcv(str(cache_path))
    else:
        print(f"Fetching fresh data from Yahoo Finance...")
        df, report = fetch_yahoo_data(
            symbol="BTC-USD",
            start=start,
            end=end,
            interval=interval
        )
        # Cache for future use
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_ohlcv(df, str(cache_path))
        print(f"Cached to {cache_path}")

    return df


# =============================================================================
# Coinbase Data Fetcher - For Live/Paper Trading
# =============================================================================

def fetch_coinbase_data(
    symbol: str,
    lookback_days: int = 120,
    interval: str = "4h",
    verbose: bool = True
) -> Tuple[pd.DataFrame, DataQualityReport]:
    """
    Fetch historical data from Coinbase API.

    This is THE function for live/paper trading data.
    Uses the same data source as live execution - eliminates data drift.

    Args:
        symbol: Trading pair (e.g., "BTC-USD")
        lookback_days: Number of days to fetch (default 120 for 200 bars at 4H)
        interval: Target interval ("1h", "4h", "1d")
        verbose: Print progress

    Returns:
        Tuple of (sanitized DataFrame, DataQualityReport)

    Note:
        Coinbase doesn't have 4H directly, so we fetch 1H and resample.
        Supported Coinbase intervals: ONE_HOUR, TWO_HOUR, SIX_HOUR, ONE_DAY
    """
    import sys
    import os
    from datetime import timezone

    # Add connector path (configurable via environment)
    v3_connector_path = os.getenv("COINBASE_CONNECTOR_PATH", "./src")
    if v3_connector_path not in sys.path:
        sys.path.insert(0, v3_connector_path)

    # Load environment variables if not already loaded
    if not os.getenv("COINBASE_API_KEY"):
        from dotenv import load_dotenv
        load_dotenv()

    try:
        from connectors.coinbase.client import CoinbaseClient
    except ImportError as e:
        raise ImportError(
            f"Coinbase connector not available: {e}\n"
            "Set COINBASE_CONNECTOR_PATH environment variable"
        )

    report = DataQualityReport()

    if verbose:
        print(f"Fetching {symbol} from Coinbase API...")
        print(f"  Lookback: {lookback_days} days")
        print(f"  Target interval: {interval}")

    # Initialize Coinbase client
    try:
        client = CoinbaseClient()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Coinbase client: {e}")

    # Calculate date range
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=lookback_days)

    if verbose:
        print(f"  Date range: {start_dt.date()} to {end_dt.date()}")

    # Coinbase granularity mapping
    # We always fetch at 1H resolution and resample for intervals > 1H
    coinbase_granularity = "ONE_HOUR"
    fetch_interval = "1h"

    # Fetch data in chunks (Coinbase limit is 300 candles per request)
    all_candles = []
    chunk_size_hours = 250  # Leave some buffer below 300

    current_start = start_dt
    while current_start < end_dt:
        current_end = min(current_start + timedelta(hours=chunk_size_hours), end_dt)

        try:
            result = client.product_candles.get_product_candles(
                product_id=symbol,
                start=current_start,
                end=current_end,
                granularity=coinbase_granularity
            )

            if result and "candles" in result:
                all_candles.extend(result["candles"])
                if verbose:
                    print(f"    Fetched {len(result['candles'])} candles ({current_start.date()} to {current_end.date()})")
        except Exception as e:
            logger.warning(f"Failed to fetch chunk {current_start} to {current_end}: {e}")

        current_start = current_end

    client.close()

    if not all_candles:
        raise ValueError(f"No data returned from Coinbase for {symbol}")

    report.original_rows = len(all_candles)

    # Convert to DataFrame
    df = pd.DataFrame(all_candles)

    # Coinbase returns: start (Unix timestamp), open, high, low, close, volume
    df["timestamp"] = pd.to_datetime(df["start"].astype(int), unit="s")
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    # Ensure numeric types
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort and clean
    df = sort_by_timestamp(df)
    df = remove_duplicates(df, report)
    df = fix_ohlc_consistency(df, report)

    # Resample to target interval if needed
    if interval.lower() in ["4h", "4H"]:
        if verbose:
            print(f"  Resampling from 1H to 4H...")
        df = resample_ohlcv(df, "4h", verbose=False)
    elif interval.lower() in ["1d", "1D"]:
        if verbose:
            print(f"  Resampling from 1H to 1D...")
        df = resample_ohlcv(df, "1d", verbose=False)

    # Validate
    validate_data(df, report)

    # Update report
    report.final_rows = len(df)
    if len(df) > 0:
        report.date_range = (df["timestamp"].iloc[0], df["timestamp"].iloc[-1])

    if verbose:
        print(f"  Downloaded: {report.original_rows} 1H bars")
        print(f"  Final: {report.final_rows} {interval} bars")
        print(f"  Date range: {report.date_range[0]} to {report.date_range[1]}")
        latest_price = df["close"].iloc[-1]
        print(f"  Latest price: ${latest_price:,.2f}")

    return df, report
