import numpy as np
import pandas as pd
from typing import Optional

def generate_mock_data(
    days: int = 365,
    coin: str = "BTC",
    base_price: float = 10000.0,
    seed: Optional[int] = 42
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)
    
    # datetime index
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days)
    
    # geometric Brownian motion for price series
    daily_returns = rng.normal(0.001, 0.02, days)  # ~0.1% daily return, 2% volatility
    price_series = base_price * np.exp(np.cumsum(daily_returns))
    
    # construct OHLCV dataframe
    df = pd.DataFrame({
        'coin': coin,
        'open': price_series * rng.uniform(0.998, 1.002, days),  # Minor open price fluctuation
        'high': price_series * rng.uniform(1.002, 1.010, days),  # Intraday high
        'low': price_series * rng.uniform(0.990, 0.998, days),   # Intraday low
        'close': price_series,                                   # Closing price
        'volume': rng.lognormal(8, 1, days)                      # Log-normal distributed volume
    }, index=dates)
    
    # add market microstructure features
    # - Weekend effect (lower volume)
    df.loc[df.index.dayofweek >= 5, 'volume'] *= 0.7
    
    # jumps events
    jump_days = rng.choice(days, size=int(days*0.05), replace=False)
    df.iloc[jump_days, df.columns.get_loc('close')] *= rng.uniform(0.9, 1.1, len(jump_days))
    
    return df.sort_index()
