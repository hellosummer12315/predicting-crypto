import numpy as np
import pandas as pd

def generate_mock_data(days: int, coin: str = "BTC", seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days)
    base_prices = 13000 * np.exp(np.cumsum(rng.normal(0.001, 0.02, days)))
    
    return pd.DataFrame({
        'coin': coin,
        'open': base_prices * rng.uniform(0.998, 1.002, days),
        'high': base_prices * rng.uniform(1.002, 1.010, days),
        'low': base_prices * rng.uniform(0.990, 0.998, days),
        'close': base_prices,
        'volume': rng.lognormal(8, 1, days)
    }, index=dates).sort_index()
