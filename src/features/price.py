"""Features de precio: nivel relativo al máximo histórico y volatilidad reciente."""
import polars as pl


def add_price_features(df: pl.DataFrame) -> pl.DataFrame:
    """price_vs_max (precio actual / máximo histórico de la serie),
    price_volatility (desviación estándar móvil de 90 días -- el nivel ya lo
    da price_vs_max/price_vs_mean; esto captura si el precio ha estado
    fluctuando -- promociones frecuentes -- vs. estable) y
    days_since_price_change (mismo patrón forward_fill que days_since_event,
    aplicado al último cambio de precio de la serie)."""
    running_max_price = pl.col("avg_sell_price").cum_max().over("series_id")
    price_volatility = (
        pl.col("avg_sell_price").rolling_std(window_size=90, min_samples=2).over("series_id")
    )
    price_changed = (
        pl.col("avg_sell_price") != pl.col("avg_sell_price").shift(1).over("series_id")
    ).fill_null(False)
    price_change_date = pl.when(price_changed).then(pl.col("date"))
    days_since_price_change = (
        pl.col("date") - price_change_date.forward_fill().over("series_id")
    ).dt.total_days().cast(pl.Int32)

    return df.with_columns(
        (pl.col("avg_sell_price") / running_max_price)
            .replace([float("inf"), float("-inf")], None)
            .cast(pl.Float32)
            .alias("price_vs_max"),
        price_volatility.cast(pl.Float32).alias("price_volatility"),
        days_since_price_change.alias("days_since_price_change"),
    )
