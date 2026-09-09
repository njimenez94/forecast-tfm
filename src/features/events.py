"""Features de eventos: distancia genérica/por-evento, codificación de tipo de
evento, lifecycle de release y detección de cierre de tienda (con el zero-out
de sales/gross_sales que ese cierre implica)."""
import polars as pl

_EVENT_TYPE_CODES = {"Sporting": 1, "Cultural": 2, "National": 3, "Religious": 4}

# Eventos sin cierre de tienda pero con caída de demanda clara y consistente,
# verificada empíricamente sobre data/processed/level_09_daily_store_dept.parquet:
# avg(sales | evento) de Thanksgiving es 301.86 (-39% vs 492.58 global, n=350) y de
# NewYear 368.12 (-25%). El resto de los ~30 eventos del calendario cae dentro de
# +-20% del promedio global, indistinguible de ruido a este nivel de agregación, así
# que no se generaliza a los 30 (evita columnas dispersas sin señal real).
_HIGH_IMPACT_EVENTS = ("Thanksgiving", "NewYear")

# Eventos con cierre total de tienda (venta ~0 independientemente de la demanda
# subyacente). Verificado empíricamente sobre data/processed: avg(sales | event)
# de Christmas es 0.22 (vs. 492.58 global, n=350), 3 órdenes de magnitud por debajo
# de cualquier otro evento (el segundo más bajo, Thanksgiving, promedia 301.86 --
# reduce demanda pero no cierra). Se modela aparte de event_name_1/event_type_1
# porque el efecto es un veto multiplicativo sobre la venta, no un desplazamiento
# aditivo.
_CLOSURE_EVENTS = {"Christmas"}

# Ventana (días) alrededor de un cierre en la que days_to_closure lleva valor; fuera
# de ella, NaN.
_CLOSURE_WINDOW = 7


def _event_distance_features(event_name: str) -> tuple[pl.Expr, pl.Expr]:
    """days_since_<evento>/days_to_<evento>: mismo patrón forward_fill/backward_fill
    que days_since_event/days_to_event, pero acotado a un único nombre de evento
    (event_name_1 o event_name_2), para las señales de _HIGH_IMPACT_EVENTS."""
    is_this_event = (
        (pl.col("event_name_1") == event_name) | (pl.col("event_name_2") == event_name)
    ).fill_null(False)
    event_date = pl.when(is_this_event).then(pl.col("date"))
    days_since = (pl.col("date") - event_date.forward_fill().over("series_id")) \
        .dt.total_days().cast(pl.Int32)
    days_to = (event_date.backward_fill().over("series_id") - pl.col("date")) \
        .dt.total_days().cast(pl.Int32)
    return days_since, days_to


def add_event_features(df: pl.DataFrame) -> pl.DataFrame:
    """days_since_release, distancia genérica a evento, distancia firmada a
    cierre de tienda (con zero-out de sales/gross_sales en los días de cierre
    detectado), codificación de tipo de evento y distancia a los eventos de
    _HIGH_IMPACT_EVENTS."""
    is_event = pl.col("has_event").cast(pl.Boolean) | pl.col("has_event_2").cast(pl.Boolean)
    event_date = pl.when(is_event).then(pl.col("date"))
    release_date = pl.when(pl.col("avg_sell_price").is_not_null()).then(pl.col("date"))
    # is_in() sobre null devuelve null (lógica de Kleene), no false: sin fill_null
    # los días sin evento (event_name_1/2 null) quedarían como null en vez de 0.
    is_closure_event = (
        pl.col("event_name_1").is_in(_CLOSURE_EVENTS).fill_null(False)
        | pl.col("event_name_2").is_in(_CLOSURE_EVENTS).fill_null(False)
    )
    # Cierre determinado solo por el calendario (is_closure_event): _CLOSURE_EVENTS
    # ya está restringido a fechas de cierre total verificadas empíricamente arriba
    # (Christmas, avg(sales)=0.22 -- ~0 en la práctica). No se usa 'sales' de la fila
    # ni una media de la serie para "confirmar" el cierre: eso miraría el propio
    # target (incluyendo fechas de valid/test) para construir un feature que además
    # pone ese mismo target a cero -- leakage tanto de información futura como del
    # target actual. Un cierre real es conocido de antemano por el calendario, no
    # inferido de la venta observada.
    is_store_closed = is_closure_event.cast(pl.Int8)

    # Distancia firmada al cierre más cercano (positiva = faltan N días, negativa =
    # pasaron N días, 0 el propio día de cierre), acotada a +-_CLOSURE_WINDOW y NaN
    # fuera de esa ventana. A diferencia de days_since_event/days_to_event (genéricos,
    # sin signo, uno por cada evento) esta es una única señal específica de cierres,
    # pensada para que el árbol distinga "día después del cierre" (rebote de demanda)
    # de un día normal sin depender de lag1, que en esos días vale 0 por el zero-out
    # de is_store_closed. Sin acotar, el rango real (+-180 días entre un Christmas y
    # el siguiente) diluye la señal entre cientos de valores similares en otras
    # temporadas y el árbol no la usa; NaN fuera de la ventana la deja casi siempre
    # nula salvo justo alrededor del cierre, mucho más fácil de explotar en un split.
    closure_date = pl.when(is_closure_event).then(pl.col("date"))
    days_since_closure = (pl.col("date") - closure_date.forward_fill().over("series_id")) \
        .dt.total_days().cast(pl.Int32)
    days_to_closure = (closure_date.backward_fill().over("series_id") - pl.col("date")) \
        .dt.total_days().cast(pl.Int32)
    signed_days_to_closure = (
        pl.when(days_to_closure.is_null()).then(-days_since_closure)
        .when(days_since_closure.is_null()).then(days_to_closure)
        .when(days_to_closure <= days_since_closure).then(days_to_closure)
        .otherwise(-days_since_closure)
    )
    signed_days_to_closure = (
        pl.when(signed_days_to_closure.abs() <= _CLOSURE_WINDOW)
        .then(signed_days_to_closure)
        .otherwise(None)
    )

    high_impact_exprs = {}
    for event_name in _HIGH_IMPACT_EVENTS:
        days_since, days_to = _event_distance_features(event_name)
        key = event_name.lower()
        high_impact_exprs[f"days_since_{key}"] = days_since.alias(f"days_since_{key}")
        high_impact_exprs[f"days_to_{key}"] = days_to.alias(f"days_to_{key}")

    # min().over("series_id") mira toda la serie (incluye fechas futuras), pero solo
    # es leakage en las filas *previas* al release (avg_sell_price aún null): ahí el
    # resultado sería "días que faltan para el release", una fecha futura que en
    # producción no se conoce con esa precisión. Para filas ya releasadas la fecha
    # es un hecho ya ocurrido (no depende de qué pase después), así que no hace falta
    # tocarlas -- solo se anula el pre-release con el propio release_date.is_null().
    days_since_release_signed = (pl.col("date") - release_date.min().over("series_id")).dt.total_days()
    days_since_release = (
        pl.when(days_since_release_signed >= 0).then(days_since_release_signed).otherwise(None)
    ).cast(pl.Int32)

    return df.with_columns(
        days_since_release.alias("days_since_release"),
        (pl.col("date") - event_date.forward_fill().over("series_id"))
            .dt.total_days().cast(pl.Int32).alias("days_since_event"),
        (event_date.backward_fill().over("series_id") - pl.col("date"))
            .dt.total_days().cast(pl.Int32).alias("days_to_event"),
        signed_days_to_closure.alias("days_to_closure"),
        pl.col("event_type_1").replace_strict(_EVENT_TYPE_CODES, default=0, return_dtype=pl.Int8)
            .alias("event_type_1_enc"),
        pl.col("event_type_2").replace_strict(_EVENT_TYPE_CODES, default=0, return_dtype=pl.Int8)
            .alias("event_type_2_enc"),
        is_store_closed.alias("is_store_closed"),
        *high_impact_exprs.values(),
    ).with_columns(
        pl.when(is_store_closed.cast(pl.Boolean)).then(0.0).otherwise(pl.col("sales"))
            .alias("sales"),
        pl.when(is_store_closed.cast(pl.Boolean)).then(0.0).otherwise(pl.col("gross_sales"))
            .alias("gross_sales"),
    )
