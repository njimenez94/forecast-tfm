import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.dates as mdates

def plot_forecast(df_pred, target_col, series_id, n=30, date=None):
    """Serie real vs. predicha para `series_id`; si se pasa `date`, recorta +-n días
    alrededor y marca la fecha con una línea vertical."""
    df_plot = df_pred[df_pred["series_id"] == series_id].copy()
    if date:
        date = pd.Timestamp(date)
        since = date - pd.Timedelta(days=n)
        until = date + pd.Timedelta(days=n)
        df_plot = df_plot[(df_plot["date"] >= since) & (df_plot["date"] <= until)]

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.lineplot(df_plot, x="date", y=target_col, label="Real", marker="o", ax=ax)
    sns.lineplot(df_plot, x="date", y="y_pred", label="Predicción", marker="o", ax=ax)
    if date:
        ax.axvline(date, color="red", linestyle="--", linewidth=1)

    ax.set_title(series_id, loc="left")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%y"))
    plt.xticks(rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.ylim(0)
    sns.despine()
    plt.tight_layout()
    plt.show()


def explain_prediction(test_df, X_test, df_pred, explainer, target_col, series_id, date, max_display=10):
    """Waterfall de SHAP para la fila (series_id, date), con el detalle real/pred/bias
    en el título. Requiere un `explainer` de shap ya construido sobre el modelo."""
    import shap

    row_mask = (test_df["series_id"] == series_id) & (test_df["date"] == date)
    row_idx = test_df.index[row_mask][0]

    row = X_test.loc[[row_idx]]
    row_shap_values = explainer.shap_values(row)

    info = df_pred.loc[row_idx, ["series_id", "date", target_col, "y_pred", "error"]]

    series_id_val = info["series_id"]
    date_val = info["date"]
    sales_val = info[target_col]
    pred_val = info["y_pred"]
    bias_val = -info["error"]  # pred - actual (positivo = sobreestima)
    bias_pct = (bias_val / sales_val * 100) if sales_val != 0 else float("nan")

    shap.plots.waterfall(
        shap.Explanation(
            values=row_shap_values[0],
            base_values=explainer.expected_value,
            data=row.iloc[0],
            feature_names=row.columns,
        ),
        max_display=max_display,
        show=False,
    )
    plt.gcf().set_size_inches(10, plt.gcf().get_size_inches()[1])  # más ancho para que no se pisen los labels
    plt.title(
        f"{series_id_val} | {date_val:%Y-%m-%d} | "
        f"sales={sales_val:.0f} | pred={pred_val:.0f} | "
        f"bias={bias_val:.0f} ({bias_pct:.1f}%)",
        pad=20,
        loc="left",
    )
    plt.show()


def analizar_prediccion(test_df, X_test, df_pred, target_col, series_id, date=None, max_display=10, explainer=None):
    """Grafica la serie completa (`plot_forecast`) y, si se pasa `date`, el waterfall
    de SHAP para ese día (`explain_prediction`, requiere `explainer`)."""
    plot_forecast(df_pred, target_col, series_id, date=date)
    if date:
        if explainer is None:
            raise ValueError("se necesita `explainer` para explicar una fecha puntual")
        explain_prediction(test_df, X_test, df_pred, explainer, target_col, series_id, date, max_display=max_display)
