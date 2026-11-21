"""Phase 3 — Exploratory Data Analysis.

Every plot answers a question that drives a design decision:
  target distribution -> is the target capped/skewed?      (drop capped rows)
  feature histograms  -> which features are skewed?          (clip outliers)
  box plots           -> how extreme are the outliers?       (percentile clipping)
  correlation heatmap -> which features are redundant?       (replace AveBedrms)
  geographic map      -> does location matter?               (DistToCity feature)
"""
import matplotlib

matplotlib.use("Agg")  # render to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from fairprice import config
from fairprice.features import FeatureEngineer, build_preprocessor, correlation_report
from fairprice.logger import get_logger

log = get_logger(__name__)
sns.set_theme(style="whitegrid")


def _save(fig, name: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / name, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_target_distribution(raw_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(raw_df[config.TARGET], bins=60, ax=ax, color="#3b6ea5")
    ax.axvline(config.TARGET_CAP, color="crimson", ls="--", label="$500k cap (censored)")
    ax.set(title="Target: median house value ($100k)", xlabel="MedHouseVal")
    ax.legend()
    _save(fig, "01_target_distribution.png")


def plot_feature_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.flat, config.FEATURES):
        sns.histplot(df[col], bins=50, ax=ax, color="#5a9e6f")
        ax.set_title(f"{col} (skew={df[col].skew():.1f})")
    fig.suptitle("Feature distributions", fontsize=14)
    fig.tight_layout()
    _save(fig, "02_feature_distributions.png")


def plot_outliers(df: pd.DataFrame) -> None:
    cols = ["AveRooms", "AveBedrms", "AveOccup", "Population"]
    fig, axes = plt.subplots(1, len(cols), figsize=(16, 4))
    for ax, col in zip(axes, cols):
        sns.boxplot(y=df[col], ax=ax, color="#e0a458")
        ax.set_title(col)
    fig.suptitle("Outliers in raw features (handled by percentile clipping)", fontsize=13)
    fig.tight_layout()
    _save(fig, "03_outliers_boxplot.png")


def plot_correlation_heatmap(df: pd.DataFrame, name: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title(title)
    _save(fig, name)


def plot_geography(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    points = ax.scatter(df["Longitude"], df["Latitude"], c=df[config.TARGET], cmap="viridis",
                        s=df["Population"] / 400, alpha=0.5)
    for city, (lat, lon) in config.CITY_COORDS.items():
        ax.scatter(lon, lat, marker="*", s=200, color="red")
        ax.annotate(city, (lon, lat), xytext=(5, 5), textcoords="offset points", fontsize=9)
    fig.colorbar(points, label="MedHouseVal ($100k)")
    ax.set(title="House value by location (size = population)", xlabel="Longitude", ylabel="Latitude")
    _save(fig, "05_geographic_value_map.png")


def run_eda(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> dict:
    """Create all EDA figures and return the numbers that justify design decisions."""
    plot_target_distribution(raw_df)
    plot_feature_distributions(raw_df)
    plot_outliers(raw_df)
    plot_correlation_heatmap(raw_df, "04_correlation_heatmap_raw.png", "Correlation (raw features)")
    plot_geography(clean_df)

    engineered = FeatureEngineer().transform(clean_df[config.FEATURES])
    engineered[config.TARGET] = clean_df[config.TARGET].values
    plot_correlation_heatmap(engineered, "06_correlation_heatmap_engineered.png",
                             "Correlation (after feature engineering)")

    # Same features after imputation + clipping: shows how much outliers distorted correlations.
    processed = build_preprocessor().fit_transform(clean_df[config.FEATURES])
    processed_target_corr = processed.corrwith(clean_df[config.TARGET].set_axis(processed.index))

    summary = {
        "skewness": raw_df[config.FEATURES].skew().round(2).to_dict(),
        "target_correlation_raw": raw_df.corr()[config.TARGET].drop(config.TARGET).round(3).to_dict(),
        "target_correlation_processed": processed_target_corr.round(3).to_dict(),
        "high_correlation_pairs_raw": correlation_report(raw_df[config.FEATURES]),
        "high_correlation_pairs_engineered": correlation_report(engineered.drop(columns=config.TARGET)),
        "high_correlation_pairs_after_clipping": correlation_report(processed),
    }
    log.info("EDA figures saved to %s", config.FIGURES_DIR.relative_to(config.ROOT_DIR))
    log.info("Highly correlated raw pairs: %s", summary["high_correlation_pairs_raw"])
    return summary
