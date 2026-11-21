"""Phase 7 — Model evaluation: metrics, leaderboard, model selection and diagnostic plots.

The best model is chosen on **cross-validation RMSE**, not on the test set — the test set
is only used once, at the end, to report an honest final score.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from fairprice import config
from fairprice.logger import get_logger

log = get_logger(__name__)


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_leaderboard(results: list[dict], X_train, y_train, X_test, y_test, save: bool = False) -> pd.DataFrame:
    """Score every model; only the official pipeline run saves it (so notebook experiments don't overwrite it)."""
    rows = []
    for res in results:
        train_m = regression_metrics(y_train, res["pipeline"].predict(X_train))
        test_m = regression_metrics(y_test, res["pipeline"].predict(X_test))
        rows.append({
            "model": res["name"],
            "family": res.get("family", ""),
            "cv_rmse_mean": res["cv_rmse_mean"],
            "cv_rmse_std": res["cv_rmse_std"],
            "train_rmse": train_m["rmse"],
            "test_rmse": test_m["rmse"],
            "test_mae": test_m["mae"],
            "test_r2": test_m["r2"],
            "test_mae_dollars": round(test_m["mae"] * config.PRICE_UNIT),
            # large gap between train and test error = overfitting
            "overfit_gap": test_m["rmse"] - train_m["rmse"],
            "fit_seconds": res["fit_seconds"],
        })
    board = pd.DataFrame(rows).sort_values("cv_rmse_mean").reset_index(drop=True).round(4)
    if save:
        config.LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
        board.to_csv(config.LEADERBOARD_PATH, index=False)
    log.info("Leaderboard (sorted by CV RMSE):\n%s", board.to_string(index=False))
    return board


def select_best(results: list[dict], leaderboard: pd.DataFrame) -> dict:
    best_name = leaderboard.iloc[0]["model"]
    log.info("Best model by CV RMSE: %s", best_name)
    return next(r for r in results if r["name"] == best_name)


def _save(fig, name: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / name, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(board: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    melted = board.melt(id_vars="model", value_vars=["cv_rmse_mean", "test_rmse"],
                        var_name="metric", value_name="RMSE")
    sns.barplot(data=melted, x="model", y="RMSE", hue="metric", ax=ax, palette="Set2")
    ax.set_title("Model comparison (lower RMSE is better, $100k units)")
    _save(fig, "07_model_comparison.png")


def plot_predictions(name: str, y_test, y_pred) -> None:
    residuals = np.asarray(y_test) - y_pred
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    axes[0].scatter(y_test, y_pred, alpha=0.25, s=8)
    lims = [0, config.TARGET_CAP]
    axes[0].plot(lims, lims, "r--", label="perfect prediction")
    axes[0].set(title=f"{name}: predicted vs actual", xlabel="Actual ($100k)", ylabel="Predicted ($100k)")
    axes[0].legend()

    axes[1].scatter(y_pred, residuals, alpha=0.25, s=8, color="#7b4f9d")
    axes[1].axhline(0, color="red", ls="--")
    axes[1].set(title="Residuals vs predicted (should be random around 0)",
                xlabel="Predicted ($100k)", ylabel="Residual")

    sns.histplot(residuals, bins=60, ax=axes[2], color="#7b4f9d")
    axes[2].set(title="Residual distribution", xlabel="Residual ($100k)")
    fig.tight_layout()
    _save(fig, "08_residual_analysis.png")


def plot_ann_training_curve(ann_pipeline) -> None:
    """Training vs validation loss per epoch: shows learning progress and where early stopping kicked in."""
    model = ann_pipeline.named_steps["model"]
    history = model.history_
    best_epoch = int(np.argmin(history["val_loss"])) + 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, metric, title in [(axes[0], "loss", "MSE loss"), (axes[1], "mae", "MAE ($100k)")]:
        ax.plot(history[metric], label="train")
        ax.plot(history[f"val_{metric}"], label="validation")
        ax.axvline(best_epoch - 1, color="grey", ls="--", label=f"best epoch ({best_epoch})")
        ax.set(title=f"ANN {title} per epoch", xlabel="Epoch")
        ax.legend()
    fig.suptitle(f"ANN layers {model.hidden_layers}, {model.activation}, {model.optimizer}, "
                 f"dropout {model.dropout} - stopped after {model.n_epochs_} epochs", fontsize=11)
    fig.tight_layout()
    _save(fig, "11_ann_training_curve.png")


def plot_feature_importance(name: str, pipeline, X=None, y=None) -> pd.Series:
    model = pipeline.named_steps["model"]
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=names)
        label = "Importance (tree split gain)"
    elif hasattr(model, "coef_"):  # linear model: coefficient size on standardized features
        importance = pd.Series(np.abs(model.coef_), index=names)
        label = "|coefficient| (standardized features)"
    else:  # e.g. the ANN: how much RMSE worsens when a feature is shuffled
        processed = pipeline.named_steps["preprocess"].transform(X)
        result = permutation_importance(model, processed, y, n_repeats=5,
                                        random_state=config.RANDOM_STATE,
                                        scoring="neg_root_mean_squared_error")
        importance = pd.Series(result.importances_mean, index=names)
        label = "Permutation importance (RMSE increase)"
    importance = importance.sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    importance.plot.barh(ax=ax, color="#3b6ea5")
    ax.set(title=f"Feature importance — {name}", xlabel=label)
    _save(fig, "09_feature_importance.png")
    return importance.sort_values(ascending=False)
