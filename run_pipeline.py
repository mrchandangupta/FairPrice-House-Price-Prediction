"""Runs the whole FairPrice workflow end to end, phase by phase.

    python run_pipeline.py           # full run (20 search iterations, 6 for the ANN; ~15 min)
    python run_pipeline.py --fast    # quick run for demos / debugging
    python run_pipeline.py --skip-eda
"""
import argparse
import json
from datetime import datetime

import joblib

from fairprice import config
from fairprice.data_ingestion import ingest
from fairprice.eda import run_eda
from fairprice.evaluate import (build_leaderboard, plot_ann_training_curve, plot_feature_importance,
                                plot_model_comparison, plot_predictions, regression_metrics, select_best)
from fairprice.explain import make_explainer, plot_shap_summary
from fairprice.logger import get_logger
from fairprice.predict import BACKGROUND_PATH
from fairprice.preprocessing import clean_data, save_clean_data
from fairprice.train import NEURAL, split_data, train_all

log = get_logger("pipeline")


def save_model(best: dict, ann: dict, X_train, test_metrics: dict, ann_test_metrics: dict,
               feature_importance) -> None:
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["pipeline"], config.MODEL_PATH)
    # The ANN is saved too (even when it isn't the best) so the app can show it side by side.
    joblib.dump(ann["pipeline"], config.ANN_MODEL_PATH)
    X_train.sample(200, random_state=config.RANDOM_STATE).to_csv(BACKGROUND_PATH, index=False)

    meta = {
        "model_name": best["name"],
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "best_params": best["best_params"],
        "cv_rmse_mean": best["cv_rmse_mean"],
        "cv_rmse_std": best["cv_rmse_std"],
        "test_metrics": test_metrics,
        "test_mae_dollars": round(test_metrics["mae"] * config.PRICE_UNIT),
        # used by the app to warn when a user enters values the model never saw
        "feature_ranges": {c: [float(X_train[c].min()), float(X_train[c].max())] for c in config.FEATURES},
        "feature_importance": feature_importance.round(4).to_dict(),
        "price_unit": config.PRICE_UNIT,
        "ann": {
            "best_params": ann["best_params"],
            "cv_rmse_mean": ann["cv_rmse_mean"],
            "cv_rmse_std": ann["cv_rmse_std"],
            "epochs_trained": ann["pipeline"].named_steps["model"].n_epochs_,
            "test_metrics": ann_test_metrics,
            "test_mae_dollars": round(ann_test_metrics["mae"] * config.PRICE_UNIT),
        },
    }
    config.MODEL_META_PATH.write_text(json.dumps(meta, indent=2, default=str))
    log.info("Saved best model -> %s and ANN -> %s", config.MODEL_PATH.relative_to(config.ROOT_DIR),
             config.ANN_MODEL_PATH.relative_to(config.ROOT_DIR))


def main(fast: bool = False, skip_eda: bool = False) -> None:
    log.info("=== Phase 1: Data ingestion & schema validation ===")
    raw_df = ingest()

    log.info("=== Phase 2: Data cleaning ===")
    clean_df, cleaning_report = clean_data(raw_df)
    save_clean_data(clean_df)

    eda_summary = {}
    if not skip_eda:
        log.info("=== Phase 3: Exploratory data analysis ===")
        eda_summary = run_eda(raw_df, clean_df)

    log.info("=== Phase 4-6: Split, feature pipeline, training, tuning, K-fold CV ===")
    X_train, X_test, y_train, y_test = split_data(clean_df)
    results = train_all(X_train, y_train, fast=fast)

    log.info("=== Phase 7: Evaluation & model selection ===")
    leaderboard = build_leaderboard(results, X_train, y_train, X_test, y_test, save=True)
    best = select_best(results, leaderboard)
    y_pred = best["pipeline"].predict(X_test)
    test_metrics = regression_metrics(y_test, y_pred)
    plot_model_comparison(leaderboard)
    plot_predictions(best["name"], y_test, y_pred)
    importance = plot_feature_importance(best["name"], best["pipeline"], X_test, y_test)

    ann = next(r for r in results if r["family"] == NEURAL)
    ann_test_metrics = regression_metrics(y_test, ann["pipeline"].predict(X_test))
    plot_ann_training_curve(ann["pipeline"])
    log.info("ANN test R2 %.3f vs best conventional model R2 %.3f", ann_test_metrics["r2"], test_metrics["r2"])

    log.info("=== Phase 7b: Explainability (SHAP) ===")
    explainer = make_explainer(best["pipeline"], X_train.sample(200, random_state=config.RANDOM_STATE))
    plot_shap_summary(best["pipeline"], explainer, X_test.sample(1000, random_state=config.RANDOM_STATE))

    log.info("=== Phase 8: Save model & reports ===")
    save_model(best, ann, X_train, test_metrics, ann_test_metrics, importance)
    config.METRICS_PATH.write_text(json.dumps({
        "cleaning": cleaning_report,
        "eda": eda_summary,
        "leaderboard": leaderboard.to_dict(orient="records"),
        "best_model": best["name"],
        "best_params": best["best_params"],
        "test_metrics": test_metrics,
        "ann_test_metrics": ann_test_metrics,
    }, indent=2, default=str))
    log.info("Done. Best = %s | test RMSE %.3f | MAE $%s | R2 %.3f", best["name"], test_metrics["rmse"],
             f"{test_metrics['mae'] * config.PRICE_UNIT:,.0f}", test_metrics["r2"])
    log.info("Next: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FairPrice training pipeline")
    parser.add_argument("--fast", action="store_true", help="fewer tuning iterations")
    parser.add_argument("--skip-eda", action="store_true", help="skip EDA figure generation")
    args = parser.parse_args()
    main(fast=args.fast, skip_eda=args.skip_eda)
