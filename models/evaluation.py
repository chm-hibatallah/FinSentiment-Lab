"""
Evaluation metrics for all FinSentiment predictive models.

Metrics computed
----------------
Classification metrics (direction prediction)
  accuracy          : fraction of correct direction calls
  f1_score          : harmonic mean of precision and recall
  auc_roc           : area under the ROC curve (1.0 = perfect, 0.5 = random)
  precision         : of all "up" predictions, how many were correct?
  recall            : of all actual "up" days, how many did we catch?

Financial metrics (signal quality)
  hit_rate          : same as accuracy but named for financial convention
  sharpe_ratio      : annualised Sharpe of a long/short strategy driven by
                      the model's predictions
                      long  when model predicts up   (+1)
                      short when model predicts down (-1)
  max_drawdown      : worst peak-to-trough equity curve decline (%)
  cumulative_return : total strategy return over the test period

Regression metrics (return magnitude)
  mae               : mean absolute error
  rmse              : root mean squared error
  r_squared         : coefficient of determination
  directional_accuracy : sign(pred) == sign(actual) — direction from regressor

Output
------
ModelEvaluation dataclass containing all metrics + an interpretation string.
EvaluationReport containing results for all models side by side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.logger import get_logger

log = get_logger(__name__)

TRADING_DAYS_PER_YEAR = 252


@dataclass
class ModelEvaluation:
    """All metrics for one model on one ticker."""
    model_name:   str
    ticker:       str
    task:         str          # "classification" | "regression"
    n_test:       int

    # Classification
    accuracy:     Optional[float] = None
    f1:           Optional[float] = None
    auc_roc:      Optional[float] = None
    precision:    Optional[float] = None
    recall:       Optional[float] = None

    # Financial
    hit_rate:     Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    cum_return:   Optional[float] = None

    # Regression
    mae:          Optional[float] = None
    rmse:         Optional[float] = None
    r_squared:    Optional[float] = None
    dir_accuracy: Optional[float] = None

    interpretation: str = ""


class ModelEvaluator:
    """
    Computes all evaluation metrics for a fitted model.

    Usage
    -----
        evaluator = ModelEvaluator()

        # Classification
        result = evaluator.evaluate_classifier(
            model_name="XGBoost",
            ticker="AAPL",
            y_true=y_test,
            y_pred=model.predict(X_test),
            y_proba=model.predict_proba(X_test),
            actual_returns=test_returns,   # for Sharpe calculation
        )

        # Regression
        result = evaluator.evaluate_regressor(
            model_name="XGBoostReg",
            ticker="AAPL",
            y_true=y_test,
            y_pred=model.predict(X_test),
        )
    """

    # ------------------------------------------------------------------
    # Classification evaluation
    # ------------------------------------------------------------------

    def evaluate_classifier(
        self,
        model_name:      str,
        ticker:          str,
        y_true:          np.ndarray,
        y_pred:          np.ndarray,
        y_proba:         Optional[np.ndarray] = None,
        actual_returns:  Optional[np.ndarray] = None,
    ) -> ModelEvaluation:
        """Compute all classification + financial metrics."""
        from sklearn.metrics import (
            accuracy_score, f1_score, roc_auc_score,
            precision_score, recall_score,
        )

        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)

        accuracy  = float(accuracy_score(y_true, y_pred))
        f1        = float(f1_score(y_true, y_pred, zero_division=0))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall    = float(recall_score(y_true, y_pred, zero_division=0))

        auc_roc = None
        if y_proba is not None and len(np.unique(y_true)) > 1:
            try:
                auc_roc = float(roc_auc_score(y_true, y_proba))
            except Exception:
                pass

        # Financial metrics
        sharpe, max_dd, cum_ret = None, None, None
        if actual_returns is not None and len(actual_returns) == len(y_pred):
            sharpe, max_dd, cum_ret = self._financial_metrics(y_pred, actual_returns)

        result = ModelEvaluation(
            model_name  = model_name,
            ticker      = ticker,
            task        = "classification",
            n_test      = len(y_true),
            accuracy    = round(accuracy, 4),
            f1          = round(f1, 4),
            auc_roc     = round(auc_roc, 4) if auc_roc else None,
            precision   = round(precision, 4),
            recall      = round(recall, 4),
            hit_rate    = round(accuracy, 4),
            sharpe_ratio= round(sharpe, 4)   if sharpe  is not None else None,
            max_drawdown= round(max_dd, 4)   if max_dd  is not None else None,
            cum_return  = round(cum_ret, 4)  if cum_ret is not None else None,
        )
        result.interpretation = self._interpret_classifier(result)
        self._log_result(result)
        return result

    # ------------------------------------------------------------------
    # Regression evaluation
    # ------------------------------------------------------------------

    def evaluate_regressor(
        self,
        model_name:     str,
        ticker:         str,
        y_true:         np.ndarray,
        y_pred:         np.ndarray,
        actual_returns: Optional[np.ndarray] = None,
    ) -> ModelEvaluation:
        """Compute regression + directional accuracy metrics."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mae       = float(mean_absolute_error(y_true, y_pred))
        rmse      = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2        = float(r2_score(y_true, y_pred))
        dir_acc   = float(np.mean(np.sign(y_pred) == np.sign(y_true)))

        # Financial Sharpe using sign of regression output as signal
        sharpe, max_dd, cum_ret = None, None, None
        ret_source = actual_returns if actual_returns is not None else y_true
        signal = (np.sign(y_pred) > 0).astype(int)
        if len(signal) == len(ret_source):
            sharpe, max_dd, cum_ret = self._financial_metrics(signal, ret_source)

        result = ModelEvaluation(
            model_name   = model_name,
            ticker       = ticker,
            task         = "regression",
            n_test       = len(y_true),
            mae          = round(mae,     6),
            rmse         = round(rmse,    6),
            r_squared    = round(r2,      4),
            dir_accuracy = round(dir_acc, 4),
            hit_rate     = round(dir_acc, 4),
            sharpe_ratio = round(sharpe,  4) if sharpe  is not None else None,
            max_drawdown = round(max_dd,  4) if max_dd  is not None else None,
            cum_return   = round(cum_ret, 4) if cum_ret is not None else None,
        )
        result.interpretation = self._interpret_regressor(result)
        self._log_result(result)
        return result

    # ------------------------------------------------------------------
    # Financial metrics
    # ------------------------------------------------------------------

    def _financial_metrics(
        self,
        y_pred:    np.ndarray,    # binary: 1=long, 0=short
        returns:   np.ndarray,    # actual daily log returns
    ) -> tuple[float, float, float]:
        """
        Compute Sharpe ratio, max drawdown, and cumulative return
        of a long/short strategy driven by y_pred.

        Signal:
          y_pred == 1 → long  (+1 × return)
          y_pred == 0 → short (-1 × return)
        """
        signal       = np.where(y_pred == 1, 1, -1).astype(float)
        strat_returns = signal * returns

        # Sharpe (annualised)
        daily_mean = np.mean(strat_returns)
        daily_std  = np.std(strat_returns) + 1e-10
        sharpe     = (daily_mean / daily_std) * np.sqrt(TRADING_DAYS_PER_YEAR)

        # Cumulative return
        cum_ret = float(np.expm1(np.sum(strat_returns)))

        # Max drawdown
        equity    = np.exp(np.cumsum(strat_returns))
        rolling_max = np.maximum.accumulate(equity)
        drawdowns   = (equity - rolling_max) / rolling_max
        max_dd      = float(np.min(drawdowns))

        return float(sharpe), float(max_dd), float(cum_ret)

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------

    def _interpret_classifier(self, r: ModelEvaluation) -> str:
        parts = []
        if r.auc_roc is not None:
            if r.auc_roc > 0.60:
                parts.append(f"AUC={r.auc_roc:.3f} — meaningful discrimination above random (0.5)")
            elif r.auc_roc > 0.55:
                parts.append(f"AUC={r.auc_roc:.3f} — weak but above random")
            else:
                parts.append(f"AUC={r.auc_roc:.3f} — near random, model has no predictive power")

        if r.hit_rate is not None:
            if r.hit_rate > 0.55:
                parts.append(f"hit_rate={r.hit_rate:.1%} — above 50% random baseline")
            else:
                parts.append(f"hit_rate={r.hit_rate:.1%} — at or below random baseline")

        if r.sharpe_ratio is not None:
            if r.sharpe_ratio > 1.0:
                parts.append(f"Sharpe={r.sharpe_ratio:.2f} — strong signal quality")
            elif r.sharpe_ratio > 0.5:
                parts.append(f"Sharpe={r.sharpe_ratio:.2f} — moderate signal quality")
            else:
                parts.append(f"Sharpe={r.sharpe_ratio:.2f} — weak signal, high noise")

        return f"{r.ticker} {r.model_name}: " + " | ".join(parts)

    def _interpret_regressor(self, r: ModelEvaluation) -> str:
        parts = []
        if r.r_squared is not None:
            parts.append(f"R²={r.r_squared:.4f}")
        if r.dir_accuracy is not None:
            parts.append(f"directional_acc={r.dir_accuracy:.1%}")
        if r.sharpe_ratio is not None:
            parts.append(f"Sharpe={r.sharpe_ratio:.2f}")
        return f"{r.ticker} {r.model_name}: " + " | ".join(parts)

    def _log_result(self, r: ModelEvaluation):
        if r.task == "classification":
            log.info(
                "  %s | %s | acc=%.3f | AUC=%s | Sharpe=%s",
                r.ticker, r.model_name, r.accuracy or 0,
                f"{r.auc_roc:.3f}" if r.auc_roc else "N/A",
                f"{r.sharpe_ratio:.2f}" if r.sharpe_ratio else "N/A",
            )
        else:
            log.info(
                "  %s | %s | R²=%.4f | dir_acc=%.3f | Sharpe=%s",
                r.ticker, r.model_name, r.r_squared or 0,
                r.dir_accuracy or 0,
                f"{r.sharpe_ratio:.2f}" if r.sharpe_ratio else "N/A",
            )

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------

    def comparison_table(self, results: List[ModelEvaluation]) -> pd.DataFrame:
        """
        Side-by-side comparison of all models.
        Sorted by AUC-ROC (classifiers) or Sharpe ratio.
        """
        rows = []
        for r in results:
            rows.append({
                "ticker":      r.ticker,
                "model":       r.model_name,
                "task":        r.task,
                "n_test":      r.n_test,
                "accuracy":    r.accuracy,
                "f1":          r.f1,
                "auc_roc":     r.auc_roc,
                "hit_rate":    r.hit_rate,
                "sharpe":      r.sharpe_ratio,
                "max_drawdown":r.max_drawdown,
                "cum_return":  r.cum_return,
                "r_squared":   r.r_squared,
                "dir_accuracy":r.dir_accuracy,
            })
        df = pd.DataFrame(rows)
        if "auc_roc" in df.columns and df["auc_roc"].notna().any():
            df = df.sort_values("auc_roc", ascending=False, na_position="last")
        return df.reset_index(drop=True)