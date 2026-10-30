from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # Allows plots to be generated without opening a desktop window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "student-mat.csv"
MODEL_PATH = BASE_DIR / "student_performance_model.pkl"
DATABASE_PATH = BASE_DIR / "prediction_history.db"
GENERATED_DIR = BASE_DIR / "static" / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["studytime", "absences", "G1", "G2"]
TARGET = "G3"

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "local-student-prediction-demo-change-me")

# These variables are filled by setup_resources(). The website can still open if a
# file is missing so the page can tell the user what needs to be fixed.
df: pd.DataFrame | None = None
model = None
resource_errors: list[str] = []
comparison_rows: list[dict] = []
summary: dict = {}
optimization_summary: dict = {}


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database() -> None:
    with get_db_connection() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                studytime INTEGER NOT NULL,
                absences INTEGER NOT NULL,
                g1 INTEGER NOT NULL,
                g2 INTEGER NOT NULL,
                predicted_grade REAL NOT NULL
            )"""
        )


def generate_charts(data: pd.DataFrame, actual: pd.Series | None = None,
                    predicted: np.ndarray | None = None) -> None:
    """Create chart PNGs locally so the dashboard does not require a chart CDN."""
    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})

    # Final-grade distribution.
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.hist(data[TARGET].dropna(), bins=np.arange(-0.5, 21.5, 1),
            color="#536dfe", edgecolor="white")
    ax.set_title("Distribution of Final Grades")
    ax.set_xlabel("Final grade (G3) out of 20")
    ax.set_ylabel("Number of students")
    ax.set_xlim(-0.5, 20.5)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(GENERATED_DIR / "grade_distribution.png", dpi=130)
    plt.close(fig)

    # Average grade by study-time category.
    avg_by_study = data.groupby("studytime", as_index=False)[TARGET].mean()
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.bar(avg_by_study["studytime"].astype(str), avg_by_study[TARGET], color="#11a683", width=0.6)
    ax.set_title("Average Final Grade by Study-Time Category")
    ax.set_xlabel("Study-time category (1 = lowest, 4 = highest)")
    ax.set_ylabel("Average final grade")
    ax.set_ylim(0, 20)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(GENERATED_DIR / "study_time_grade.png", dpi=130)
    plt.close(fig)

    # Actual vs predicted, only if the model evaluation was successful.
    if actual is not None and predicted is not None:
        fig, ax = plt.subplots(figsize=(8.0, 3.5))
        ax.scatter(actual, predicted, alpha=0.72, color="#536dfe", edgecolors="white", linewidths=0.5)
        ax.plot([0, 20], [0, 20], linestyle="--", color="#ee6b6e", label="Perfect prediction")
        ax.set_title("Actual vs Predicted Grades — Random Forest")
        ax.set_xlabel("Actual grade")
        ax.set_ylabel("Predicted grade")
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(GENERATED_DIR / "actual_vs_predicted.png", dpi=130)
        plt.close(fig)


def setup_resources() -> None:
    global df, model, resource_errors, comparison_rows, summary, optimization_summary
    resource_errors = []
    comparison_rows = []
    summary = {}
    optimization_summary = {}

    if not DATASET_PATH.exists():
        resource_errors.append(
            f"Dataset not found: {DATASET_PATH.name}. Copy student-mat.csv into the same folder as app.py."
        )
    else:
        try:
            loaded_df = pd.read_csv(DATASET_PATH, sep=";")
            required = set(FEATURES + [TARGET])
            missing = required.difference(loaded_df.columns)
            if missing:
                raise ValueError("Expected columns are missing: " + ", ".join(sorted(missing)))
            original_rows = len(loaded_df)
            cleaned_df = loaded_df.copy()
            for column in FEATURES + [TARGET]:
                cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors="coerce")
            cleaned_df = cleaned_df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
            if cleaned_df.empty:
                raise ValueError("No complete numeric rows remain after preprocessing.")
            df = cleaned_df
            summary = {
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "original_rows": int(original_rows),
                "removed_rows": int(original_rows - len(df)),
                "average_grade": round(float(df[TARGET].mean()), 2),
                "pass_rate": round(float((df[TARGET] >= 10).mean() * 100), 1),
                "mean_absences": round(float(df["absences"].mean()), 1),
            }
        except Exception as exc:
            df = None
            resource_errors.append(f"Could not read {DATASET_PATH.name}: {exc}")

    if not MODEL_PATH.exists():
        resource_errors.append(
            f"Trained model not found: {MODEL_PATH.name}. Copy your saved model into the same folder as app.py."
        )
    else:
        try:
            model = joblib.load(MODEL_PATH)
        except Exception as exc:
            model = None
            resource_errors.append(
                "Could not load the saved model. This can happen when scikit-learn versions differ. "
                f"Details: {exc}"
            )

    if df is not None:
        # Re-create the same reproducible 80:20 split used in the notebook. The saved
        # Random Forest model is evaluated on the test rows; Linear Regression is fit
        # on the corresponding training rows for a like-for-like baseline comparison.
        try:
            X = df[FEATURES]
            y = df[TARGET]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            rf_pred = None
            if model is not None:
                rf_pred = model.predict(X_test[FEATURES])
                rf_mae = mean_absolute_error(y_test, rf_pred)
                rf_mse = mean_squared_error(y_test, rf_pred)
                rf_rmse = float(np.sqrt(rf_mse))
                rf_r2 = r2_score(y_test, rf_pred)
                comparison_rows.append({
                    "algorithm": "Random Forest",
                    "mae": round(float(rf_mae), 3),
                    "mse": round(float(rf_mse), 3),
                    "rmse": round(rf_rmse, 3),
                    "r2": round(float(rf_r2), 3),
                })
                generate_charts(df, y_test, rf_pred)
            else:
                generate_charts(df)

            linear = LinearRegression()
            linear.fit(X_train, y_train)
            linear_pred = linear.predict(X_test)
            comparison_rows.append({
                "algorithm": "Linear Regression",
                "mae": round(float(mean_absolute_error(y_test, linear_pred)), 3),
                "mse": round(float(mean_squared_error(y_test, linear_pred)), 3),
                "rmse": round(float(np.sqrt(mean_squared_error(y_test, linear_pred))), 3),
                "r2": round(float(r2_score(y_test, linear_pred)), 3),
            })
            for algorithm, estimator in (
                ("Gradient Boosting", GradientBoostingRegressor(random_state=42)),
                ("Decision Tree", DecisionTreeRegressor(random_state=42)),
            ):
                estimator.fit(X_train, y_train)
                estimator_pred = estimator.predict(X_test)
                comparison_rows.append({
                    "algorithm": algorithm,
                    "mae": round(float(mean_absolute_error(y_test, estimator_pred)), 3),
                    "mse": round(float(mean_squared_error(y_test, estimator_pred)), 3),
                    "rmse": round(float(np.sqrt(mean_squared_error(y_test, estimator_pred))), 3),
                    "r2": round(float(r2_score(y_test, estimator_pred)), 3),
                })
            # Tune the deployed model family on training data only to avoid test-set leakage.
            search = GridSearchCV(
                RandomForestRegressor(random_state=42, n_jobs=-1),
                param_grid={
                    "n_estimators": [100, 200],
                    "max_depth": [None, 8],
                    "min_samples_leaf": [1, 2],
                },
                scoring="neg_mean_absolute_error",
                cv=5,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            optimized_pred = search.best_estimator_.predict(X_test)
            comparison_rows.append({
                "algorithm": "Optimized Random Forest",
                "mae": round(float(mean_absolute_error(y_test, optimized_pred)), 3),
                "mse": round(float(mean_squared_error(y_test, optimized_pred)), 3),
                "rmse": round(float(np.sqrt(mean_squared_error(y_test, optimized_pred))), 3),
                "r2": round(float(r2_score(y_test, optimized_pred)), 3),
            })
            optimization_summary = {
                "method": "GridSearchCV with 5-fold cross-validation",
                "scoring": "negative mean absolute error",
                "best_parameters": search.best_params_,
            }
            if model is None:
                generate_charts(df)
        except Exception as exc:
            resource_errors.append(f"Could not calculate model analytics: {exc}")
            if df is not None:
                try:
                    generate_charts(df)
                except Exception:
                    pass


initialize_database()
setup_resources()


def fetch_history(limit: int = 12) -> list[dict]:
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


def predict_grade(studytime: int, absences: int, g1: int, g2: int) -> dict:
    if model is None:
        raise ValueError("The saved model is not ready. Check the setup message at the top of this page.")
    if studytime not in (1, 2, 3, 4):
        raise ValueError("Study-time category must be 1, 2, 3, or 4.")
    if not 0 <= absences <= 100:
        raise ValueError("Absences must be between 0 and 100.")
    if not 0 <= g1 <= 20 or not 0 <= g2 <= 20:
        raise ValueError("G1 and G2 grades must be between 0 and 20.")

    features = pd.DataFrame([{
        "studytime": studytime,
        "absences": absences,
        "G1": g1,
        "G2": g2,
    }], columns=FEATURES)
    raw_prediction = float(model.predict(features)[0])
    predicted_grade = float(np.clip(raw_prediction, 0, 20))
    if predicted_grade < 10:
        risk = "Needs support"
        recommendation = "Prioritize attendance and targeted revision before the final assessment."
    elif predicted_grade < 14:
        risk = "Developing"
        recommendation = "Maintain study consistency and focus on strengthening weaker topics."
    else:
        risk = "On track"
        recommendation = "Keep the current study routine and consider extension activities."
    return {
        "grade": round(predicted_grade, 2),
        "raw_grade": round(raw_prediction, 2),
        "label": risk,
        "recommendation": recommendation,
    }


@app.route("/", methods=["GET", "POST"])
def dashboard():
    prediction = None
    entered = {"studytime": "2", "absences": "5", "g1": "12", "g2": "14"}

    if request.method == "POST":
        entered = {
            "studytime": request.form.get("studytime", "").strip(),
            "absences": request.form.get("absences", "").strip(),
            "g1": request.form.get("g1", "").strip(),
            "g2": request.form.get("g2", "").strip(),
        }
        try:
            studytime = int(entered["studytime"])
            absences = int(entered["absences"])
            g1 = int(entered["g1"])
            g2 = int(entered["g2"])
            prediction = predict_grade(studytime, absences, g1, g2)
            with get_db_connection() as conn:
                conn.execute(
                    """INSERT INTO predictions
                    (created_at, studytime, absences, g1, g2, predicted_grade)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), studytime,
                     absences, g1, g2, prediction["grade"]),
                )
            flash("Prediction created and saved to history.", "success")
        except (ValueError, TypeError) as exc:
            flash(str(exc), "error")
        except Exception as exc:
            flash(f"Prediction failed: {exc}", "error")

    history = fetch_history()
    return render_template(
        "index.html",
        title="Student Performance Analytics",
        resource_errors=resource_errors,
        model_ready=model is not None,
        dataset_ready=df is not None,
        summary=summary,
        comparisons=comparison_rows,
        optimization=optimization_summary,
        prediction=prediction,
        entered=entered,
        history=history,
    )


@app.get("/health")
def health():
    return jsonify({
        "status": "online",
        "service": "Student Performance Prediction Dashboard",
        "model_ready": model is not None,
        "dataset_ready": df is not None,
    })


@app.get("/api/stats")
def api_stats():
    return jsonify(summary)


@app.get("/api/metrics")
def api_metrics():
    return jsonify({
        "models": comparison_rows,
        "test_split": "80:20",
        "random_state": 42,
        "optimization": optimization_summary,
    })


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True) or {}
    try:
        prediction = predict_grade(
            int(payload.get("studytime", 2)),
            int(payload.get("absences", 5)),
            int(payload.get("g1", 12)),
            int(payload.get("g2", 14)),
        )
        return jsonify(prediction)
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/clear-history")
def clear_history():
    with get_db_connection() as conn:
        conn.execute("DELETE FROM predictions")
    flash("Prediction history cleared.", "success")
    return redirect(url_for("dashboard"))


if __name__ == "__main__":
    # Local-only development server. Do not expose this debug server to the internet.
    app.run(host="127.0.0.1", port=5000, debug=False)
