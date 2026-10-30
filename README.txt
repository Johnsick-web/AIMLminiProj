STUDENT PERFORMANCE PREDICTION SYSTEM
=====================================

AI/ML mini project: Student Performance Prediction (CO1–CO5)

1. PROBLEM DEFINITION
---------------------
The system estimates a student's final mathematics grade (G3, out of 20)
from study time, absences, and the first two period grades (G1 and G2).
The purpose is early academic-support decision making: the output is an
estimate and must not be treated as a guaranteed examination outcome.

2. DATASET COLLECTION
---------------------
Dataset file: student-mat.csv
It is the mathematics student-performance dataset with 395 student records.
The semicolon-delimited file includes school, demographic, academic, attendance,
and grade attributes. The application uses these columns:

  studytime  Study-time category (1 = lowest, 4 = highest)
  absences   Number of school absences
  G1         First-period grade (0–20)
  G2         Second-period grade (0–20)
  G3         Final grade / target (0–20)

Keep the CSV in the same folder as app.py. The app reads it locally and does not
upload student records.

3. DATA PREPROCESSING
---------------------
At startup the app:
  - reads the semicolon-delimited CSV;
  - verifies all five required columns;
  - converts the four features and G3 to numeric values;
  - removes rows incomplete in those required columns;
  - reports original and cleaned record counts in the /api/stats response;
  - uses only the four declared features, preventing accidental feature leakage.

4. MODEL DEVELOPMENT
--------------------
student_performance_model.pkl is the saved Random Forest regression model used
for live predictions. The dashboard also fits and compares:
  - Linear Regression (baseline)
  - Gradient Boosting Regressor
  - Decision Tree Regressor
  - Optimized Random Forest Regressor

The comparison uses train_test_split(test_size=0.2, random_state=42), so results
are reproducible.

5. PERFORMANCE EVALUATION
-------------------------
The held-out test set is evaluated with:
  - MAE (mean absolute error)
  - MSE (mean squared error)
  - RMSE (root mean squared error)
  - R² score

The dashboard displays the metrics and generates:
  static/generated/grade_distribution.png
  static/generated/study_time_grade.png
  static/generated/actual_vs_predicted.png

6. MODEL OPTIMIZATION
---------------------
The app runs GridSearchCV for a Random Forest using five-fold cross-validation
on training data only. It searches n_estimators, max_depth, and
min_samples_leaf, scoring candidates by negative mean absolute error. The best
parameters and optimized test metrics are exposed at /api/metrics and shown in
the model-comparison section.

7. RESULT ANALYSIS AND DOCUMENTATION
-------------------------------------
The web dashboard provides:
  - dataset summary and pass-rate descriptive statistic;
  - grade distribution and study-time analysis charts;
  - actual-versus-predicted visualization;
  - model metric comparison;
  - an input form with validation and support recommendations;
  - locally stored prediction history in prediction_history.db.

The “Mini-project methodology” section in the dashboard maps every required
rubric item to the implementation. Limitations include the small, subject-
specific dataset, dependence on G1/G2 availability, and the fact that
correlation does not establish causation.

WINDOWS SETUP
-------------
1. Open a terminal in this folder.
2. Install dependencies:

     py -m pip install -r requirements.txt

3. Start the local server:

     py app.py

4. Open http://127.0.0.1:5000

The saved PKL model must be compatible with the installed scikit-learn version.
If it cannot be loaded, the dashboard reports the error and still shows any
dataset-only analytics it can calculate.

API ENDPOINTS
-------------
GET  /health
GET  /api/stats
GET  /api/metrics
POST /api/predict

Example request:

  {"studytime": 2, "absences": 5, "g1": 12, "g2": 14}

This is a local educational demonstration. Do not expose Flask's development
server or identifiable student data to the public internet.
