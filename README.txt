STUDENT PERFORMANCE PREDICTION WEBSITE — LOCAL WINDOWS SETUP

FILES NEEDED IN ONE FOLDER
--------------------------
app.py
requirements.txt
templates\index.html
static\style.css
student-mat.csv                  (copy your existing dataset here)
student_performance_model.pkl    (copy your existing trained model here)

The website also creates prediction_history.db and chart PNGs automatically.

STEP 1 — COPY WEBSITE FILES
---------------------------
Extract this ZIP into:
C:\Users\Admin\Desktop\StudentPerformancePrediction

Allow the templates and static folders to be added. Do not delete or replace your
existing student-mat.csv or student_performance_model.pkl files.

STEP 2 — OPEN COMMAND PROMPT IN THE PROJECT FOLDER
--------------------------------------------------
Open the folder in File Explorer. Click the address bar, type cmd, then press Enter.

STEP 3 — INSTALL THE REQUIRED PACKAGES
--------------------------------------
Because your Jupyter Notebook already trained the model, first install Flask without
upgrading the machine-learning libraries that created your model:
  py -m pip install Flask

If the app later reports a missing package, install that package too. For a completely
fresh Python setup, you can install all packages with:
  py -m pip install -r requirements.txt

Important: keep scikit-learn compatible with the version used to save the PKL model.
A pickle/joblib model may not be compatible with every scikit-learn version.

STEP 4 — START THE WEBSITE
--------------------------
Run:
  py app.py

STEP 5 — OPEN IT IN YOUR BROWSER
--------------------------------
Go to:
  http://127.0.0.1:5000

Keep the Command Prompt window open while using the website. To stop it, return to
Command Prompt and press Ctrl+C.

FEATURES
--------
- Grade prediction form using studytime, absences, G1 and G2.
- Actionable next-step guidance based on the predicted grade band.
- Dataset summary and grade charts.
- Actual-vs-predicted chart for the saved Random Forest model.
- Random Forest, Linear Regression, Gradient Boosting, and Decision Tree metric comparison.
- Prediction history saved locally in SQLite.
- Optional JSON endpoints: /health, /api/stats, /api/metrics, and /api/predict.

API EXAMPLE
-----------
Send a POST request to http://127.0.0.1:5000/api/predict with:
  {"studytime": 2, "absences": 5, "g1": 12, "g2": 14}

TROUBLESHOOTING
---------------
1. If the dashboard says the dataset is missing, make sure student-mat.csv is in
the same folder as app.py. This app reads it with sep=';'.
2. If the trained model is missing, copy student_performance_model.pkl into the
same folder as app.py.
3. If model loading fails due to a scikit-learn version mismatch, check the version
used when you trained/saved the model in Jupyter and install a compatible version.
4. If port 5000 is already in use, close the other Flask app or change port=5000 at
the bottom of app.py to another unused port such as 5001.
5. This is a local educational demo. Do not expose the built-in development server
or sensitive student data to the public internet.
