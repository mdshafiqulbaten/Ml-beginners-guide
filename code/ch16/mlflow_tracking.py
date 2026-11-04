# mlflow_tracking.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: NOT CONFIRMED - please review.
# Review before running, especially if the status above is not "passed".

the mlflow package). Syntax-checked only.
 """Experiment tracking: log parameters, metrics and the model so every run is
 reproducible."""
 import mlflow
 import mlflow.sklearn
 from sklearn.datasets import load_breast_cancer
 from sklearn.ensemble import RandomForestClassifier
 from sklearn.metrics import roc_auc_score
 from sklearn.model_selection import train_test_split

 X, y = load_breast_cancer(return_X_y=True)
 X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y,
 random_state=42)
 mlflow.set_experiment("breast-cancer")
 for n_trees in [50, 200]:
      with mlflow.start_run():
          model = RandomForestClassifier(n_trees, random_state=0).fit(X_tr, y_tr)
          mlflow.log_param("n_estimators", n_trees)
          mlflow.log_metric("test_auc", roc_auc_score(y_te, model.predict_proba(X_te)[:,
 1]))
          mlflow.sklearn.log_model(model, "model")
 # View the runs: mlflow ui

Docker: Making Your Model Portable
Your model works on your laptop. Will it work on the production server? Docker
containers package your code, dependencies, and environment into a single portable
unit that runs identically anywhere.

Companion file: code/ch16/Dockerfile | Not executed in the book’s test environment (needs Docker).
Syntax-checked only.
 # Container for the churn API (see code/project). NOT built in the book-production
 environment.
 FROM python:3.12-slim
 WORKDIR /app
 COPY requirements-api.txt .
 RUN pip install --no-cache-dir -r requirements-api.txt
 COPY app.py churn_model.joblib ./
 EXPOSE 8000
 CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

Key Terms
 Term                                              Meaning

 Baseline                                          A simple model or rule you must beat to justify
                                                   anything more complex.
 Deployment                                        Making a trained model available to be used by
                                                   other software or people.
 Monitoring                                        Watching a live model’s inputs and results to
                                                   detect problems and drift.
 Data drift                                        A change over time in the data the model sees,
                                                   which can quietly reduce accuracy.
 Model card                                        A short document stating what a model is for,
                                                   how it was tested and where it fails.

Check Your Understanding
Try these before you read on. Model answers are in Appendix D. Exercises that
mention a script use the free companion code.
  1.     Run the five churn scripts in code/project/. What is the best decision
         threshold, and where does it come from?
  2.     Logistic regression beat random forest and gradient boosting (AUC 0.831 vs
