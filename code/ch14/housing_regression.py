# housing_regression.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: NOT CONFIRMED - please review.
# Review before running, especially if the status above is not "passed".

 """A real regression project: predict California house prices from 1990 census data."""
 import numpy as np
 import pandas as pd
 from sklearn.compose import ColumnTransformer
 from sklearn.dummy import DummyRegressor
 from sklearn.ensemble import HistGradientBoostingRegressor
 from sklearn.impute import SimpleImputer
 from sklearn.linear_model import LinearRegression
 from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
 from sklearn.model_selection import train_test_split
 from sklearn.pipeline import Pipeline
 from sklearn.preprocessing import OneHotEncoder, StandardScaler

 df = pd.read_csv("../data/real/housing.csv")
 print("rows:", len(df), "| missing total_bedrooms:", df.total_bedrooms.isna().sum())
 print(f"houses priced at the 500,000 cap: {(df.median_house_value >= 500000).mean():.1%}
 <- the data stops here")

 X, y = df.drop(columns="median_house_value"), df["median_house_value"]
 X["rooms_per_household"] = X.total_rooms / X.households            # feature engineering
 X["people_per_household"] = X.population / X.households
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
 random_state=42)

 num = [c for c in X.columns if c != "ocean_proximity"]
 prep = ColumnTransformer([("num", Pipeline([("fill", SimpleImputer(strategy="median")),
 ("scale", StandardScaler())]), num),
                            ("cat", OneHotEncoder(handle_unknown="ignore"),
 ["ocean_proximity"])])
 models = {"Always guess the average": DummyRegressor(),
            "Linear regression": Pipeline([("p", prep), ("m", LinearRegression())]),
            "Gradient boosting": Pipeline([("p", prep), ("m",
 HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=0))])}
 preds = {}
 print(f"\n{'model':26s}{'RMSE':>10s}{'MAE':>10s}{'R^2':>8s}")
 for name, m in models.items():
     m.fit(X_train, y_train)
     p = m.predict(X_test); preds[name] = p
     print(f"{name:26s}{np.sqrt(mean_squared_error(y_test, p)):10,.0f}
 {mean_absolute_error(y_test, p):10,.0f}{r2_score(y_test, p):8.3f}")

 err = np.abs(preds["Gradient boosting"] - y_test)
 capped = y_test >= 500000
 print(f"\nmean absolute error on capped (500,000) houses: {err[capped].mean():,.0f}   on
 all others: {err[~capped].mean():,.0f}")
 np.savez("../outputs/ch14_housing.npz", y=y_test.to_numpy(), p=preds["Gradient
 boosting"], lat=df.loc[X_test.index, "latitude"].to_numpy(),
          lon=df.loc[X_test.index, "longitude"].to_numpy())
