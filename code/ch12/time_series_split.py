# time_series_split.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: NOT CONFIRMED - please review.
# Review before running, especially if the status above is not "passed".

 """Time series: why you must never shuffle the past into the future. Real data: weekly
 CO2 at Mauna Loa since 1958."""
 import numpy as np
 import pandas as pd
 import statsmodels.api as sm
 from sklearn.ensemble import GradientBoostingRegressor
 from sklearn.metrics import mean_absolute_error
 from sklearn.model_selection import KFold

 co2 = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean().interpolate()
 # monthly series
 t = np.arange(len(co2))
 frame = pd.DataFrame({"t": t, "month": co2.index.month, "lag12": co2.shift(12).values,
 "lag1": co2.shift(1).values}, index=co2.index)
 frame["y"] = co2.values
 frame = frame.dropna()
 X, y = frame[["t", "month", "lag12"]], frame["y"]            # lag12 = the value one
 year earlier (known at prediction time)
 model = lambda: GradientBoostingRegressor(n_estimators=300, max_depth=3, random_state=0)

 # WRONG: random 5-fold CV. Training folds contain months on BOTH sides of every test
 month.
 shuffled = []
 for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
      shuffled.append(mean_absolute_error(y.iloc[te], model().fit(X.iloc[tr],
 y.iloc[tr]).predict(X.iloc[te])))
 # RIGHT: train on the past, test on the future (the last 8 years)
 cut = len(frame) - 96
 fut = model().fit(X.iloc[:cut], y.iloc[:cut]).predict(X.iloc[cut:])
 future_mae = mean_absolute_error(y.iloc[cut:], fut)
 naive = mean_absolute_error(y.iloc[cut:], X["lag12"].iloc[cut:])       # 'same as a year
 ago'
 print(f"Shuffled cross-validation error: {np.mean(shuffled):.2f} ppm    <- looks
 excellent")
 print(f"Honest forecast error (last 8 years, never seen): {future_mae:.2f} ppm")
 print(f"'Same as last year' baseline error:                 {naive:.2f} ppm")
 print("\nTrees cannot extrapolate a rising trend beyond the values they saw in training,
 which is why the honest error is larger.")
 np.savez("../outputs/ch12_co2.npz", dates=(frame.index.year + (frame.index.month - 1) /
 12).to_numpy(), y=y.to_numpy(), cut=cut, fut=fut)
