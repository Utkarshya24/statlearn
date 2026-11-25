import numpy as np;

class LinearRegression:

  def __init__(self):
    self.coef_ = None
    self.intercept_ = None

  def fit(self, x, y):
    
    # convert inputs to numpy arrays
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    if len(x) != len(y):
      raise ValueError("x and y must have equal lengths")

    if len(x) < 2:
      raise ValueError("At least two observation are required")

    # Calculate the means
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Calculate Weight/Slope 
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if denominator == 0:
      raise ValueError("X must contain varying values")

    self.coef_ = numerator / denominator

    # Calculate intercept
    self.intercept_ = y_mean - self.coef_ * x_mean

    return self


  def predict(self, x):
    if self.coef_ is None:
      raise ValueError("Train the model before prediction")

    x = np.asarray(x, dtype=float)

    return self.coef_ * x + self.intercept_

  def score(self, x, y):
    y = np.asarray(y, dtype=float).reshape(-1)
    predictions = self.predict(x)

    ss_res = np.sum((y - predictions) ** 2)
    ss_total = np.sum((y - np.mean(y)) ** 2)

    if ss_total == 0:
      raise ValueError("R^2 is undefined for constant targets")

    return 1 - ( ss_res /ss_total )