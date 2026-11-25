# statlearn

A small Python library that provides a simple linear regression implementation.

## Features

- Fit a linear relationship between two one-dimensional datasets
- Predict values from the fitted model
- Calculate the R-squared score

## Installation

Clone the repository and install the required dependency:

```powershell
git clone <repository-url>
cd statlearn
pip install numpy
```

## Usage

Import `LinearRegression` directly from the package:

```python
from statlearn import LinearRegression

model = LinearRegression()
model.fit([1, 2, 3, 4], [2, 4, 6, 8])

print(model.coef_)      # 2.0
print(model.intercept_) # 0.0
print(model.predict([5, 6]))
print(model.score([1, 2, 3, 4], [2, 4, 6, 8]))
```

## API

### `fit(x, y)`

Fits the model using NumPy arrays or array-like data for the independent and dependent variables.

### `predict(x)`

Returns predicted values for the supplied input values. The model must be fitted before calling this method.

### `score(x, y)`

Returns the R-squared score for the supplied observations.

## Running the test

```powershell
python test_linear_regression.py
```
