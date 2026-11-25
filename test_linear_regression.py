import sys
import numpy as np
from pathlib import Path

# Allow this package-style test to run directly from the statlearn folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from statlearn import LinearRegression


def test_fit_and_predict(x , y ):
    model = LinearRegression()
    model.fit(x, y)

    predictions = model.predict(x)

    print("Slope:", model.coef_)
    print("Intercept:", model.intercept_)
    print("Predictions:", predictions)
    print("R² Score:", model.score(x, y))

    # Predict for a new observation
    print("Prediction for 6 study hours:",
        model.predict([1.2]))

x = np.array([1, 2, 3, 4, 5])
y = np.array([35, 40, 50, 55, 65])  
test_fit_and_predict(x, y)