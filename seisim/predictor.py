"""Time series predictor"""
import numpy as np
import math


class Predictor:
    """Base predictor class"""
    def __init__(self):
        """Constructor"""
        pass

    def run(self, data_in):
        """Place holder run prediction
        
        Parameters
        ----------
        data_in : array
            Data input

        Returns
        -------
        prediction : array
            Prediction
        """
        prediction = self.predict(data_in)
        
        return prediction
    
    def predict(self, data_in):
        """Prediction algorithm placeholder"""
        return data_in


class ConstantOrder(Predictor):
    """Constant order predictor"""
    def __init__(self, dt, order=2, n_predict=1):
        """Constructor
        
        Parameters
        ----------
        dt : float
            Time step
        order : int, default=2 
            Model order:
                0 = constant
                1 = constant velocity
                2 = constant acceleration
                ...
        n_predict : int, default=1
            Number of points to forecast
        """
        super().__init__()
        self.dt = dt
        self.order = order
        self.n_predict = n_predict

        # Required length for prediction:
        self.n_history = order + 1

    def predict(self, y):
        """Run algorithm

        Parameters
        ----------
        y : array
            The time series.
            len(y) must be greater order.

        Returns
        -------
        forecast : array
            Forecasted time series.
        """
        y = np.asarray(y, dtype=float)
        n = len(y)
        dt = self.dt
        order = self.order
        n_predict = self.n_predict

        if n < order + 1:
            raise ValueError("Not enough data for requested order")

        # --- Fit local polynomial to last points ---
        t = np.arange(-(order), 1) * dt
        coeffs = np.polyfit(t, y[-(order+1):], order)

        # --- Extract derivatives at t=0 ---
        # poly: c0*t^N + ... + cN
        state = np.zeros(order + 1)
        for k in range(order + 1):
            state[k] = math.factorial(k) * coeffs[-(k+1)]

        # --- Build state transition matrix ---
        F = np.zeros((order + 1, order + 1))
        for i in range(order + 1):
            for j in range(i, order + 1):
                F[i, j] = dt**(j - i) / math.factorial(j - i)

        # --- One-step-ahead fit ---
        # fitted_last = (F @ state)[0]

        # --- Forecast ---
        forecast = np.zeros(n_predict)
        x = state.copy()
        for k in range(n_predict):
            x = F @ x
            forecast[k] = x[0]

        return forecast
        
