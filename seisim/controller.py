"""Controller"""


class Controller:
    """Controller base class"""
    def __init__(self):
        """Constructor"""
        pass

    def run(self, measurement):
        """Output"""
        return self.get_actuation(measurement)

class MCRHC(Controller):
    """Minimizing coherence rolling horizon control"""
    pass

class MPCFF(Controller):
    """Model Predictive Feedforward Control"""
    def __init__(self, model, n_predict, n_control, predictor):
        """Constructor

        Parameters
        ----------
        model : seisim.system.System
            Seismic isolation system plant.
            Here we assume actuation and disturbance the same plant.
        n_predict : int
            Prediction horizon
        n_control : int
            Control horizon
        predictor : seisim.predictor.Predictor
            Seismic noise predictor.
        """
        super().__init__()
        self.model = model
        self.n_predict = n_predict
        self.n_control = n_control
        self.predictor = predictor

    def get_actuation(self, seismic):
        """Get actuation signals
        
        Parameters
        ----------
        seismic : array
            Past seismic noise time series.

        Returns
        -------
        actuation : array
            Actuation signals.
        """
        # Get matching-length inputs for predictor
        n_history = self.predictor.n_history
        if len(seismic) < n_history:
            raise ValueError("Not enough seismic data for prediction\n"
                             f"Predictor requires: {n_history}\n"
                             f"Provided: {len(seismic)}.")
        data_in = seismic[-n_history:]

        # Get prediction
        seismic_forecast = self.predictor.run(data_in)

        # # TODO Need to estimate real-time model states (state observer etc).

        # model_y = np.zeros_like(seismic_forecast)
        # # Predict disturbance
        # for i in range(len(seismic_forecast)):
        #     model_y[i] = model.step(seismic_forecast[i])

        # Use closed form solution, assume same actuation and disturbance plant
        # For now
        actuation = -seismic_forecast[:self.n_control]
        
        return actuation







