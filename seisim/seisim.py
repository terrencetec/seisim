"""Run simulation and benchmark"""
import numpy as np
import scipy.signal


class Seisim:
    """Seismic simulation class

    Takes system, controller, and predictor, simulate and assign scores
    """
    def __init__(self, system, controller):
        """Constructor
        
        Parameters
        ----------
        system : seisim.system.System
            The seismic isolation system
        controller : seisim.controller.Controller
            The controller
        """
        self.system = system
        self.controller = controller
        # self.predictor = predictor

    def simulate(self, t, ground_motion):
        """Run simulation
        
        Parameters
        ----------
        t : array
            Time array.
        ground_motion : array
            Seismic time series to be benchmarked.

        Returns
        -------
        motion : array
            Isolation motion.
        """
        # Get history, prediction, and control horizons.
        n_history = self.controller.predictor.n_history
        n_predict = self.controller.n_predict
        n_control = self.controller.n_control
        
        # Separate actual ground motion and
        # filtered ground motion for prediction
        g_actual = ground_motion

        # Band pass for prediction
        N = 4
        lowcut = 0.05
        highcut = 0.5
        fs = 1/(t[1]-t[0])
        sos = scipy.signal.butter(
            N, (lowcut, highcut), btype="bandpass", fs=fs, output="sos")
        g_filt = scipy.signal.sosfiltfilt(sos, g_actual)

        # Initialize empty isolation motion array
        motion = np.zeros_like(g_actual)

        # Simulate isolation motion before control kicks in
        for i in range(n_history):
            u = g_actual[i]
            motion[i] = self.system.step(u)[0, 0]

        # Simulate isolation motion after control kics in
        for i in range(n_history, len(t), n_control):
            # Get system input
            g_future = g_actual[i:i+n_control]
            g_filt_past = g_filt[i-n_history : i]
            actuation = self.controller.run(g_filt_past)
            u = g_future + actuation[:len(g_future)]
            # Assume seismic and actuation use same path

            # Forward in time
            y_running = np.zeros_like(u)
            for j in range(len(u)):
                y_running[j] = self.system.step(u[j])[0, 0]

            # Put segment into final motion array
            # Separate boundary cases
            if len(motion[i:i+n_control]) < n_control:
                len_remain = len(motion[i:i+n_control])
                motion[i:] = u[:len_remain]
            else:
                motion[i:i+n_control] = u

        return motion


        
