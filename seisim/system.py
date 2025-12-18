"""
Seisim system class.

A system defined with a transfer function.
"""
import control
import numpy as np


class System:
    def __init__(self, tf, dt):
        """Constructor
        
        Parameters
        ----------
        tf : TransferFunction
            Transfer function
        dt : float
            Sampling time.
        """
        # Initialize systems
        self.tf = tf
        self.dt = dt
        self.ss = control.tf2ss(self.tf)
        self.sysd = self.ss.sample(self.dt)

        # Initialize system states
        self.nstates = self.sysd.nstates
        self.ninputs = self.sysd.ninputs
        self.x = np.zeros((self.nstates, 1))

    def step(self, u):
        """Get new output and states
        
        Parameters
        ----------
        u : array
            Input vector.
        """
        
        u = np.reshape(u, (self.ninputs, 1))
        A = self.sysd.A
        B = self.sysd.B
        C = self.sysd.C
        D = self.sysd.D
        x = self.x
        x_ = A @ x + B @ u
        y = C @ x + D @ u
        self.x = x_

        return y

