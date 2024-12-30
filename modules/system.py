import numpy as np

class System():
    def __init__(self, Omega):
        self.xi = 0.044
        self.alpha = 1
        self.beta = 0
        self.gamma = -0.1
        self.f = 0.08
        self.Omega = Omega
        self.T = 2*np.pi/self.Omega

    def rhs(self, t, state):
        return [
            state[1],
            - self.xi * state[1] 
            - self.alpha * state[0] 
            - self.beta * state[0]**2 
            - self.gamma * state[0]**3 
            + self.f * np.cos(self.Omega*t)
            ]
    
    def jacobian(self, state):
        return np.array[
            [0, 1],
            [ - self.alpha - 2*self.beta * state[0] - 3*self.gamma * state[0]**2, -self.xi ]
            ] 