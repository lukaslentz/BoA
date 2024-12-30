import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class BiF():
    def __init__(self, system):
        self.system = system
        self.solutions = []

    def set_bif_parameter(self, min=0.1, max=2, step_size=1e-2):
        self.para_min = min
        self.para_max = max
        self.para_step_size = step_size

    def sweep(self):
        y0 = (2,0)
        for para in np.arange(self.para_min, self.para_max, self.para_step_size):
            cur_sol = Solution(self.system, y0, para)
            cur_sol.solve()
            if cur_sol.success:
                self.solutions.append(cur_sol)
                y0 = cur_sol.final_state
            else:
                print('couldnt find solution')
                break


class Solution():
    def __init__(self, system, y0 : list, Omega):
        self.sys = system
        self.y0 = y0
        self.Omega = Omega
        self.T = 2*np.pi/Omega

        self.turning_points = np.array([], dtype=float)
        self.turning_times = np.array([], dtype=float)

        self.success = False
        self.solved = False
        self.is_stable = False
        self.is_stationary = False
        self.type = None
        self.transient_time = None

    @staticmethod
    def v_event(t, state, Omega):
        return state[1]

    def check_if_solution_is_stationary(self, sol):
        cur_turning_points = sol.y_events[0][:,0]
        cur_turning_times = sol.t_events[-1]
        if self.pattern_match(self.turning_points, cur_turning_points):
            self.is_stationary = True
        self.turning_points = np.concatenate([self.turning_points, cur_turning_points])
        self.turning_times = np.concatenate([self.turning_times, cur_turning_times])

    def pattern_match(self, array, pattern, eps=1e-7):
        matches = [i for i in range(len(array) - len(pattern) + 1) if np.linalg.norm(array[i:i + len(pattern)] - pattern) < eps]
        if len(matches) > 2: 
            differences = np.diff(matches)
            is_uniform = np.all(differences == differences[0])
            if is_uniform:
                self.turning_points_sequence = array[matches[0]:matches[1]] 
                return True
        else: return False

    def solve(self,max_iterations=5000):
        print(f'start new solution process with Omega = {self.Omega} and y0 = {self.y0}')
        y0 = self.y0
        iteration = 0
        t0, t1 = 0, self.T
        while iteration < max_iterations:
            iteration += 1
            sol = solve_ivp(self.sys.rhs, [t0,t1], y0, args=(self.Omega,), events=self.v_event)
            if sol.success:
                self.check_if_solution_is_stationary(sol)
                if self.is_stationary:
                    self.success = True
                    self.final_state = sol.y[:,-1]
                    self.iteration = iteration
                    print(f'\t-->found stationary solution after {iteration} iterations')
                    break
                y0 = sol.y[:,-1]
                t0, t1 = t1, t1 + self.T   
            if not sol.success:
                print(f'numerical integration failed in iteration {iteration} with initial conditions {y0}')
                break
        if iteration == max_iterations:
                    print(f'couldnt find a stationary solution within {iteration} iterations for initial conditions {y0}')

class System():
    def __init__(self):
        self.xi = 0.044
        self.alpha = 1
        self.beta = 0
        self.gamma = -0.1
        self.f = 0.08

    def rhs(self, t, state, Omega):
        return [
            state[1],
            - self.xi * state[1] 
            - self.alpha * state[0] 
            - self.beta * state[0]**2 
            - self.gamma * state[0]**3 
            + self.f * np.cos(Omega*t)
            ]
    
    def jacobian(self, state):
        return np.array[
            [0, 1],
            [ - self.alpha - 2*self.beta * state[0] - 3*self.gamma * state[0]**2, -self.xi ]
            ]



def create_new_axes():
    fig, ax = plt.subplots()
    #ax.set_xlabel('Time (s)')       
    #ax.set_ylabel('Amplitude')       
    #ax.set_title('Sine Wave Example') 
    #ax.set_xlim(-3, 3)               
    #ax.set_ylim(-1.5, 1.5)           
    ax.grid(True)                    
    return ax                     

def main():
    sys = System()
    bif = BiF(sys)
    bif.set_bif_parameter()
    bif.sweep()

    ax_1 = create_new_axes()
    for sol in bif.solutions:
        ax_1.scatter(
            [sol.Omega]*len(sol.turning_points_sequence),
            sol.turning_points_sequence,
            marker='o',
            color='blue'
            )
    plt.show()


    # a = np.array([1,2,3,4,5,6,7,8,9])
    # b = np.array([2.9999999999999999999,4,5])
    # cur_sol.pattern_match(a,b)

    # 
    # 
    # plt.show()


main()