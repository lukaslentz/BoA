import os
import json
from tqdm import tqdm
import numpy as np
from scipy.integrate import solve_ivp
from itertools import product
import matplotlib.pyplot as plt
import sqlite3
import logging

script_path = os.path.dirname(os.path.abspath(__file__))
print(script_path)

# Configure logging
logging.basicConfig(
    filename= script_path + '//app.log',  # Log file name
    level=logging.INFO,  # Logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
    datefmt='%Y-%m-%d %H:%M:%S'  # Date format
)

# Write log messages
logging.info("This is an info message.")

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
        if not sol.y_events == []:
            cur_turning_points = sol.y_events[0][:,0]
            cur_turning_times = sol.t_events[-1]
            if self.pattern_match(self.turning_points, cur_turning_points):
                self.is_stationary = True
            self.turning_points = np.concatenate([self.turning_points, cur_turning_points])
            self.turning_times = np.concatenate([self.turning_times, cur_turning_times])

    def pattern_match(self, array, pattern, eps=1e-11):
        matches = [i for i in range(len(array) - len(pattern) + 1) if np.linalg.norm(array[i:i + len(pattern)] - pattern) < eps]
        if len(matches) > 2: 
            differences = np.diff(matches)
            is_uniform = np.all(differences == differences[0])
            if is_uniform:
                self.turning_points_sequence = array[matches[0]:matches[1]] 
                return True
        else: return False

    def solve(self, max_iterations=1000):
        logging.info(f'start new solution process with Omega = {self.Omega} and y0 = {self.y0}')
        y0 = self.y0
        iteration = 0
        t0, t1 = 0, 10*self.T
        while iteration < max_iterations:
            iteration += 1
            self.iteration = iteration
            sol = solve_ivp(self.sys.rhs, [t0,t1], y0, args=(self.Omega,), events=self.v_event)
            if sol.success:
                self.check_if_solution_is_stationary(sol)
                if self.is_stationary:
                    self.success = True
                    self.final_state = sol.y[:,-1]
                    logging.info(f'\t-->found stationary solution after {iteration} iterations')
                    break
                y0 = sol.y[:,-1]
                t0, t1 = t1, t1 + self.T   
            if not sol.success:
                logging.info(f'numerical integration failed in iteration {iteration} with initial conditions {y0}')
                break
        if iteration == max_iterations:
                logging.info(f'couldnt find a stationary solution within {iteration} iterations for initial conditions {y0}')

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

class Grid():
    def __init__(self, x_min=-2, x_max=2, y_min=-2, y_max=2, dim_x=1001, dim_y=1001):
        self.x = np.linspace(x_min, x_max, dim_x)
        self.y = np.linspace(y_min, y_max, dim_y)
        self.cells = [[round(x,3), round(y,3)] for x, y in product(self.x, self.y)]

class Database():
    def __init__(self, path):
        self.conn = sqlite3.connect(path)
        self.cursor = self.conn.cursor()
        self.table_names = []
             
    def add_table(self, table_name):
        self.table_names.append(table_name)
        self.cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            initial_values TEXT NOT NULL, 
            signature TEXT NOT NULL,  
            type INTEGER NOT NULL,
            steps INTEGER NOT NULL
        )
        """)
        self.conn.commit()

    def delete_table(self, table_name):
        self.cursor.execute(f"DELETE FROM {table_name}")
        self.conn.commit()

    def add_entry(self, table_name, initial_values, signature, type, steps):
        self.cursor.execute(f"""
            INSERT INTO {table_name} (initial_values, signature, type, steps)
            VALUES (?, ?, ?, ?)
            """, (json.dumps(initial_values), json.dumps(signature), type, steps))
        self.conn.commit()

    def close(self):
        self.conn.close()

def main():
    system = System()
    table_name = "tab_1"
    db = Database(script_path + '//BoA.db')
    db.delete_table(table_name)
    db.add_table(table_name)
    grid = Grid()
    for cell in tqdm(grid.cells):
        sol = Solution(system, cell, 0.8)
        sol.solve()
        if sol.success:
            db.add_entry(table_name, cell, tuple(sol.turning_points_sequence), -99, sol.iteration)
        else:
            db.add_entry(table_name, cell, (-99), -99, sol.iteration)
    db.close()


main()