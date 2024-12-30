from modules.grid import Grid
from modules.system import System 

import os
import json
from tqdm import tqdm
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
import logging
import pickle

script_path = os.path.dirname(os.path.abspath(__file__))

# Configure logging
logging.basicConfig(
    filename= script_path + '//zellabbildung.log',  # Log file name
    level=logging.INFO,  # Logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
    datefmt='%Y-%m-%d %H:%M:%S'  # Date format
)

def main():
    for Omega in tqdm(np.arange(0.875,0.4,-0.05)):
        system = System(Omega)
        grid = Grid((-5, 5, 1001),(-5, 5, 1001))
        for cell in tqdm(grid.cells):
            grid.solve_single(system, cell)
        grid.evaluate_solutions()
        grid.create_plot_matrix()
        grid.plot()
        plt.savefig(script_path + f"/boa_Omega_{Omega}.pdf")
        plt.close()
        with open(script_path + f"/data_Omega_{Omega}.pkl", "wb") as file:
            pickle.dump(grid, file)


if __name__ == "__main__":
    main()