from modules.grid import Grid
from modules.system import System 

import os
from tqdm import tqdm
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import logging
import pickle

script_path = os.path.dirname(os.path.abspath(__file__))

Omega = 0.9

with open(script_path + f"/Ergebnisse/data_Omega_{Omega}.pkl", "rb") as file:
    grid = pickle.load(file)

def group_m(a):
    threshold = 8
    groups = [[a[0]]]
    for value in a[1:]:
        if np.linalg.norm(np.array(value) - np.array(groups[-1][-1])) <= threshold:
            groups[-1].append(value)
        else:
            groups.append([value])
    return groups

def plot_fix_points(ax):
    fix_points = [value for index, value in enumerate(grid.transition) if index == value]
    plot_values = np.zeros(grid.dim)
    for point in fix_points:
        plot_values[point] = 1
    plot_matrix = np.transpose(plot_values.reshape(grid.m, grid.n))[::-1, :]
    cax = ax.imshow(plot_matrix, cmap='viridis', interpolation='nearest',extent=(grid.j_vec[0],grid.j_vec[-1],grid.i_vec[0],grid.i_vec[-1]))
    ax.figure.colorbar(cax, ax=ax, label='Value')

def plot_fix_point_orbits(ax):
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow','orange', 'purple', 'pink', '#FF5733', '#33FF57', '#3357FF','blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'pink', '#FF5733', '#33FF57', '#3357FF']
    ax.set_xticks(grid.j_vec)
    ax.set_yticks(grid.i_vec)
    #ax.grid(True)
    system = System(Omega)
    t0 = 0*system.T
    te = 1*system.T
    fix_points = [value for index, value in enumerate(grid.transition) if index == value]
    for c,point in enumerate(fix_points):
        y0 = grid.get_initial_values(point)
        sol = solve_ivp(system.rhs, [t0, te], y0, t_eval=np.linspace(t0, te, 101))
        (i,j) = grid.get_position_as_tuple(sol.y[:,-1])
        y = (grid.i_vec[i-1],grid.j_vec[j-1])
        ax.plot(sol.y[0],sol.y[1],color=colors[c])
        ax.scatter(*y0,color=colors[c])
        ax.scatter(*y,color=colors[c])

def plot_solutions(ax):
    system = System(Omega)
    t0 = 150*system.T
    te = 151*system.T
    fix_points = [value for index, value in enumerate(grid.transition) if index == value]
    for c,point in enumerate(fix_points):
        y0 = grid.get_initial_values(point)
        sol = solve_ivp(system.rhs, [0, te], y0, t_eval=np.linspace(t0, te, 101))
        ax.plot(sol.y[0],sol.y[1],color='black')

def plot_boa(ax):
    fixed_point_tuples = sorted([grid.int_to_tuple(value) for index, value in enumerate(grid.transition) if index == value])
    groups_tuples = group_m(fixed_point_tuples)
    groups_int = [[grid.tuple_to_int(t) for t in group] for group in groups_tuples]
    plot_values = np.zeros(grid.dim)
    for point in grid.outsiders:
        plot_values[point] = -1
    for point in grid.fails:
        plot_values[point] = -2
    for i,group in enumerate(groups_int):
        for type in group:
            ind = grid.solutions_types.index(type)
            for point in grid.solutions_cells[ind]:
                plot_values[point] = i

    colors = [(0, 'white'),(0.25,'white'), (0.5, 'blue'), (0.75, 'red'), (1,'yellow')]  # (position, color)
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    
    plot_matrix = np.transpose(plot_values.reshape(grid.m, grid.n))[::-1, :]
    cax = ax.imshow(plot_matrix, cmap=cmap, interpolation='nearest',extent=(grid.j_vec[0],grid.j_vec[-1],grid.i_vec[0],grid.i_vec[-1]))
    ax.figure.colorbar(cax, ax=ax, label='Value')

def plots():
    fig, ax = plt.subplots()
    #plot_boa(ax)
    #plot_fix_points(ax)
    plot_fix_point_orbits(ax)
    #plot_solutions(ax)
    plt.show()

def main():
    #plots()
    print(grid)

if __name__ == "__main__":
    main()