import numpy as np
from scipy.integrate import solve_ivp

class Grid():
    def __init__(self, i_values, j_values):
        self.i_vec = np.linspace(*i_values)
        self.j_vec = np.linspace(*j_values)
        self.m = len(self.i_vec)
        self.n = len(self.j_vec)
        self.dim = self.m * self.n
        self.cells = set(range(0, self.dim))
        self.transition = [0] * self.dim
        self.solutions = set(())
        self.solutions_types = []
        self.solutions_cycles = []
        self.solutions_cells = []
        self.outsiders = set(())
        self.fails = set(())
        self.not_finished = set(())
        self.cnt = [0,0,0,0]

    def is_inside_grid(self, point):
        if self.i_vec[0] <= point[0] <= self.i_vec[-1] and self.j_vec[0] <= point[1] <= self.j_vec[-1]:
            return True
        else:
            return False
    
    def tuple_to_int(self, tuple):
        return (tuple[0] - 1) * self.n + tuple[1] - 1

    def int_to_tuple(self, ind):
        return (ind // self.n + 1, np.mod(ind, self.n) + 1)
    
    def get_position_as_tuple(self, point):
        i = np.argmin(np.abs(self.i_vec - point[0])) + 1
        j = np.argmin(np.abs(self.j_vec - point[1])) + 1 
        return (i,j)
    
    def get_position_as_int(self, point):
        return self.tuple_to_int(self.get_position_as_tuple(point))
    
    def get_initial_values(self, int):
        i, j = self.int_to_tuple(int)
        return (round(self.i_vec[i-1],5), round(self.j_vec[j-1],5))
    
    def solve_single(self, system, cell):
        sol = solve_ivp(system.rhs, [0, system.T], self.get_initial_values(cell), t_eval=[0, system.T])
        if sol.success:
            if self.is_inside_grid(sol.y[:,-1]):
                self.transition[cell] = self.get_position_as_int(sol.y[:,-1])
                self.solutions.add(cell)
            else:
                self.transition[cell] = -1
                self.outsiders.add(cell)
        else:
            self.transition[cell] = -2
            self.fails.add(cell)   

    def evaluate_solutions(self):
        solution_set = set(self.solutions)
        while solution_set:
            cur_cycle = []
            cur_ind = solution_set.pop()
            while cur_ind not in cur_cycle:
                if cur_ind == -1:
                    self.outsiders.update(cur_cycle)
                    break
                elif cur_ind == -2:
                    self.fails.update(cur_cycle)
                    break
                cur_cycle.append(cur_ind)
                cur_ind = self.transition[cur_ind]
            if cur_ind in self.solutions_types:
                sol_ind = self.solutions_types.index(cur_ind)
            else:
                self.solutions_types.append(cur_ind)
                sol_ind = len(self.solutions_types) -1  
                self.solutions_cycles.append([])
                self.solutions_cells.append(set(()))
            self.solutions_cycles[sol_ind].append(cur_cycle)
            self.solutions_cells[sol_ind].update(cur_cycle)

    def create_plot_matrix(self):
        values = np.zeros(self.dim)
        sol = 0
        for i,type in enumerate(self.solutions_types):
            if type > 0:
                print(type)
                sol += 1
                for cell in self.solutions_cells[i]:
                    values[cell] = sol
        self.plot_matrix = np.transpose(values.reshape(self.m, self.n))[::-1, :]
    
    def info(self):
        print('\n' + '*' * 20)
        print(f'\t {self.dim:>10} cells:')
        print(f'\t {len(self.solutions):>10} solutions:')
        print(f'\t {len(self.fails):>10} fails:')
        print(f'\t {len(self.outsiders):>10} outsiders:')
        print('\n' + '-' * 50)
        print(f'\t total: {len(self.outsiders) + len(self.fails) + len(self.solutions) :>10}')
        print('*' * 20 + '\n')