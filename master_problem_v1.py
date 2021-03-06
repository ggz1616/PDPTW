
# coding: utf-8

# In[61]:


#probably SP1 returns always a path

class Label:
    '''
    This class is used to construct the Labels for the Labelling algorithm used to solve the constrained shortest path problem
    '''
    def __init__(self ,number_of_node, arrival_time, load, cost, V_parent, O_parent, P, D, parent_node, n, dij_cost):
        self.id = number_of_node 
        self.t = arrival_time #arrival time in Label L 
        self.l = load # cummulative load at this node using the a path
        self.c = cost # the cost on this solution is the distance
        self.V = Set(V_parent) # V is the set of requests served by this path that may have already been served
        self.construct_V(V_parent, P, D)
        self.O = Set(O_parent) #O is the set of open requests. Requests that have started but not delivered yet
        self.construct_O(P, D, n)
        self.p = parent_node # previous node in the path
        self.d = dij_cost
        
    def construct_V(self, V_par, P, D):
        self.V = self.V.union(Set([self.id]))
       
    
    def construct_O(self, P, D, n):
        #print("id",self.id)
        #print("O",self.O)
        #print(self.id in self.O)
        if (self.id in D) and self.id-n in self.O:
            self.O = self.O.difference(Set([self.id-n]))
        elif self.id in P:
            self.O = self.O.union(Set([self.id]))
        
    def construct_path(self):
        self.path.append(self.id)
        
    
    def check_cardinality_O(self):
        if O.size <=2:
            return True
        else:
            return False 
        
def check_dominance(U, L):
    '''
    To use this function triangle inequality should be satisfied both on distances and times
    '''
    flag = False
    for paths in U:
        L_i = paths[-1]
        if L_i.id == L.id: #for the labels paths ending on the same node
            if L_i.c <= L.c and L_i.V.issubset(L.V) and L_i.O.issubset(L.O) and L_i.t <= L.t:
                flag = True
    return flag

def display_min_path(result):
    final_path = []
    min_cost = 100000
    for res in result:
        if res[-1].c <= min_cost:
            min_cost = res[-1].c
    
    for res in result:
        if res[-1].c == min_cost:
            path_temp = []
            for i in res:
                path_temp.append(i.id)
            final_path.append(path_temp)
    


def travel_time(from_node, to_node, data):
    """Gets the travel times between two locations."""
    travel_time = (data["distances"][from_node][to_node] / data["vehicle_speed"]) * 60
    return travel_time
  
def get_paths(result):
    all_paths = []
    costs = []
    d_cost = []
    for pathing in result:
        temp = []
        for noding in pathing:
            temp.append(noding.id)
        costs.append(pathing[-1].c)
        d_cost.append(pathing[-1].d)
        all_paths.append(temp)
    return all_paths, costs, d_cost

        
        
def SP1(g, data_dict, dij, discovered_paths):
    '''
    This function uses the SP1 labelling algorithm to calculate all the feasible paths from s to t node
    :g: sagemath graph
    :param n: number of requests
    :source_time: time at source node
    :T: matrix containing time to cross the edges
    :load_per_req: number of passengers per request
    :return: feasible paths from s to t including the shortest
    '''
    
    n = data_dict["num_of_requests"]
    load_per_req = data_dict["load"]
    time_window = data_dict["time_windows"]
    P = set(range(1, n+1))
    D = set(range(n+1, 2*n+1))
    
    
    #initialize variables
    result = []
    U = []
    s = 0
    cost = 0
    V_parent = []
    O_parent = [] 
    parent_node = -1
    source_time = time_window[s][0]
    
    #First node s=0
    L = Label(s, source_time, load_per_req[s], cost, V_parent, O_parent, P, D, parent_node, n, 0) 
    U.append([L]) #U stores all the paths
   
    while U: 
        path = U.pop(0) #U is a queue first in first out!
        L = path[-1] #get the last node of the path
        i = L.id
        #storing the paths reaching the final node 7
        if i == 2*n+1 and L.d < 0:
            current_path , temp_cost , temp_dcost = get_paths([path])
            if current_path[0] not in discovered_paths:
                result.append(path)
                return result
            

        #expand path on every incident edge
        if g.edges_incident(i) != []:
            _, incident_nodes, _ = zip(*g.edges_incident(i))
            for j in range(len(incident_nodes)): #get all incident nodes from i
                current_node = incident_nodes[j]
                cost = data_dict["distances"][i][current_node] + L.c #cummulative cost
                dij_cost = dij[i][current_node] + L.d
                cummulative_load = L.l + load_per_req[current_node]
                t_j = travel_time(i, current_node, data_dict) + L.t
                #check suggested conditions for SP1 to avoid cycles
                if cummulative_load <=3:
                    if 0<current_node and current_node<=n and (current_node not in L.V):
                        L_new = Label(current_node, t_j, cummulative_load, cost, L.V, L.O, P, D, i,n, dij_cost)
                        if check_dominance(U, L_new) == False:
                            new_path = list(path)
                            new_path.append(L_new)
                            U.append(new_path)
                    elif n < current_node and current_node <= 2*n+1 and ((current_node-n) in L.O):
                        L_new = Label(current_node, t_j, cummulative_load, cost, L.V, L.O, P, D, i, n, dij_cost)
                        if check_dominance(U, L_new) == False:
                            new_path = list(path)
                            new_path.append(L_new)
                            U.append(new_path)
                    elif current_node == 2*n+1 and L.O.is_empty():
                        L_new = Label(current_node, t_j, cummulative_load, cost, L.V, L.O, P, D, i,n, dij_cost)
                        if check_dominance(U, L_new) == False:
                            new_path = list(path)
                            new_path.append(L_new)
                            U.append(new_path)
    return result


# ### Master Problem 

# In[86]:


import pickle
from itertools import chain
import time
from sage.all import *

def read_data():
    """
    Reads the graph g (cgraph) and a data in dictionary containing distance matrix, time windows, load per request,
    vehicle speed, num of requests and maximum waiting time. 
    """
    g = load('my_graph.sobj')
    with open("data_file", "r") as file:
        data_dict = pickle.load(file)
    #print(data_dict["distances"])
    #data_dict["distances"]
    return g, data_dict


def create_initial_path(g):
    """
    Creates a initial dummy path containing all nodes and assigns a large cost
    """
    path = []
    c = []
    for v in g.vertices():
        path.append(v)
    c.append(10000)
    return path, c

def calculate_cost_matrix(data_dict, duals):
    """
    Calculates dij of the paper by substituting the dual variables of master problem solution
    """
    
    dist = copy(data_dict["distances"])
    duals = [int(i) for i in duals]
    for col in range(dist.ncols()):
    #data_dict["distances"][:data_dict["num_of_requests"],col] -= vector(duals[:data_dict["num_of_requests"]]) 
        temp_a = vector(chain(*dist[:data_dict["num_of_requests"]+1,col]))
       
        temp_b = -vector(duals[:data_dict["num_of_requests"]+1])
        for i in range(data_dict["num_of_requests"]+1):
            dist[i,col] = map(operator.add, temp_a, temp_b)[i]

    return dist

def solve_master_problem(a, r_cost, vs, set_of_paths):
    """
    :p: MixedIntegerLinearProgram object
    :a: how many times row(node) i appears in column(path) j
    :r_cost: distances matrix used to define the objective function 
    """
    #print("a",a)
  
    #print("vs",vs)
    #print("omega",set_of_paths)
    #create variable
    p =  MixedIntegerLinearProgram(solver = 'GLPK', maximization = False)
    p.solver_parameter("simplex_or_intopt", "simplex_only") 
    p.solver_parameter("primal_v_dual", "GLP_DUAL")
    y = p.new_variable(integer = False, nonnegative=True)
    #create objective function
    
    p.set_objective(p.sum(r_cost[r] * y[r] for r in range(len(set_of_paths))))
    for i in vs:
        p.add_constraint(p.sum(a[i,r] * y[r] for r in range(len(set_of_paths)))==1) 

    #solve the master problem
    p.solve()
    d = p.get_backend()
    duals = []
    for i in vs:
        duals.append(d.get_row_dual(i))
    
    return p.get_values(y), p.get_objective_value(), duals


def create_a(a, set_of_paths, vs):
    """
    Given the current set of paths and the vertices of the graph return A matrix used in the constraints
    """
    a = matrix(len(vs),len(set_of_paths))
    for r in range(len(set_of_paths)):
        a[:,r] = 0
        for i in vs:
            if i in set_of_paths[r]:
                a[i,r] += 1
    return a

def master_problem(data_dict, a, r_cost, set_of_paths):
    """
    Main function of calculating the solution
    """
    vs = g.vertices()
    
    sp1_result = set_of_paths
    
    res_variables, res_obj, duals = solve_master_problem(a, r_cost, vs, set_of_paths)
    while sp1_result != []:
        dij = calculate_cost_matrix(data_dict, duals)
        sp1_result = SP1(g, data_dict, dij, set_of_paths)
       
        new_paths, r_cost_new, d_cost = get_paths(sp1_result)
        set_of_paths.extend(new_paths)
        #print("omega:",set_of_paths)
        r_cost.extend(r_cost_new)
        a = create_a(a, set_of_paths, vs)
        res_variables, res_obj, duals = solve_master_problem(a, r_cost, vs, set_of_paths)
        #print("vars", res_variables)
    
    print(a) 
    return res_variables, res_obj, set_of_paths
        
    
g, data_dict = read_data()
path, r_cost = create_initial_path(g)
set_of_paths = [path]

a = matrix(len(g.vertices()),len(set_of_paths))
a = create_a(a, set_of_paths, g.vertices())

start = time.time()
res_variables, res_obj, omega = master_problem(data_dict, a, r_cost, set_of_paths)
end = time.time()

print(res_variables, res_obj)
print("Path is:")
for key, value in res_variables.items():
    if value != 0:
        print(omega[key])
        print(value)

print("TIME:", end-start)

