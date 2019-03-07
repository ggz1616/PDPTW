
# coding: utf-8

# In[37]:

import pandas as pd
import pickle
from itertools import chain
import time
from sage.all import *
import data_generator as data_gen
import sys


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
        

class SP1:
    def __init__(self, g, data_dict):
        self.distance_matrix = data_dict["distances"]
        self.n = data_dict["num_of_requests"]
        self.load_per_req = data_dict["load"]
        self.time_window = data_dict["time_windows"]
        self.speed = data_dict["vehicle_speed"]
        
    def check_dominance(self,U, L):
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



    def travel_time(self, from_node, to_node):
        """Gets the travel times between two locations."""
        travel_time = (self.distance_matrix[from_node][to_node] / self.speed) * 60
        return travel_time

    def get_paths(self, result):
        all_paths = []
        costs = []
        d_cost = []
        for path in result:
            temp = []
            for node in path:
                temp.append(node.id)
            costs.append(path[-1].c)
            d_cost.append(path[-1].d)
            all_paths.append(temp)
        return all_paths, costs, d_cost

    def calculate_H1_old(self, dij, discovered_paths, num_arcs_to_consider, req_not_sat):
            '''
            This function uses the SP1 labelling algorithm to calculate all the feasible paths from s to t node
            :g: sagemath graph
            :param n: number of requests
            :source_time: time at source node
            :T: matrix containing time to cross the edges
            :load_per_req: number of passengers per request
            :return: feasible paths from s to t including the shortest
            '''

            time_window = data_dict["time_windows"]
            P = set(range(1, self.n+1))
            D = set(range(self.n+1, 2 * self.n+1))


            #initialize variables
            result = []
            U = []
            s = 0
            cost = 0
            V_parent = []
            O_parent = [] 
            parent_node = -1
            source_time = self.time_window[s][0]

            #First node s=0
            L = Label(s, source_time, self.load_per_req[s], cost, V_parent, O_parent, P, D, parent_node, self.n, 0) 
            U.append([L]) #U stores all the paths
            print(discovered_paths)
            while U: 
                path = U.pop(0) #U is a queue first in first out!
                L = path[-1] #get the last node of the path
                i = L.id
                #storing the paths reaching the final node 7
                if i == 2*self.n+1:
                    if L.d < 0:
                        current_path , temp_cost , temp_dcost = self.get_paths([path])
                        if current_path[0] not in discovered_paths:
                            result.append(path)
                            return result
                    elif L.d >= 0 and result ==[]:
                        num_arcs_to_consider += 5 
                        

                #expand path on every incident edge
                if g.edges_incident(i) != []:
                    #print(g.edges_incident(i))
                    
                    _, incident_nodes, _ = zip(*g.edges_incident(i))
                    d_sorted = dij[i,incident_nodes].numpy()[0]
                    sorted_y_idx_list = sorted(range(len(d_sorted)),key=lambda x:d_sorted[x])
                    incident_nodes_sorted = [incident_nodes[i] for i in sorted_y_idx_list]
                    print(incident_nodes)
                    print("sorted",incident_nodes_sorted)
                    #iter_set = incident_nodes_sorted
                    '''
                    if  len(req_not_sat) > 0:
                        n_arcs = num_arcs_to_consider + len(req_not_sat)
                        iter_set = incident_nodes_sorted + list(req_not_sat.intersection(set(incident_nodes)))
                        print("iter",iter_set)
                    '''
                    print(incident_nodes_sorted)
                    for j in range(len(incident_nodes)): #range(min(num_arcs_to_consider,len(incident_nodes_sorted))): #get all incident nodes from i
                        current_node = incident_nodes[j]
                        cost = self.distance_matrix[i][current_node] + L.c #cummulative cost
                        dij_cost = dij[i][current_node] + L.d
                        cummulative_load = L.l + self.load_per_req[current_node]
                        t_j = self.travel_time(i, current_node) + L.t
                        #check suggested conditions for SP1 to avoid cycles
                        if cummulative_load <= 3:
                            if 0<current_node and current_node<=n and (current_node not in L.V):
                                L_new = Label(current_node, t_j, cummulative_load, cost, L.V, L.O, P, D, i, self.n, dij_cost)
                                if self.check_dominance(U, L_new) == False:
                                    new_path = list(path)
                                    new_path.append(L_new)
                                    U.append(new_path)
                            elif self.n < current_node and current_node <= 2*self.n+1 and ((current_node-self.n) in L.O):
                                L_new = Label(current_node, t_j, cummulative_load, cost, L.V, L.O, P, D, i, self.n, dij_cost)
                                if self.check_dominance(U, L_new) == False:
                                    new_path = list(path)
                                    new_path.append(L_new)
                                    U.append(new_path)
                            elif current_node == 2*self.n+1 and L.O.is_empty():
                                L_new = Label(current_node, t_j, cummulative_load, cost, L.V, L.O, P, D, i, self.n, dij_cost)
                                if self.check_dominance(U, L_new) == False:
                                    new_path = list(path)
                                    new_path.append(L_new)
                                    U.append(new_path)
            return result

    def calculate_H1(self, dij, discovered_paths, num_arcs_to_consider,  paths_per_itter = 1):
        time_window = data_dict["time_windows"]
        P = set(range(1, self.n+1))
        D = set(range(self.n+1, 2 * self.n+1))
        #initialize variables
        result = []
        r_num = 0
        U = []
        s = 0
        cost = 0
        V_parent = []
        O_parent = [] 
        parent_node = -1
        source_time = self.time_window[s][0]

        #First node s=0
        L = Label(s, source_time, self.load_per_req[s], cost, V_parent, O_parent, P, D, parent_node, self.n, 0) 
        U.append([L]) #U stores all the paths

        while U: 
            path = U.pop(0) #U is a queue first in first out!
            L = path[-1] #get the last node of the path
            i = L.id
            #storing the paths reaching the final node 7
            
       
            temp = []
            for node in path:
                temp.append(node.id)
                
            
            if i == 2*self.n+1 and L.d < 0:
                current_path , temp_cost , temp_dcost = self.get_paths([path])
                if current_path[0] not in discovered_paths:
                    result.append(path)
                    r_num += 1
                    if r_num >= paths_per_itter:
                        return result
            
          
            #expand path on every incident edge
            if g.edges_incident(i) != []:
                pickup_inc = []
                delivery_inc = []
                _, incident_nodes, _ = zip(*g.edges_incident(i))
                d_sorted = dij[i,incident_nodes].numpy()[0]
                sorted_y_idx_list = sorted(range(len(d_sorted)),key=lambda x:d_sorted[x])
                incident_nodes_sorted = tuple(incident_nodes[i] for i in sorted_y_idx_list)
                
                for iterator1 in range(len(incident_nodes_sorted)):
                    if incident_nodes_sorted[iterator1] in P and len(pickup_inc) < num_arcs_to_consider and incident_nodes_sorted[iterator1] not in temp:
                        pickup_inc.append(incident_nodes_sorted[iterator1])
                    elif ((incident_nodes_sorted[iterator1] in D) or incident_nodes_sorted[iterator1]== 2* self.n + 1) and (len(delivery_inc) < num_arcs_to_consider) and incident_nodes_sorted[iterator1] not in temp:
                        delivery_inc.append(incident_nodes_sorted[iterator1])
                    if len(delivery_inc)== num_arcs_to_consider and len(pickup_inc)== num_arcs_to_consider:
                            break
                inc_nodes  = pickup_inc + delivery_inc
                 
                for j in range(len(inc_nodes)): #get all incident nodes from i
                    current_node = inc_nodes[j]
                    cost = self.distance_matrix[i][current_node] + L.c #cummulative cost
                    dij_cost = dij[i][current_node] + L.d
                    
                    cummulative_load = L.l + self.load_per_req[current_node]
                    t_j = self.travel_time(i, current_node) + L.t
                    #check suggested conditions for SP1 to avoid cycles
                    if cummulative_load <=3:
                        if 0<current_node and current_node <= self.n and (current_node not in L.V):
                            L_new = Label(current_node, t_j, cummulative_load, cost, L.V, L.O, P, D, i, self.n, dij_cost)
                            if self.check_dominance(U, L_new) == False:
                                new_path = list(path)
                                new_path.append(L_new)
                                U.append(new_path)
                        elif self.n < current_node and current_node <= 2*self.n+1 and ((current_node-self.n) in L.O):
                            L_new = Label(current_node, t_j, cummulative_load, cost, L.V, L.O, P, D, i, self.n, dij_cost)
                            if self.check_dominance(U, L_new) == False:
                                new_path = list(path)
                                new_path.append(L_new)
                                U.append(new_path)
                        elif current_node == 2*self.n+1 and L.O.is_empty():
                            L_new = Label(current_node, t_j, cummulative_load, cost, L.V, L.O, P, D, i, self.n, dij_cost)
                            if self.check_dominance(U, L_new) == False:
                                new_path = list(path)
                                new_path.append(L_new)
                                U.append(new_path)
        
        return result

            
            
    def calculate_SP1(self, dij, discovered_paths, paths_per_itter = 1):
        '''
        This function uses the SP1 labelling algorithm to calculate all the feasible paths from s to t node
        :g: sagemath graph
        :param n: number of requests
        :source_time: time at source node
        :T: matrix containing time to cross the edges
        :load_per_req: number of passengers per request
        :return: feasible paths from s to t including the shortest
        '''

        time_window = data_dict["time_windows"]
        P = set(range(1, self.n+1))
        D = set(range(self.n+1, 2 * self.n+1))


        #initialize variables
        result = []
        r_num = 0
        U = []
        s = 0
        cost = 0
        V_parent = []
        O_parent = [] 
        parent_node = -1
        source_time = self.time_window[s][0]

        #First node s=0
        L = Label(s, source_time, self.load_per_req[s], cost, V_parent, O_parent, P, D, parent_node, self.n, 0) 
        U.append([L]) #U stores all the paths

        while U: 
            path = U.pop(0) #U is a queue first in first out!
            L = path[-1] #get the last node of the path
            i = L.id
            #storing the paths reaching the final node 7
            if i == 2*self.n+1 and L.d < 0:
                current_path , temp_cost , temp_dcost = self.get_paths([path])
                if current_path[0] not in discovered_paths:
                    result.append(path)
                    r_num += 1
                    if r_num >= paths_per_itter:
                        return result


            #expand path on every incident edge
            if g.edges_incident(i) != []:
                _, incident_nodes, _ = zip(*g.edges_incident(i))
                for j in range(len(incident_nodes)): #get all incident nodes from i
                    current_node = incident_nodes[j]
                    cost = self.distance_matrix[i][current_node] + L.c #cummulative cost
                    dij_cost = dij[i][current_node] + L.d
                    cummulative_load = L.l + self.load_per_req[current_node]
                    t_j = self.travel_time(i, current_node) + L.t
                    #check suggested conditions for SP1 to avoid cycles
                    if cummulative_load <=3:
                        if 0<current_node and current_node <= self.n and (current_node not in L.V):
                            L_new = Label(current_node, t_j, cummulative_load, cost, L.V, L.O, P, D, i, self.n, dij_cost)
                            if self.check_dominance(U, L_new) == False:
                                new_path = list(path)
                                new_path.append(L_new)
                                U.append(new_path)
                        elif self.n < current_node and current_node <= 2*self.n+1 and ((current_node-self.n) in L.O):
                            L_new = Label(current_node, t_j, cummulative_load, cost, L.V, L.O, P, D, i, self.n, dij_cost)
                            if self.check_dominance(U, L_new) == False:
                                new_path = list(path)
                                new_path.append(L_new)
                                U.append(new_path)
                        elif current_node == 2*self.n+1 and L.O.is_empty():
                            L_new = Label(current_node, t_j, cummulative_load, cost, L.V, L.O, P, D, i, self.n, dij_cost)
                            if self.check_dominance(U, L_new) == False:
                                new_path = list(path)
                                new_path.append(L_new)
                                U.append(new_path)
        return result


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

def save_data(data, url):
    with open(url, "wb") as file:
        data_dict = pickle.dump(data,file)
     

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
    p =  MixedIntegerLinearProgram(maximization = False)
    p.solver_parameter("simplex_or_intopt", "simplex_only") 
    #p.solver_parameter("primal_v_dual", "GLP_DUAL")
    y = p.new_variable(integer = False, nonnegative=True)
    #create objective function
    
    p.set_objective(p.sum(r_cost[r] * y[r] for r in range(len(set_of_paths))))
    vs_size = len(vs)
    for i in vs[1:vs_size-1]:
        if i != 0 and i != len(vs)-1:
            p.add_constraint(p.sum(a[i,r] * y[r] for r in range(len(set_of_paths)))==1) 

    #solve the master problem
    p.solve()
    d = p.get_backend()
    
    duals = []
    for i in vs[:vs_size-2]:
        duals.append(d.get_row_dual(i))
    #print(d.is_variable_binary(0))
    duals.insert(0, 0)
    duals.append(0)
    #print("duals",duals)
    #p.show()
    #print(set_of_paths)
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

def master_problem(g, data_dict, a, r_cost, set_of_paths, paths_per_itter, num_arcs_to_consider):
    """
    Main function of calculating the solution
    """
    sp1_object = SP1(g, data_dict)
    vs = g.vertices()
   
    sp1_result = set_of_paths
    
    res_variables, res_obj, duals = solve_master_problem(a, r_cost, vs, set_of_paths)
    iter = 0
    while sp1_result != [] or res_obj ==10000: #or len(set(vs)-set(requests_satisfied)) != 0: #resolves until all requests are satisfied
        iter += 1
        requests_satisfied = []
        dij = calculate_cost_matrix(data_dict, duals)
        #sp1_result = sp1_object.calculate_SP1(dij, set_of_paths, paths_per_itter)
        
        sp1_result = sp1_object.calculate_H1(dij, set_of_paths, num_arcs_to_consider, paths_per_itter)
        #print(sp1_result) 
        if sp1_result == [] and res_obj == 10000:
            num_arcs_to_consider +=  num_arcs_to_consider//2
            print(num_arcs_to_consider)
                      
        #sp1_result = sp1_object.calculate_H1_old(dij, set_of_paths, num_arcs_to_consider, set(vs)-set(requests_satisfied))
        #if len(set(vs)-set(requests_satisfied)) == g.order() and sp1_result == []:
        #    num_arcs_to_consider = num_arcs_to_consider + 5
        #    print("here")
        if sp1_result != []:
            new_paths, r_cost_new, d_cost = sp1_object.get_paths(sp1_result)
            set_of_paths.extend(new_paths)
            r_cost.extend(r_cost_new)
            a = create_a(a, set_of_paths, vs)
        res_variables, res_obj, duals = solve_master_problem(a, r_cost, vs, set_of_paths)
        #after solving the master problem we check which requests are satisfied
        #for k, v in res_variables.items():    # for name, age in dictionary.iteritems():  (for Python 2.x)
        #    if v != 0 and k != 0:
        #        requests_satisfied.extend(set_of_paths[k])
          
        #print("vars", res_variables)
        #print(set_of_paths)
    #print("r_cost",r_cost)
    #print(dij)
    #print("obj",res_obj)
    return res_variables, res_obj, set_of_paths
        
    
        
if __name__ == "__main__":
#g, data_dict = read_data()
    mean_time = 0
    n = 7 
    num_of_arcs_to_use_list = [n//2,n]
    path_batch = [1]
    results_dict = {"id":[], "num_of_arcs_used":[], "path":[], "num_of_requests":[], "time":[], "cost":[], "num_of_paths":[], "omega_size":[], "variables":[],"frac_sol":[],"method":[],"distance_matrix":[]}
    for num_of_arcs_to_use in num_of_arcs_to_use_list:
        for batch in path_batch:
            id = 0
            for i in range(10):
                path_list = []
                id += 1
                n += 0
                print("****************")
                
                print(n)
                results_dict["num_of_requests"].append(n)
                g, data_dict = data_gen.generate_graph_main(n, seed_num = i)   
                path, r_cost = create_initial_path(g)
                set_of_paths = [path]

                a = matrix(g.order(),len(set_of_paths))
                a = create_a(a, set_of_paths, g.vertices())

                start = time.time()
                res_variables, res_obj, omega = master_problem(g, data_dict, a, r_cost, set_of_paths, batch, num_of_arcs_to_use)
                end = time.time()
                results_dict["id"].append(id)
                results_dict["time"].append(end-start)
                results_dict["cost"].append(res_obj)
                results_dict["omega_size"].append(len(omega))
                results_dict["variables"].append(res_variables)
                results_dict["method"].append(batch)
                results_dict["distance_matrix"].append(data_dict["distances"])
                results_dict["num_of_arcs_used"].append(num_of_arcs_to_use)
                print(res_variables, res_obj)

                print("Path is:")
                ones = 0
                frac = 0
                for key, value in res_variables.items():
                    if value != 0:
                        ones += 1
                        print(omega[key])
                        path_list.append(omega[key])
                        print(value)
                        if value != 1.0:
                            frac += 1      
                results_dict["path"].append(path_list)
                results_dict["num_of_paths"].append(ones)
                results_dict["frac_sol"].append(frac)
                mean_time += end-start
    res_pd = pd.DataFrame.from_dict(results_dict)
    save_data(res_pd,url = "C:/Users/nataz/Desktop/Github_repositories/PDPTW/res_h1")
    print(res_pd)
'''
#generate graph
#n = 3 #number of requests
#maximum_load = 3
#passenger_maximum_waiting_time = 10 #10 minutes
#time_window_interval = [0, 4*60]
#time_limit = 8 * 60 # maximum route is 8 hours  
#distance_range = [0,100] 


g, data_dict = read_data()
path, r_cost = create_initial_path(g)
set_of_paths = [path]
#set_of_paths = [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 4, 7], [0, 2, 5, 7], 
#                [0, 1, 4, 2, 5, 7], [0, 3, 6, 7], [0, 1, 4, 3, 6, 7], 
#               [0, 2, 5, 3, 6, 7], [0, 1, 4, 2, 5, 3, 6, 7]]
#r_cost = [10000, 3, 3, 105, 3, 105, 105, 207]

a = matrix(g.order(),len(set_of_paths))
a = create_a(a, set_of_paths, g.vertices())

#y, val, dual = solve_master_problem(a, r_cost, g.vertices(), set_of_paths)
#print(y)
#print(val)

start = time.time()
res_variables, res_obj, omega = master_problem(data_dict, a, r_cost, set_of_paths, g)
end = time.time()

print(res_variables, res_obj)
print("Path is:")
for key, value in res_variables.items():
    if value != 0:
        print(omega[key])
        print(value)

print("TIME:", end-start)
'''
