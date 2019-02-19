
# coding: utf-8

# ##  SP1 Implementation

# In[2]:


import time
import pickle

class Label:
    '''
    This class is used to construct the Labels for the Labelling algorithm used to solve the constrained shortest path problem
    '''
    def __init__(self ,number_of_node, arrival_time, load, cost, V_parent, O_parent, P, D, parent_node, n):
        self.id = number_of_node 
        self.t = arrival_time #arrival time in Label L 
        self.l = load # cummulative load at this node using the a path
        self.c = cost # the cost on this solution is the distance
        self.V = Set(V_parent) # V is the set of requests served by this path that may have already been served
        self.construct_V(V_parent, P, D)
        self.O = Set(O_parent) #O is the set of open requests. Requests that have started but not delivered yet
        self.construct_O(P, D, n)
        self.p = parent_node # previous node in the path
     
        
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
    print(final_path)
    print("min_cost:",min_cost)


def travel_time(from_node, to_node, data):
    """Gets the travel times between two locations."""
    travel_time = (data["distances"][from_node][to_node] / data["vehicle_speed"]) * 60
    return travel_time
  
def display_all_paths(result):
    for pathing in result:
        temp = []
        for noding in pathing:
            temp.append(noding.id)
        print(temp)
    

        
        
def SP1(g, data_dict):
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
    L = Label(s, source_time, load_per_req[s], cost, V_parent, O_parent, P, D, parent_node, n) 
    U.append([L]) #U stores all the paths
   
    while U: 
        path = U.pop(0) #U is a queue first in first out!
        L = path[-1] #get the last node of the path
        i = L.id
        #storing the paths reaching the final node 7
        if i == 2*n+1:
            result.append(path)
            if len(result)>100000:
                return result
            

        #expand path on every incident edge
        if g.edges_incident(i) != []:
            _, incident_nodes, _ = zip(*g.edges_incident(i))
            for j in range(len(incident_nodes)): #get all incident nodes from i
                current_node = incident_nodes[j]
                cost = data_dict["distances"][i][current_node] + L.c #cummulative cost
                cummulative_load = L.l + load_per_req[current_node]
                t_j = travel_time(i, current_node, data_dict) + L.t
                #check suggested conditions for SP1 to avoid cycles
                if cummulative_load <=3:
                    if 0<current_node and current_node<=n and (current_node not in L.V) and t_j <= time_window[current_node][1]                    and (t_j + data_dict["maximum_waiting_time"]) >= time_window[current_node][0]:
                        L_new = Label(current_node, t_j, cummulative_load, cost, L.V, L.O, P, D, i,n)
                        if check_dominance(U, L_new) == False:
                            new_path = list(path)
                            new_path.append(L_new)
                            U.append(new_path)
                    elif n < current_node and current_node <= 2*n+1 and ((current_node-n) in L.O) and t_j <= time_window[current_node][1]                    and (t_j + data_dict["maximum_waiting_time"]) >= time_window[current_node][0]:
                        L_new = Label(current_node, t_j, cummulative_load, cost, L.V, L.O, P, D, i,n)
                        if check_dominance(U, L_new) == False:
                            new_path = list(path)
                            new_path.append(L_new)
                            U.append(new_path)
                    elif current_node == 2*n+1 and L.O.is_empty() and t_j <=  time_window[current_node][1]                    and (t_j + data_dict["maximum_waiting_time"]) >= time_window[current_node][0]:
                        L_new = Label(current_node, t_j, cummulative_load, cost, L.V, L.O, P, D, i,n)
                        if check_dominance(U, L_new) == False:
                            new_path = list(path)
                            new_path.append(L_new)
                            U.append(new_path)
    return result



#load graph
g = load('my_graph.sobj')

#load data_dict
with open("data_file", "r") as file:
    data_dict = pickle.load(file)
data_dict

start = time.time()
result = SP1(g, data_dict)
end = time.time()
#display_all_paths(result)
#display_min_path(result)

#print(end - start)
