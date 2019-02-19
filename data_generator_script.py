# coding: utf-8

#My graph generator using sagemath
# The maximum load per request is 3 but later I have to allow more passenger per request.

import numpy as np
import pickle

set_random_seed(5)
rnd = current_randstate().python_random()

def travel_time(from_node, to_node):
    """Gets the travel times between two locations."""
    travel_time = data["distances"][from_node][to_node] / data["vehicle_speed"]
    return travel_time


def create_distances_matrix(g, n ,distance_range):
    A = random_matrix(ZZ,2*n+2,2*n+2,x = distance_range[0], y=distance_range[1])
    for i in range(A.nrows()):
        A[i,i] = 0
        A[i,2*n+1] = 0
        
    #set weights of g using distances matrix
    for edge in g.edges():
        u = edge[0]
        v = edge[1]
        i = A[u,v]
        g.set_edge_label(edge[0], edge[1], i)

    #calculate all pairs shortest path 
    dist, pred = g.shortest_path_all_pairs(by_weight=True, algorithm="Dijkstra_Boost")
    
    #create graph G' which is guaranteed to satisfy the triangle inequality
    for edge in g.edges():
        u = edge[0]
        v = edge[1]
        A[u,v] = dist[u][v]
        w = dist[u][v]
        g.set_edge_label(u, v, w)
        
    return g, A
    
def create_time_windows_per_node(g, time_interval, P, D):
    time_windows = []
    time_windows.append([0,24*60])
    time_window_range = 20
    for i in P:
        b = rnd.randrange(time_interval[0], time_interval[1])
        e = b + time_window_range
        time_windows.append([b,e])
    for i in D:
        b = time_windows[i - len(P)][0] + rnd.randrange(time_interval[0], time_interval[1]- time_windows[i - len(D)][1])
        e = b + time_window_range
        time_windows.append([b,e])
            
    time_windows.append([0,24*60])
    
    return time_windows

def create_load(P, D, maximum_load):
    load = [0]
    for i in P:
        load.append(rnd.randrange(1, 3))
    for i in D:
        load.append(-load[i-len(P)])
    load.append(0)
    
    return load

def delete_edges(g, P, D):
    g.delete_edge(2*n+1,0)
    g.delete_edge(0,2*n+1) 
    for i in D:
        g.delete_edge(0 , i)
        g.delete_edge(2*n+1,i)
        g.delete_edge(i,0)

    for i in P:
        g.delete_edge(i, 2*n+1)
        g.delete_edge(2*n+1, i)
        g.delete_edge(i,0)
    return g


#Construct graph using information given 
def construct_graph(n, maximum_load, passenger_maximum_waiting_time, time_window_interval, time_limit, distance_range):
    
    data_dict = {}
    g = graphs.CompleteGraph(2*n+2)
    g = g.to_directed(time_window_interval)

    #delete not needed edges
    P = set(range(1,n+1))
    D = set(range(n+1,2*n+1))

    g = delete_edges(g, P, D)
    g, A = create_distances_matrix(g, n, distance_range)

    data_dict["time_windows"] = create_time_windows_per_node(g, time_window_interval, P, D)
    data_dict["load"]  = create_load(P, D, maximum_load)
    data_dict["num_of_requests"] = n  
    data_dict["vehicle_speed"] = 80
    data_dict["distances"] = A
    data_dict["maximum_waiting_time"] = 60
    
 
    return g, data_dict

#********************************************************************************************************************

#create random graph with positive weights satisfying triangle inequality
#information needed to construct the graph:
#1)number of requests, 2) maximum load of the vehicle, 3) time window interval distance 
#4) maximium route time, 5) range of the distances between nodes to produce
 
n = 10 #number of requests
maximum_load = 3
passenger_maximum_waiting_time = 10 #10 minutes
time_window_interval = [0, 4*60]
time_limit = 8 * 60 # maximum route is 8 hours  
distance_range = [0,100] 

g, data_dict = construct_graph(n, maximum_load, passenger_maximum_waiting_time, time_window_interval, time_limit, distance_range)

#save data
g.save("my_graph")   
with open("data_file", "w") as write_file:
    pickle.dump(data_dict, write_file)

