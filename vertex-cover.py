# Fetch and import APX wrapper class
# !wget -q https://raw.githubusercontent.com/rasmus-pagh/apx/main/apx.py -O apx.py

import apx
from importlib import reload
reload(apx)
from apx import DataFile, LinearProgram, np
import matplotlib.pyplot as plt

def list_triangles(graph):
    m = 0
    deg = {}
    edges = set()
    for (u, v) in graph:
        m += 1
        edges.add((u, v))
        edges.add((v, u))
        if u in deg:
            deg[u] += 1
        else:
            deg[u] = 1
        if v in deg:
            deg[u] += 1
        else:
            deg[u] = 1
    
    heavy = set()
    light = {}
    thr = np.sqrt(m) 
    for u in deg:
        if deg[u] > thr:
            heavy.add(u)
        else:
            light[u] = set()
    for (u, v) in edges:
        if u in light:
            light[u].add(v)
        if v in light:
            light[v].add(u)
    
    triangles = set()
    # detect heavy triangle
    for u in heavy:
        for v in heavy:
            for z in heavy:
                if (u, v) in edges and (v, z) in edges and (z, u) in edges:
                    if u < v and v < z:
                        triangles.add((u, v, z))
    # detect light triangles
    for (u, v) in edges:
        if u in light:
            for z in light[u]:
                if (z, v) in edges:
                    triangles.add(tuple(sorted((u, v, z))))
        if v in light:
            for z in light[v]:
                if (z, u) in edges:
                    triangles.add(tuple(sorted((u, v, z))))
    return triangles
    

def run_experiment(triangle_constraint=False):
    lp_sol, r_sol = [], []
    for filename in DataFile.graph_files:
        graph = list(DataFile(filename))
        vertex_cover_lp = LinearProgram('min')
        objective = {}
        
        if triangle_constraint:
            triangles = list_triangles(graph)
            for (x, y, z) in triangles:
                vertex_cover_lp.add_constraint({x: 1, y: 1, z: 1}, 2)
                
        for (u,v) in graph:
            vertex_cover_lp.add_constraint({u: 1, v: 1}, 1)
            objective[u] = 1.0
            objective[v] = 1.0
        
        print(filename, len(graph))
        vertex_cover_lp.set_objective(objective)
        value, solution = vertex_cover_lp.solve()
        rounded_value, rounded_solution = 0, {}
        for x in solution:
            r = int(np.round(solution[x] + 1e-10)) 
            # Add small constant to deal with numerical issues for numbers close to 1/2
            rounded_solution[x] = r
            rounded_value += r
        lp_sol.append(value)
        r_sol.append(rounded_value)
    return lp_sol, r_sol   
 
lp_sol_1, r_sol_1 = run_experiment()
lp_sol_2, r_sol_2 = run_experiment(triangle_constraint=True)


legend_str = [x.replace('.txt', '').upper() for x in DataFile.graph_files]

lp_1 = np.array(lp_sol_1) / np.array(lp_sol_1)
lp_2 = np.array(lp_sol_2) / np.array(lp_sol_1)
r_1 = np.array(r_sol_1) / np.array(lp_sol_1)
r_2 = np.array(r_sol_2) / np.array(lp_sol_1)

lp_1 = np.append(lp_1, lp_1[0])
lp_2 = np.append(lp_2, lp_2[0])
r_1 = np.append(r_1, r_1[0])
r_2 = np.append(r_2, r_2[0])

# Initialise the spider plot by setting figure size and polar projection
plt.figure(figsize=(12, 12))
plt.subplot(polar=True)

theta = np.linspace(0, 2 * np.pi, len(legend_str) + 1)
lines, labels = plt.thetagrids(
    range(0, 360, int(360 / len(legend_str))), (legend_str))

plt.plot(theta, lp_1, 'b', linewidth=2)
plt.plot(theta, lp_2, 'r-', linewidth=2)
plt.plot(theta, r_1, 'g-', linewidth=2)
plt.plot(theta, r_2, 'k-', linewidth=2)
plt.fill(theta, lp_1, 'b', alpha=0.1)
plt.ylim(0,2.2)

plt.legend(['LP', 'LP with triangles', 'Rounding', 'Rounding with triangles'], prop={"size":14}, frameon=True, loc=4)
plt.show()

 