import apx
import numpy as np
from importlib import reload
reload(apx)
import picos as pc
from scipy.linalg import cholesky

# filenames
fnames = ['learning-0.6.txt', 'copenhagen-0.5.txt', 'denmark-0.6.txt']

# testing purposes
filename = fnames[2]

# Read vectors and construct matrices
with open(filename, 'r') as f:
  feature_vectors = []
  words = []
  for line in f:
    word, vector = line.split(';')
    words.append(word)
    vector = [ float(x) for x in vector.split(',') ]
    feature_vectors.append(vector)
  n = len(words)
  feature_vectors = np.array(feature_vectors)
  W_plus = np.dot(feature_vectors, np.transpose(feature_vectors))
  np.fill_diagonal(W_plus, 0)
  W_minus = np.ones(shape=(n,n)) * np.average(W_plus)
  np.fill_diagonal(W_minus, 0)
  
  cc = pc.Problem()
  X = pc.SymmetricVariable('X',(n,n))
  weight_plus = pc.Constant('W_plus', W_plus)
  weight_minus = pc.Constant('W_minus', W_minus)
  ones = pc.Constant('1', np.ones((n,n)))
  cc.add_constraint(pc.maindiag(X) == 1) # 1s on the main diagonal
  cc.add_constraint(X >= 0)
  cc.add_constraint(X >> 0) # positive semidefinite
  cc.set_objective('max', 0.5 * pc.trace(weight_plus * X + weight_minus * (ones - X)))
  
  cc.solve(solver='cvxopt')

  V = cholesky(X.value + 1e-3 * np.identity(n))
  
  ex_obj = np.sum(W_plus) / 8 + np.sum(W_minus) * 3/8
  
  print(f"{filename}\nUpper bound from relaxation is {cc.value}")
  print(f"Random 4-clustering has expected obj function {ex_obj}")

  mx = 0
  rnd = []
  repetitions = 1000
  # Perform randomized rounding many times and compute the objective function
  for _ in range(repetitions):
    r1 = np.random.normal(size = n)
    r2 = np.random.normal(size = n)
    rounding = list(zip(np.sign(np.dot(r1, V)), np.sign(np.dot(r2, V))))
    val = 0
    for i in range(n):
      for j in range(n):
        if rounding[i] == rounding[j]:
          val += W_plus[i, j]
        else:
          val += W_minus[i, j]
    val /= 2
    if mx <= val:
        mx = val
        rnd = rounding
    
  clusters = {(-1, -1): [], (-1, 1): [], (1, -1): [], (1, 1): []}
  for w, r in zip(words, rnd):
      clusters[r].append(w)
  
  print(f"Randomized rounding found a 4-clustering with objective function {mx} \n")
  for i, key in zip(range(1, 5), clusters):
      print(f'Cluster {i}:\n {clusters[key]} \n')
