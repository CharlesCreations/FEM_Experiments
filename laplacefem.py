# Credit to Jay Wang's book Computational Modeling and visualization of physical systems with python for the core for this code. Small additional modifications made by myself.

from scipy.linalg import solve
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt, numpy as np

def mesh(L,N):
    """Generate mesh"""
    elm, bv = [], []    # elements: [[n1,n2,n3...]], boundry value
    x, y = np.linspace(0,L,N+1), np.linspace(0,L,N+1)   # produce evenly divided intervals for mesh
    ndn = np.arange((N+1)*(N+1)).reshape(N+1,N+1)       # node number
    node = [[xi, yj] for xi in x for yj in y]
    for j in range(N):
        for i in range(N):
            elm.append([ndn[i,j], ndn[i+1,j+1], ndn[i,j+1]]) # upper
            elm.append([ndn[i,j], ndn[i+1,j], ndn[i+1,j+1]]) # lower

    ip = ndn[1:-1,1:-1].flatten()      # internal nodes
    bp = np.delete(ndn, ip)
    for p in bp:
        bv.append((node[p][0]*node[p][1])**2)   # boundary nodes
    return node, elm, bp, ip, bv, x, y

def fem_mat(node, elm):
    """Fills matrix Aij"""
    A = np.zeros((len(node),len(node)))
    for e in elm:
        (x1,y1), (x2,y2), (x3,y3) = node[e[0]], node[e[1]], node[e[2]]
        beta, gama = [y2-y3, y3-y1, y1-y2], [x3-x2, x1-x3, x2-x1]
        ar = 2*(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
        for i in range(3):
            for j in range(i,3):
                A[e[i], e[j]] += (beta[i]*beta[j] + gama[i]*gama[j])/ar
                if (i != j):
                    A[e[j],e[i]] = A[e[i],e[j]] # symmetry
    return A

print("Starting...")
L, N = 1.0, 40                              # length of square, number of intervals
node, elm, bp, ip, bv, x, y = mesh(L,N)     # generate mesh
print(node, elm, bp, ip, bv, x, y)
ip.sort()                                   # sort ip, just in case
A, b = fem_mat(node,elm), np.zeros(len(ip)) # build matricies
print(A)
for j in range(len(ip)):
    b[j] = np.dot(A[ip[j], bp], bv)          # boundary conditions Eq. (7.23)

A = np.delete(A, bp, axis=0)    #delete rows specified by bp
A = np.delete(A, bp, axis=1)    #delete rows specified by bp
u = solve(A, -b)                #solve

u = np.concatenate((u, bv))     #combine internal and boundary values
all = np.concatenate((ip, bp))  # internal + boundary nodes
idx = np.argsort(all)           #index sort nodes
u = np.take(u,idx)              # now u[n] is the value at node n
u = np.reshape(u, (N+1, N+1))   #reshape grid for graphing
x, y = np.meshgrid(x,y)

plt.figure()
ax = plt.subplot(111, projection='3d')
ax.plot_surface(x,y,u, rstride=1, cstride=1, linewidth=0)
ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('V')
plt.figure()
plt.subplot(111, aspect='equal')
plt.contour(x, y, u, 26)
plt.show()
print("Done")

