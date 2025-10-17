import pennylane as pl
from pennylane import numpy as np
import matplotlib.pyplot as plt

n_bits = 4
dev = pl.device("default.qubit", wires=n_bits)
@pl.qnode(dev)
def uniformsuperposition():
    pl.Hadamard(wires=0)
    pl.Hadamard(wires=1)
    pl.Hadamard(wires=2)
    pl.Hadamard(wires=3)
    return pl
#pl.drawer.use_style("black_white")
#pl.draw_mpl(uniformsuperposition)();
#plt.show()

def oracle_matrix(key):
    """Create the unitary matrix corresponding to the binary key (list[int])
    e.g. key=[0,1,1] should give the diagonal matrix [1,1,1,-1,1,1,1,1]"""
    matrix = np.identity(2**len(key))
    index = np.ravel_multi_index(key, [2]*len(key))
    matrix[index][index] = -1
    return matrix

#print(oracle_matrix([0,1,1,0]))


@pl.qnode(dev)
def oracle_circuit(key):
    for i in range(n_bits):
        pl.Hadamard(wires=[i])
    pl.QubitUnitary(oracle_matrix(key), wires=[0,1,2,3])
    return pl.probs(wires=range(n_bits))
#pl.draw_mpl(oracle_circuit)([0,1,1,0])
#plt.show()

#print(oracle_circuit([0,1,1,0]))

dev = pl.device("default.qubit", wires=n_bits)
@pl.qnode(dev)
def pair_circuit(probe, key):
    """Test whether probe (list[int]) contains a solution to key (list[int])"""
    pl.PauliX(wires=[0])
    pl.Hadamard(wires=[n_bits-1])
    pl.QubitUnitary(oracle_matrix(key), wires=list(range(n_bits)))
    pl.Hadamard(wires=[n_bits-1])
    return pl.probs(wires=n_bits-1)
"""pl.draw_mpl(pair_circuit)([1,0,0,1],[0,1,1,1]);
plt.show()
print(pair_circuit([0,1,1,1],[0,1,1,1]))
"""



secretkey = [0,1,0,1]

def pair_lock_picker(trials):
    keystrings = [np.binary_repr(n, n_bits-1) for n in range(2**(n_bits-1))]
    keys = [[int(s) for s in keystring] for keystring in keystrings]
    testnumbers = []
    for trial in range(trials):
        counter = 0
        for key in keys:
            counter += 1
            if np.isclose(pair_circuit(key, secretkey)[1], 1):
                break
        testnumbers.append(counter)
    return sum(testnumbers)/trials
trials = 500
output = pair_lock_picker(trials)
#print(f"For {n_bits} bits, it takes", output, "pair tests on average.")


"""Deutsch Jozsa"""
def oracle_matrix(keys):
    """Create the unitary matrix corresponding to the binary key (list[int])
    e.g. key=[0,1,1] should give the diagonal matrix [1,1,1,-1,1,1,1,1]"""
    # Hint: use np.ravel_multi_index
    matrix = np.identity(2 ** n_bits)
    index = [np.ravel_multi_index(key, [2] * len(keys)) for key in keys]
    for i in range(len(keys)):
        matrix[index][index] = -1
    return matrix
@pl.qnode(dev)
def deutschjozsa(keys):
    """Build the Deutsch-Jozsa circuit"""
    pl.broadcast(pl.Hadamard, wires=list(range(n_bits)), pattern="single")
    pl.QubitUnitary(oracle_matrix(keys), wires=list(range(n_bits)))
    pl.broadcast(pl.Hadamard, wires=list(range(n_bits)), pattern="single")
    return pl.probs(wires=range(n_bits))
keys = ([[0,0,0,0],[0,0,0,1],[1,0,0,0],[0,1,0,0],[1,1,0,0],[0,0,1,0],[0,0,1,1],[0,1,1,0]])
if np.isclose(deutschjozsa(keys)[0],0):
    print("balanced")
else:
    print("constant")

pl.draw_mpl(deutschjozsa(keys)[0], 0)
plt.draw()