import pennylane as pl
from pennylane import numpy as np


def normalise(alpha, beta):
    """Given complex amplitudes alpha and beta for the |0> and |1> states,
    return a vector (np.array[complex]) of size 2 for the normalised state"""
    # Compute vector psi [a,b] based on alpha and beta such that |a|^2+|b|^2=1
    norm = np.sqrt(np.abs(alpha) ** 2 + np.abs(beta) ** 2)
    psi = np.array([alpha/norm, beta/norm])
    return psi

print(normalise(3, 4j))


def innerproduct(phi, psi):
    """Compute the (complex) inner product between two normalised states (np.array[complex]) phi and psi"""
    # Compute the inner product of phi and psi
    z = np.conjugate(phi).dot(psi)
    return z

ket_0 = np.array([1, 0])
ket_1 = np.array([0, 1])

print(f"<0|0> = {innerproduct(ket_0, ket_0)}")
print(f"<0|1> = {innerproduct(ket_0, ket_1)}")
print(f"<1|0> = {innerproduct(ket_1, ket_0)}")
print(f"<1|1> = {innerproduct(ket_1, ket_1)}")



def measure_state(psi, n):
    """Simulate n quantum measurements of state psi, returning n samples 0 or 1"""
    # Compute the measurement outcome probabilities
    # Return a list of sample measurement outcomes
    # Hint: use numpy.random.choice
    # works for qubits only in comp basis
    p = np.abs(psi[0]) ** 2
    outcomes = [ np.random.choice([0,1], p = [p, 1-p]) for i in range(n) ]
    return outcomes

psi = normalise(1,1j)
print(measure_state(psi, 20))


def apply_unitary(U, psi):
    #Apply a unitary operation U to state psi
    # Apply U to psi and return the result
    phi = np.dot(U, psi)
    return phi

U = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
print(f"H|0> = {apply_unitary(U, ket_0) * np.sqrt(2)}")
print(f"H|1> = {apply_unitary(U, ket_1) * np.sqrt(2)}")


def quantum_simulator(U, psi):
    #Use previous exercises to sample the result of applying gate U to state psi 100 times
    statistics = measure_state(apply_unitary(U, psi), 100)
    return statistics

U = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
print(quantum_simulator(U, ket_0))

