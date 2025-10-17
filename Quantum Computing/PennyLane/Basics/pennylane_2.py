import pennylane as pl
from pennylane import numpy as np
import matplotlib.pyplot as plt


def circuit():
    # Call pl.Hadamard and pl.CNOT to build and return this circuit
    pl.Hadamard(wires=0)
    pl.CNOT(wires=[0,1])
    pl.Hadamard(wires=0)
    return pl.state()

print(pl.draw(circuit)())

def circuit2(alpha,beta,gamma):
    # Call pl.RX, pl.RY, pl.RZ, and pl.CNOT to build and return this circuit
    pl.RX(alpha, wires=0)
    pl.RY(beta, wires=1)
    pl.RZ(gamma, wires=2)
    pl.CNOT(wires=[0, 1])
    pl.CNOT(wires=[1, 2])
    pl.CNOT(wires=[2, 0])
    return pl.state()

#pl.drawer.use_style("black_white")
#pl.draw_mpl(circuit2)(np.pi/4, np.pi/8, 0)
#plt.show()
#print(pl.draw(circuit2)(np.pi/4, np.pi/8, 0))


device = pl.device('default.qubit', wires=['aux', 'q1', 'q2'], shots=[1,10,100,1000])
def circuit(alpha, beta):
    # Build and return this circuit, using pl.expval and pl.PauliZ to measure qubit 2 in the Z basis
    pl.RZ(alpha, wires='aux')
    pl.RY(beta, wires='q1')
    pl.CNOT(wires=['aux', 'q1'])
    pl.CNOT(wires=['q1', 'q2'])
    return pl.expval(pl.PauliZ('q2'))

#qnode = pl.QNode(circuit,device)
#print(qnode(np.pi/4, np.pi/4))


device = pl.device('default.qubit', wires=3)
@pl.qnode(device)
def circuit(x, y):
    pl.Toffoli(wires=[0,1,2])
    pl.CNOT(wires=[1,0])
    pl.RX(np.pi/4, wires=2)
    return pl.state()
#pl.draw_mpl(circuit)(0,0)
#print(pl.specs(circuit)(0,0)['resources'])

@pl.compile
@pl.qnode(device)
def circuit(x, y):
    pl.Toffoli(wires=[0, 1, 2])
    pl.CNOT(wires=[1, 0])
    pl.RX(x, wires=2)
    pl.RY(y, wires=1)
    return pl.state()

"""pl.draw_mpl(circuit)(0,0)
plt.show()
print(pl.specs(circuit)(np.pi/4,np.pi/2)['resources'])"""


# use pl.StatePrep to prepare wire S in the given state
def state_preparation (state):
    pl.StatePrep(state, wires=0, normalize=True)

# use pl.Hadamard and pl.CNOT to create a Bell pair
def entangle_qubits ():
    pl.Hadamard(wires=1)
    pl.CNOT(wires=[1, 2])

# use pl.Hadamard and pl.CNOT to rotate the basis
def basis_rotation ():
    pl.CNOT(wires=[0, 1])
    pl.Hadamard(wires=0)

# use pl.measure for Alice's measurement and pl.PauliX, pl.PauliZ, and pl.cond for Bob's correction
def measure_and_update ():
    m_0 = pl.measure(0)
    m_1 = pl.measure(1)
    pl.cond(m_1 == 0 and m_0 == 1, pl.PauliX)(wires=2)
    pl.cond(m_0 == 0 and m_1 == 1, pl.PauliZ)(wires=2)



def teleport(state):
    state_preparation (state)
    entangle_qubits ()
    basis_rotation ()
    measure_and_update ()

state = np.array([1 / np.sqrt(2) + 0.3j, 0.4 - 0.5j])
_ = pl.draw_mpl(teleport)(state)
#plt.show()
