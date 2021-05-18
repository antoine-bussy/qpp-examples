#!/usr/bin/env python3

from fractions import Fraction
import matplotlib.pyplot as plt
import pyzx as zx

circuit = zx.Circuit(qubit_amount=3)

circuit.add_gate("HAD", 0)
circuit.add_gate("CNOT", 1, 2)

circuit.add_gate("HAD", 1)

circuit.add_gate("CNOT", 0, 1)

circuit.add_gate("HAD", 1)
circuit.add_gate("HAD", 2)

circuit.add_gate("CNOT", 0, 2)

circuit.add_gate("CNOT", 1, 0)
circuit.add_gate("HAD", 2)

circuit.add_gate("HAD", 1)

circuit.add_gate("CNOT", 2, 1)

circuit.add_gate("CNOT", 0, 2)

circuit.add_gate("HAD", 0)
circuit.add_gate("HAD", 2)

circuit.add_gate("CNOT", 1, 0)

circuit.add_gate("CNOT", 2, 1)

circuit.add_gate("HAD", 1)

circuit.add_gate("CNOT", 1, 0)

circuit.add_gate("HAD", 0)
circuit.add_gate("HAD", 1)
circuit.add_gate("HAD", 2)

circuit.add_gate("CNOT", 0, 1)

circuit.add_gate("CNOT", 2, 0)

circuit.add_gate("HAD", 0)
circuit.add_gate("HAD", 1)

circuit.add_gate("CNOT", 2, 1)

print(circuit.stats())
zx.draw_matplotlib(circuit).savefig("circuit.png")

circuit2 = zx.optimize.full_optimize(circuit.split_phase_gates())
g = circuit2.to_graph()

zx.simplify.full_reduce(g)
g.normalize()
zx.draw_matplotlib(g).savefig("graph_opt.png")

c_opt = zx.extract.extract_circuit(g.copy())
c_opt2 = zx.optimize.full_optimize(c_opt.split_phase_gates())

print(c_opt2.stats())
zx.draw_matplotlib(c_opt2).savefig("simplified.png")
