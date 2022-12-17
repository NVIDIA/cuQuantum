# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import networkx as nx
import numpy as np

from .benchmark import Benchmark


class QAOA(Benchmark):

    # Example instantiation of QAOA circuit for MaxCut paramterized by nqubits
    @staticmethod
    def generateGatesSequence(nqubits, config):
        p = config['p']
        graph = nx.complete_graph(nqubits)
        gammas = [np.pi for _ in range(p)]
        betas = [np.pi for _ in range(p)]
        circuit = QAOA._make_qaoa_maxcut_circuit(nqubits, graph, gammas, betas)
        measure = config['measure']
        if measure:
            circuit.append(('measure', [list(range(nqubits))]))
        return circuit

    def _make_qaoa_maxcut_mixer_circuit(nqubits, beta):
        mixer_circuit = [('rx', [2*beta, q]) for q in range(nqubits)]
        return mixer_circuit

    def _make_qaoa_maxcut_problem_circuit(nqubits, graph, gamma):
        problem_circuit = []
        for v1, v2 in graph.edges():
            problem_circuit.append( ('cnot', [v1, v2]) )
            problem_circuit.append( ('rz', [gamma, v2]) )
            problem_circuit.append( ('cnot', [v1, v2]) )
        return problem_circuit

    def _make_qaoa_maxcut_circuit(nqubits, graph, gammas, betas):
        circuit = []
        # Initial circuit
        initial_circuit = [('h', [idx]) for idx in range(nqubits)]
        circuit += initial_circuit
        for p in range(len(gammas)):
            circuit += QAOA._make_qaoa_maxcut_problem_circuit(nqubits, graph, gammas[p])
            circuit += QAOA._make_qaoa_maxcut_mixer_circuit(nqubits, betas[p])
        return circuit
