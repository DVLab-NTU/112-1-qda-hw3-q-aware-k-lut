'''
[ Description ] A rudimentary implementation of the Quantum-Aware Partitioning algorithm from the paper 
                [G. Meuli et al., ROS: Resource-constrained Oracle Synthesis for Quantum Computers](https://arxiv.org/abs/2005.00211)
                More efficient implementations are possible, but this implementation is intended to be as simple as possible.
[ Year        ] 2023
[ Author      ] Mu-Te Joshua Lau, DVLab, Graduate Institute of Electric Engineering, National Taiwan University
[ License     ] The author waives all rights to this work and places it in the public domain to the maximum extent permitted by law. 
                This work is provided "as is", without any warranty of any kind.
'''

from enum import Enum
import itertools
import functools
from scipy.linalg import hadamard
import numpy as np
from argparse import ArgumentParser

IdType = int
CutType = set[IdType]
SizeType = int
CostType = int

# XAG node types. In theory, there should be INVerters, too. However, we do not need them for our purposes.
class XAGNodeType(Enum):
    INPUT=0,
    XOR=1,
    AND=2

# encapsulates a single node in the XAG
class XAGNode:
    fanins: list[IdType]     # fanin node ids
    inverted: list[bool]     # whether the corresponding fanin is inverted
    fanouts: list[IdType]()  # fanout node ids. This is calculated automatically by the XAG class.
    typ: XAGNodeType         # node type
    
    def __init__(self, fanins, inverted, typ):
        self.fanins = fanins
        self.inverted = inverted
        self.typ = typ
        self.fanouts = list[IdType]()
        if typ == XAGNodeType.INPUT:
            # input nodes should not have fanins
            assert(len(fanins) == 0)
            assert(len(inverted) == 0)
        if typ == XAGNodeType.XOR or typ == XAGNodeType.AND:
            # XOR and AND nodes should have exactly two fanins
            assert(len(fanins) == 2)
            assert(len(inverted) == 2)

class XAG:
    nodes: list[XAGNode]          # list of nodes
    primary_inputs: list[IdType]  # primary input node ids
    primary_outputs: list[IdType] # primary output node ids

    def __init__(self, nodes: list[XAGNode], primary_inputs: list[IdType] = [], primary_outputs: list[IdType] = []):
        self.nodes = nodes
        self.primary_inputs = primary_inputs
        self.primary_outputs = primary_outputs
        assert(all(id in range(len(self.nodes)) for id in self.primary_inputs))
        assert(all(id in range(len(self.nodes)) for id in self.primary_outputs))
        self._evaluate_fanouts()

    def __len__(self):
        return len(self.nodes)
    
    def __iter__(self):
        return iter(self.nodes)
    
    def __getitem__(self, key):
        return self.nodes[key]
    
    def __setitem__(self, key, value):
        self.nodes[key] = value
    
    def __delitem__(self, key):
        del self.nodes[key]
    
    def __contains__(self, item):
        return item in self.nodes
    
    def __str__(self):
        ret_str = ''
        for id, node in enumerate(self.nodes):
            ret_str += f'{id}: {node.typ}  inputs: {node.fanins}  outputs: {node.fanouts}\n'

        return ret_str
    
    # internal function to calculate fanouts
    def _evaluate_fanouts(self):
        for node in self.nodes:
            assert(not (node.typ == XAGNodeType.INPUT and len(node.fanins) != 0))
            for fanin in node.fanins:
                assert(fanin in range(len(self.nodes)))
                assert(fanin >= 0)

        for id, node in enumerate(self.nodes):
            for fanin in node.fanins:
                self.nodes[fanin].fanouts.append(id)

        for id, node in enumerate(self.nodes):
            node.fanouts.sort()

    # calculates the topological order of the XAG, and returns it as a list of ids
    def calculate_topo_order(self):
        id_to_node = dict[IdType, XAGNode]()
        for id, node in enumerate(self.nodes):
            id_to_node[id] = node
        
        topo_order = list[IdType]()
        visited = set[IdType]()
        waiting = set[IdType]()

        for id in range(len(self.nodes)):
            if (id_to_node[id].typ == XAGNodeType.INPUT):
                waiting.add(id)
        
        while (len(waiting) != 0):
            id = waiting.pop()
            visited.add(id)

            topo_order.append(id)
            for output in id_to_node[id].fanouts:
                if (output not in visited):
                    waiting.add(output)
                    
        return topo_order

# enumerates all cuts of size up to max_cut_size for each node in the XAG
def enumerate_cuts(xag: XAG, max_cut_size: int) -> dict[IdType, list[CutType]]:
    assert(max_cut_size > 1)
    topo_order = xag.calculate_topo_order()
    id_to_cuts = dict[IdType, list[CutType]]()
    for id in topo_order:
        id_to_cuts[id] = list[CutType]()
        
        id_to_cuts[id].append(set([id]))

        if (xag.nodes[id].typ == XAGNodeType.INPUT):
            continue
        
        fanins = [xag.nodes[id].fanins[i] for i in range(len(xag.nodes[id].fanins))]

        # the cuts of the node are the union of the cuts of its fanins
        for cut0, cut1 in itertools.product(id_to_cuts[fanins[0]], id_to_cuts[fanins[1]]):
            merged_cut = cut0 | cut1

            # if the cut is too large, we do not add it
            if (len(merged_cut) <= max_cut_size):

                # if a cut is a subset of another cut, we do not add it
                # this is because the additional wires in the larger cut will not be used anyway, and only complicate the calculation
                if not any(cut.issubset(merged_cut) for cut in id_to_cuts[id]):
                    id_to_cuts[id].append(merged_cut)

    # remove the trivial cut {id}.
    for id, cuts in list(id_to_cuts.items()):
        cuts.remove({id})
        if (len(cuts) == 0):
            del id_to_cuts[id]

    return id_to_cuts

# gets all nodes in the cone between node_id and cut in the topological order
def get_cone_node_ids(xag: XAG, node_id: IdType, cut: CutType) -> list[IdType]:
    # traverse the XAG in the topological order, and add all nodes in the cone to the list
    def _get_cone_nodes_internal(xag: XAG, node_id: IdType, cut: CutType, cone: list[IdType]) -> None:
        for fanin in xag.nodes[node_id].fanins:
            if (node_id in cut and fanin not in cut):
                continue
            if (fanin not in cone):
                _get_cone_nodes_internal(xag, fanin, cut, cone)
        cone.append(node_id)
        

    cone = list[IdType]()
    _get_cone_nodes_internal(xag, node_id, cut, cone)
    # the cone should contain all nodes in the cut
    # assert(set(cut) - set(cone) == set())
    return cone

# calculates the truth table of node_id given the cut
def calculate_truth_table(xag: XAG, node_id: IdType, cut: CutType) -> list[bool]:
    assert(len(cut) > 1 or (len(cut) == 1 and cut[0] == node_id))
    node_ids_in_cone = get_cone_node_ids(xag, node_id, cut)
    truth_table = list[bool]()
    for minterm in list(itertools.product([0,1], repeat = len(cut))):
        # assert(all(wire in node_ids_in_cone for wire in cut))
        intermediate_results = {wire: literal for wire, literal in zip(cut, minterm)}
        
        for id in node_ids_in_cone:
            if (id in cut):
                continue
            inputs = [intermediate_results[fanin] ^ inverted for fanin, inverted in zip(xag[id].fanins, xag[id].inverted)]
            if xag[id].typ == XAGNodeType.XOR:
                intermediate_results[id] = functools.reduce(lambda x, y: x ^ y, inputs)
            elif xag[id].typ == XAGNodeType.AND:
                intermediate_results[id] = functools.reduce(lambda x, y: x & y, inputs)
            else:
                raise ValueError('node type should not be input!!')
            
        truth_table.append(intermediate_results[node_id])
    
    return truth_table

# calculates the cost of each cut, and returns it as a dictionary
def calculate_cut_costs(xag: XAG, cuts: dict[IdType, CutType], quantum_aware: bool = True) -> dict[IdType, list[CostType]]:
    # calculates the cost of a cut using the Radamacher-Walsh transform
    def calculate_radamacher_walsh_cost(truth_table: list[bool]) -> CostType:
        assert(np.log2(len(truth_table)) % 1 == 0)
        '''
        TODO - Calculate the Radamacher-Walsh cost of a cut with the truth table
               For reference, the Radamacher-Walsh Transform is defined as follows:
               G = FH_n, where
               F: [f(0), f(1), ..., f(2^n-1)] and
               H_n is the Hadamard matrix of size 2^n x 2^n

               The Radamacher-Walsh cost is the number of non-zero entries in G.
        '''
        return 0

    costs = dict[IdType, list[CostType]]()
    for id, cut in cuts.items():
        costs[id] = list[CostType]()
        for c in cut:
            costs[id].append(calculate_radamacher_walsh_cost(calculate_truth_table(xag, id, c)))
    '''
    TODO - Decrease the cost to 0 of the function if all nodes in the function are XORs and there is at least two XORs.
           This is the "quantum aware" part of the algorithm. 
           Consecutive XORs can be implemented with single-target gates on a single ancilla line, reducing the cost
           Therefore, we mark the cost of such cuts as 0 to incentivize the algorithm to use them.
    '''
    if (quantum_aware):
        pass

    return costs

# the quantum aware partitioning algorithm
def partition(xag: XAG, max_cut_size: int, quantum_aware: bool = True) -> tuple[dict[IdType, list[CutType]], dict[IdType, list[CostType]]]:
    id_to_cuts = enumerate_cuts(xag, max_cut_size)
    id_to_costs = calculate_cut_costs(xag, id_to_cuts, quantum_aware)

    # dynamic programming
    optimal_cuts = dict[IdType, CutType]()
    optimal_costs = dict[IdType, CostType]()
    for node_id in xag.calculate_topo_order():
        # initial condition
        if (xag[node_id].typ == XAGNodeType.INPUT):
            optimal_cuts[node_id] = set([node_id])
            optimal_costs[node_id] = 0
            continue
        
        # recursive condition
        optimal_costs[node_id] = np.inf
        for cut, cost in zip(id_to_cuts[node_id], id_to_costs[node_id]):
            # the cost of a cut is the sum of the cost of the cut and the cost of the optimal cuts of the fanins
            acc_cost = cost + functools.reduce(lambda x, y: x + optimal_costs[y], cut, 0)
            if (acc_cost < optimal_costs[node_id]):
                optimal_costs[node_id] = acc_cost
                optimal_cuts[node_id] = cut
    
    node_id_set = set(range(len(xag)))

    # remove intermediate nodes and nodes that are not in the optimal cut
    necessary_node_ids = set[int]()
    for id, cut in reversed(optimal_cuts.items()):
        if (id in xag.primary_outputs or id in xag.primary_inputs):
            necessary_node_ids.add(id)
        if (id in necessary_node_ids):
            for wire in cut:
                necessary_node_ids.add(wire)

    for id in node_id_set - necessary_node_ids:
        del optimal_cuts[id]

    # keep only the optimal costs for the primary outputs
    for id in node_id_set:
        if (id not in necessary_node_ids):
            del optimal_costs[id]

    return optimal_cuts, optimal_costs
    

def main():
    # define the XAG 
    xag = XAG([
        XAGNode([], [], XAGNodeType.INPUT),                 # 0
        XAGNode([], [], XAGNodeType.INPUT),                 # 1
        XAGNode([], [], XAGNodeType.INPUT),                 # 2
        XAGNode([], [], XAGNodeType.INPUT),                 # 3 
        XAGNode([], [], XAGNodeType.INPUT),                 # 4
        XAGNode([], [], XAGNodeType.INPUT),                 # 5
        XAGNode([0, 1], [False, False], XAGNodeType.AND),   # 6
        XAGNode([1, 4], [False, False], XAGNodeType.XOR),   # 7
        XAGNode([4, 5], [False, False], XAGNodeType.AND),   # 8
        XAGNode([2, 6], [False, False], XAGNodeType.AND),   # 9
        XAGNode([3, 6], [False, False], XAGNodeType.XOR),   # 10
        XAGNode([3, 7], [False, False], XAGNodeType.XOR),   # 11
        XAGNode([4, 8], [False, False], XAGNodeType.XOR),   # 12
        XAGNode([9, 10], [False, False], XAGNodeType.XOR),  # 13
        XAGNode([10, 11], [False, False], XAGNodeType.XOR), # 14
        XAGNode([11, 12], [False, False], XAGNodeType.XOR), # 15
        XAGNode([13, 14], [False, False], XAGNodeType.AND), # 16
        XAGNode([15, 16], [False, False], XAGNodeType.XOR)  # 17
    ],
    [0, 1, 2, 3, 4, 5], # PI 
    [17]) # PO
    parser = ArgumentParser()

    parser.add_argument('-n', '--cut-size', type=int, default=3, help='maximum cut size')
    parser.add_argument('-c', '--classical', action='store_true', help='use classical partitioning cost')

    args = parser.parse_args()
    cuts, costs = partition(xag, max_cut_size=args.cut_size, quantum_aware=not args.classical)

    print(cuts)
    print(costs)
        

if __name__ == '__main__':
    main()
