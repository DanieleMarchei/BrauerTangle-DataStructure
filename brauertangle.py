from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from collections.abc import Iterable
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class Sign(Enum):
    positive = 1
    negative = 2

class EdgeType(Enum):
    upper_hook = 1
    lower_hook = 2
    zero_transversal = 3
    positive_transversal = 4
    negative_transversal = 5

@dataclass(unsafe_hash=True)
class Polarity:
    sign : Sign
    value : int

@dataclass(unsafe_hash=True)
class Node:
    id : int
    polarity : Polarity | None = field(init=False,compare=False, hash=False)

    def is_upper(self):
        return self.id > 0

    def is_lower(self):
        return self.id < 0
    
    def __post_init__(self):
        self.polarity = None

    def __sub__(self, __o: Node) -> int:
        return abs(self.id) - abs(__o.id)
    
    def __add__(self, __o: Node) -> Node:
        return abs(self.id) + abs(__o.id)
    
    def __eq__(self, __o: object) -> bool:
        if __o is None: return False
        if not type(__o) is Node: return False

        return self.id == __o.id

    def __lt__(self, __o: object) -> bool:
        if __o is None: return False
        if not type(__o) is Node: return False

        return abs(self.id) < abs(__o.id)

    def __le__(self, __o: object) -> bool:
        if __o is None: return False
        if not type(__o) is Node: return False

        return abs(self.id) <= abs(__o.id)

    def __gt__(self, __o: object) -> bool:
        if __o is None: return False
        if not type(__o) is Node: return False

        return abs(self.id) > abs(__o.id)

    def __ge__(self, __o: object) -> bool:
        if __o is None: return False
        if not type(__o) is Node: return False

        return abs(self.id) >= abs(__o.id)

    def __str__(self) -> str:
        return str(self.id)
    
    def __repr__(self) -> str:
        if self.polarity is None:
            p = "None"
        else:
            p = ("+" if self.polarity.sign == Sign.positive else "-") + str(self.polarity.value)
        return str(self) + f"({p})"

@dataclass(unsafe_hash=True)
class Edge:
    e1 : Node
    e2 : Node
    number_of_crossings : int = field(init=False, compare=False, hash=False)
    
    def __post_init__(self) -> None:
        self.number_of_crossings = 0

        if self.is_hook() and self.e2 < self.e1:
            self.e1, self.e2 = self.e2, self.e1
        
        if self.e1.is_lower() and self.e2.is_upper():
            self.e1, self.e2 = self.e2, self.e1

    def type(self) -> EdgeType:
        if self.e1.is_upper() and self.e2.is_upper():
            return EdgeType.upper_hook

        if self.e1.is_lower() and self.e2.is_lower():
            return EdgeType.lower_hook

        diff = self.e1 - self.e2
        if diff < 0: return EdgeType.negative_transversal
        if diff > 0: return EdgeType.positive_transversal
        if diff == 0: return EdgeType.zero_transversal

    def is_hook(self) -> bool:
        return self.type() in [EdgeType.upper_hook, EdgeType.lower_hook]
    
    def is_transversal(self) -> bool:
        return self.type() in [EdgeType.negative_transversal, EdgeType.zero_transversal, EdgeType.positive_transversal]

    def size(self) -> int:
        return abs(self.e1 - self.e2)
    
    def incr_crossings(self):
        self.number_of_crossings += 1
    
    def decr_crossings(self):
        if self.number_of_crossings == 0: return
        self.number_of_crossings -= 1
    
    def is_generated_only_by_Ts(self):
        return self.number_of_crossings >= self.size()

    def crosses_with(self, __o : Edge) -> bool:
        if self.is_hook() and __o.is_hook():
            b1 = self.e1 < __o.e1  < self.e2 < __o.e2
            b2 = __o.e1  < self.e1 < __o.e2  < self.e2
            return b1 or b2
        elif self.type() == EdgeType.upper_hook and __o.is_transversal():
            return self.e1 < __o.e1 < self.e2
        elif self.type() == EdgeType.lower_hook and __o.is_transversal():
            return self.e1 < __o.e2 < self.e2
        elif self.is_transversal() and __o.type() == EdgeType.upper_hook:
            return __o.e1 < self.e1 < __o.e2
        elif self.is_transversal() and __o.type() == EdgeType.lower_hook:
            return __o.e1 < self.e2 < __o.e2
        else:
            b1 = self.e1 < __o.e1  and self.e2 > __o.e2
            b2 = __o.e1  < self.e1 and __o.e2  > self.e2
            return b1 or b2

    def to_tuple(self):
        return (self.e1.id, self.e2.id)

    def __getitem__(self, index):
        return [self.e1, self.e2][index]

    def __setitem__(self, index, value):
        if index == 0:
            self.e1 = value
        elif index == 1:
            self.e2 = value

    def __str__(self):
        return f"({self.e1}, {self.e2})"
    
    def __repr__(self) -> str:
        n = "-" if self.number_of_crossings is None else self.number_of_crossings 
        return f"({repr(self.e1)},{repr(self.e2)})" + f"({n})"

class Tangle:
    def __init__(self, inv : list[Iterable[int]]) -> None:
        self.N = len(inv)
        self.nodes = {i : Node(i) for edge in inv for i in edge}
        self.edges = {(e1,e2) : Edge(self.nodes[e1],self.nodes[e2]) for e1,e2 in inv}
        self.polarities = {}
        self.crossings = {}
        self.max_pos_polarities = 0
        self.max_neg_polarities = 0
        self.node_to_edge = {}
        for e1,e2 in inv:
            self.node_to_edge[self.nodes[e1]] = (self.edges[(e1,e2)], 0)
            self.node_to_edge[self.nodes[e2]] = (self.edges[(e1,e2)], 1)
        
        self.is_node_polarity_computed = False
        self.are_edge_crossings_computed = False

    def compute_edge_crossings(self) -> None:
        if self.are_edge_crossings_computed: return

        done = set()
        for edge1 in self.edges.values():
            for edge2 in self.edges.values():
                if edge1 == edge2: continue
                if (edge1, edge2) in done or (edge2, edge1) in done: continue 

                if edge1.crosses_with(edge2):
                    if edge1 not in self.crossings:
                        self.crossings[edge1] = set()
                    if edge2 not in self.crossings:
                        self.crossings[edge2] = set()
                    
                    self.crossings[edge1].add(edge2)
                    self.crossings[edge2].add(edge1)
                    edge1.incr_crossings()
                    edge2.incr_crossings()
                
                done.add((edge1, edge2))
                done.add((edge2, edge1))
        
        self.are_edge_crossings_computed = True
    
    def compute_node_polarity(self) -> None:
        if self.is_node_polarity_computed: return

        if not self.are_edge_crossings_computed:
            self.compute_edge_crossings()

        for i in [1, -1]:
            pos_counter, neg_counter = 1, 1
            for id in range(1, self.N + 1):
                node = self.nodes[i * id]
                edge, _ = self.node_to_edge[node]
                if edge.is_generated_only_by_Ts(): continue

                edge_type = edge.type()

                if edge_type == EdgeType.upper_hook:
                    if node == edge.e1:
                        node.polarity = Polarity(Sign.negative, neg_counter)
                    else:
                        node.polarity = Polarity(Sign.positive, pos_counter)
                elif edge_type == EdgeType.lower_hook:
                    if node == edge.e1:
                        node.polarity = Polarity(Sign.positive, pos_counter)
                    else:
                        node.polarity = Polarity(Sign.negative, neg_counter)
                elif edge_type == EdgeType.positive_transversal:
                    node.polarity = Polarity(Sign.positive, pos_counter)
                elif edge_type == EdgeType.negative_transversal:
                    node.polarity = Polarity(Sign.negative, neg_counter)
                
                if node.polarity.sign == Sign.positive:
                    pos_counter += 1
                else:
                    neg_counter += 1
                
                if node.polarity not in self.polarities:
                    self.polarities[node.polarity] = []
                            
                self.polarities[node.polarity].append(node)

            
        self.max_pos_polarities = pos_counter
        self.max_neg_polarities = neg_counter
        
        self.is_node_polarity_computed = True

    def tfy(self) -> Tangle:
        self.compute_node_polarity()
        edges_tfy = []
        for edge in self.edges.values():
            if edge.is_generated_only_by_Ts():
                edges_tfy.append(edge.to_tuple())
        
        for node1, node2 in self.polarities.values():
            edges_tfy.append((node1.id, node2.id))
        
        return Tangle(edges_tfy)

    def n_crossings(self) -> int:
        if not self.are_edge_crossings_computed:
            self.compute_edge_crossings()

        double_n_crossings = 0
        for _, other_crossings in self.crossings.items():
            double_n_crossings += len(other_crossings)
        
        return double_n_crossings // 2

    def compose_t(self, i : int) -> None:

        if not self.is_node_polarity_computed:
            self.compute_node_polarity() 

        node_i = self.nodes[-i]
        edge_i, idx_i = self.node_to_edge[node_i]
        if edge_i.to_tuple() == (-i, -(i + 1)): return

        node_i_1 = self.nodes[-(i + 1)]
        edge_i_1, idx_i_1 = self.node_to_edge[node_i_1]

        # UPDATE CROSSINGS
        # if the edges cross, then reduce their crossing number. Decrease it otherwise
        if edge_i.crosses_with(edge_i_1):
            edge_i.decr_crossings()
            edge_i_1.decr_crossings()
        else:
            edge_i.incr_crossings()
            edge_i_1.incr_crossings()

        #swap the nodes of the edges
        edge_i[idx_i], edge_i_1[idx_i_1] = edge_i_1[idx_i_1], edge_i[idx_i]
        
        # UPDATE POLARITY
        # if the two nodes had the same polarity, reswap their polarity
        # because if they had different polarities then they should have not been swapped
        if edge_i[idx_i].polarity.sign != edge_i_1[idx_i_1].polarity.sign:
            edge_i[idx_i].polarity, edge_i_1[idx_i_1].polarity =  edge_i_1[idx_i_1].polarity, edge_i[idx_i].polarity
    
    def merge(self, edge1 : Edge, edge2 : Edge) -> None:
        # assume that at least one edge is a hook
        assert(edge1.is_hook() or edge2.is_hook())

        # decrease the crossing number of all edges crossing edge1 and edge2
        if edge1 in self.crossings:
            for other_edge in self.crossings[edge1]:
                other_edge.decr_crossings()
                self.crossings[other_edge].remove(edge1)
        
        if edge2 in self.crossings:
            for other_edge in self.crossings[edge2]:
                other_edge.decr_crossings()
                self.crossings[other_edge].remove(edge2)

        # remove old edges
        del self.edges[edge1.to_tuple()]
        del self.edges[edge2.to_tuple()]

        # delete reference from node to edge
        del self.node_to_edge[edge1.e1]
        del self.node_to_edge[edge1.e2]
        del self.node_to_edge[edge2.e1]
        del self.node_to_edge[edge2.e2]

        # merge
        new_edge1 = Edge(edge1.e1, edge2.e1)
        new_edge2 = Edge(edge1.e2, edge2.e2)
        self.edges[(edge1.e1, edge2.e1)] = new_edge1
        self.edges[(edge1.e2, edge2.e2)] = new_edge2

        # add reference to node to edge
        self.node_to_edge[new_edge1.e1] = (new_edge1, 0)
        self.node_to_edge[new_edge1.e2] = (new_edge1, 1)
        self.node_to_edge[new_edge2.e1] = (new_edge2, 0)
        self.node_to_edge[new_edge2.e2] = (new_edge2, 1)

        # update crossings
        for other_edge in self.edges.values():
            if other_edge == new_edge1: continue
            if not other_edge.crosses_with(new_edge1): continue
            
            other_edge.incr_crossings()
            new_edge1.incr_crossings()
            if other_edge not in self.crossings:
                self.crossings[other_edge] = set()
            if new_edge1 not in self.crossings:
                self.crossings[new_edge1] = set()
            self.crossings[other_edge].add(new_edge1)
            self.crossings[new_edge1].add(other_edge)

        for other_edge in self.edges.values():
            if other_edge == new_edge2: continue
            if not other_edge.crosses_with(new_edge2): continue
            
            other_edge.incr_crossings()
            new_edge2.incr_crossings()
            if other_edge not in self.crossings:
                self.crossings[other_edge] = set()
            if new_edge2 not in self.crossings:
                self.crossings[new_edge2] = set()
            self.crossings[other_edge].add(new_edge2)
            self.crossings[new_edge2].add(other_edge)
        
        self.are_edge_crossings_computed = True
        self.compute_node_polarity()

    def inv(self):
        edge_list = []
        for edge in self.edges.values():
            edge_list.append(edge.to_tuple())
        
        return edge_list

    def __getitem__(self, index):
        return self.nodes[index]

    def __str__(self) -> str:
        return str(self.inv())

    def __repr__(self) -> str:
        edge_list = []
        for edge in self.edges.values():
            edge_list.append(f"{repr(edge)}")
        
        return str(edge_list)


class HasseDiagram:
    def __init__(self, graph) -> None:
        self.gaph = graph
        self.pos = {}

        for node, data in graph.nodes.items():
            idx = data["idx"]
            lvl = data["lvl"]
            self.pos[node] = np.array([idx, lvl])
    
    def draw(self):
        nx.draw(self.graph, self.pos)
        plt.show()


def factorize_T(tangle: Tangle) -> list:
    assert(all([edge.is_transversal() for edge in tangle.edges.values()]))
    inv = sorted(tangle.inv(), key = lambda e : e[0])
    b = [abs(e[1]) for e in inv]
    factor_list = []
    for j in range(len(b)):
        skip_one = False
        for i in range(len(b)-1):
            if skip_one:
                skip_one = False 
                continue
            if b[i] > b[i + 1]:
                b[i], b[i + 1] = b[i + 1], b[i]
                factor_list.append(f"T{i+1}")
                skip_one = True

    return factor_list

def T_factor_list_to_hasse(factors : list[str]) -> HasseDiagram:
    def get_idx(f : str):
        return int(f.replace("T",""))

    graph = nx.Graph()
    pos = {}
    node_name = f"{factors[0]} - 0"
    graph.add_node(node_name, prime = "T", idx = get_idx(factors[0]), lvl = 0)
    lvls = {
        0 : {
            get_idx(factors[0]) : node_name
        }
    }
    prev_idx = get_idx(factors[0])
    lvl = 0
    for f in factors[1:]:
        idx = get_idx(f)
        if idx < prev_idx:
            lvl += 1
        

        node_name = f"{f} - {lvl}"
        if lvl not in lvls:
            lvls[lvl] = {}

        lvls[lvl][idx] = node_name
        graph.add_node(node_name, prime = "T", idx = idx, lvl = lvl)
        prev_idx = idx

    for lvl in range(1, len(lvls)):
        for idx, node in lvls[lvl].items():
            if idx - 1 in lvls[lvl - 1]:
                graph.add_edge(node, lvls[lvl - 1][idx - 1])
            
            if idx + 1 in lvls[lvl - 1]:
                graph.add_edge(node, lvls[lvl - 1][idx + 1])    

    return HasseDiagram(graph)

def length(tangle: Tangle) -> int:
    return tangle.tfy().n_crossings()

def factorize(tangle : Tangle) -> list:
    f = length(tangle)
    t = tangle.n_crossings()
    u = f - t

if __name__ == "__main__":
    # t = Tangle([[1,-2], [2,3], [-1,-3]])
    # t = Tangle([(1,4), (2,3), (-1,-3), (-2,-4)])
    t = Tangle([(1,-6), (2,5), (3,4), (6,-1), (-2, -4), (-3,-5)])
    factors_t = factorize_T(t.tfy())
    print(factors_t)
    hasse = T_factor_list_to_hasse(factors_t)
    print(hasse)
    print(factors_t)
