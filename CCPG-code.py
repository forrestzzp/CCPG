import networkx as nx
import torch
import logging
from typing import Dict, List, Tuple

# Configure secure and professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [CCPG Engine] - %(levelname)s: %(message)s')

class ComputableComputePowerGraph:
    """
    Computable Compute-Power Graph (CCPG) Core Engine Demo
    Combines NetworkX topology management with PyTorch tensor computation.
    """
    def __init__(self):
        # Initialize a directed graph representing asymmetric network communication environments
        self.graph = nx.DiGraph()
        # Hyperparameter definition (Lagrange multipliers) to balance latency, economic cost, and node state
        self.lambda_cost = torch.tensor(0.4)    # Economic cost weight
        self.lambda_latency = torch.tensor(0.6) # Network latency weight

    def add_compute_node(self, node_id: str, flops: float, memory: float, power_cost: float, utilization: float):
        """
        Add a compute node (e.g., GPU server)
        :param node_id: Unique identifier for the node
        :param flops: Theoretical compute capacity (TFLOPS)
        :param memory: Available VRAM/Memory (GB)
        :param power_cost: Real-time cost per unit time (money/power)
        :param utilization: Current utilization rate (0.0 - 1.0)
        """
        # Strict input validation (engineering safety)
        if not (0.0 <= utilization <= 1.0):
            raise ValueError("Utilization must be between 0.0 and 1.0")
            
        # Tensorize node attributes and integrate into the PyTorch ecosystem
        feature_vector = torch.tensor([flops, memory, power_cost, utilization], dtype=torch.float32)
        
        self.graph.add_node(
            node_id, 
            features=feature_vector,
            # Extract scalars for convenient routine querying
            memory_available=memory * (1 - utilization),
            flops_available=flops * (1 - utilization),
            cost=power_cost
        )
        logging.debug(f"Added Compute Node {node_id}: {feature_vector.tolist()}")

    def add_network_edge(self, src: str, dst: str, bandwidth: float, latency: float):
        """
        Add a network link (edge)
        :param src: Source node
        :param dst: Destination node
        :param bandwidth: Remaining bandwidth (Gbps)
        :param latency: Link latency (ms)
        """
        # Tensorize edge attributes
        edge_vector = torch.tensor([bandwidth, latency], dtype=torch.float32)
        self.graph.add_edge(src, dst, features=edge_vector, latency=latency)
        logging.debug(f"Added Network Edge {src} -> {dst}: {edge_vector.tolist()}")

    def _compute_dynamic_weights(self):
        """
        [Core Theory Implementation] 
        Utilize PyTorch tensor operations to dynamically calculate the "Computable Composite Cost" across the entire graph.
        Formula: J = λ1 * Node_Cost + λ2 * Edge_Latency
        """
        for u, v, data in self.graph.edges(data=True):
            # Retrieve the compute cost tensor of the target node (index 2 is power_cost)
            node_cost = self.graph.nodes[v]['features'][2]
            # Retrieve the latency tensor of the edge (index 1 is latency)
            edge_latency = data['features'][1]
            
            # Calculate the composite objective function using PyTorch
            # This leverages the potential of the autograd mechanism, though it is only forward propagation in this demo
            dynamic_weight = (self.lambda_cost * node_cost) + (self.lambda_latency * edge_latency)
            
            # Scalarize the computed tensor and store it in the graph's weight attribute for the optimization algorithm
            self.graph[u][v]['dynamic_weight'] = dynamic_weight.item()

    def optimize_task_routing(self, src_node: str, required_memory: float, required_flops: float) -> Tuple[List[str], float]:
        """
        Constrained optimization: Calculate the optimal compute routing path while satisfying compute and memory constraints.
        :param src_node: Source node initiating the task (e.g., user gateway)
        :param required_memory: Minimum required VRAM/Memory for the task (GB)
        :param required_flops: Minimum required compute capacity for the task (TFLOPS)
        """
        # 1. Trigger tensor computation to update the latest global dynamic weights
        self._compute_dynamic_weights()
        
        # 2. Generate a physical subgraph satisfying hardware constraints
        # Filter out nodes with insufficient compute or memory, reflecting strict physical infrastructure constraints
        valid_nodes = [
            n for n, attr in self.graph.nodes(data=True)
            if (attr['memory_available'] >= required_memory and 
                attr['flops_available'] >= required_flops) or n == src_node
        ]
        subgraph = self.graph.subgraph(valid_nodes)
        
        # 3. Search for the global optimal path on the filtered computable subgraph
        best_path = []
        min_total_cost = float('inf')
        target_compute_node = None

        for target in subgraph.nodes():
            if target == src_node:
                continue
            
            try:
                # Perform shortest path optimization based on dynamic tensor weights (dynamic_weight)
                cost = nx.shortest_path_length(subgraph, source=src_node, target=target, weight='dynamic_weight')
                if cost < min_total_cost:
                    min_total_cost = cost
                    target_compute_node = target
            except nx.NetworkXNoPath:
                continue

        if target_compute_node:
            best_path = nx.shortest_path(subgraph, source=src_node, target=target_compute_node, weight='dynamic_weight')
            logging.info(f"Optimal Compute Path found: {' -> '.join(best_path)} | Total Cost Function Value: {min_total_cost:.4f}")
            return best_path, min_total_cost
        else:
            logging.warning("No valid compute nodes available satisfying the physical constraints.")
            return [], -1.0


# ==========================================
# Scenario Simulation: Cross-regional compute scheduling (Demo Execution Script)
# ==========================================
if __name__ == "__main__":
    logging.info("Initializing Computable Compute-Power Graph (CCPG)...")
    ccpg = ComputableComputePowerGraph()

    # 1. Construct physical nodes (e.g., three intelligent compute centers in an East-Data-West-Compute scenario)
    # Parameters: ID, FLOPS, Memory(GB), Cost($/hr), Utilization
    ccpg.add_compute_node("Gateway_Beijing", flops=10, memory=64, power_cost=2.0, utilization=0.9) # Congested and expensive
    ccpg.add_compute_node("Node_Ningxia", flops=500, memory=1024, power_cost=0.5, utilization=0.2) # Cheap, idle
    ccpg.add_compute_node("Node_Guizhou", flops=800, memory=2048, power_cost=0.4, utilization=0.4) # Cheaper, powerful compute

    # 2. Construct communication topology (fiber-optic network)
    # Parameters: SRC, DST, Bandwidth(Gbps), Latency(ms)
    ccpg.add_network_edge("Gateway_Beijing", "Node_Ningxia", bandwidth=100, latency=15.0)
    ccpg.add_network_edge("Gateway_Beijing", "Node_Guizhou", bandwidth=100, latency=35.0)
    ccpg.add_network_edge("Node_Ningxia", "Node_Guizhou", bandwidth=400, latency=10.0)

    # 3. Submit a large AI training task request
    logging.info("Submitting AI Training Task (Requires: 500GB Memory, 300 TFLOPS)")
    task_memory = 500.0
    task_flops = 300.0

    # 4. Execute the CCPG engine computation
    best_path, cost = ccpg.optimize_task_routing(
        src_node="Gateway_Beijing", 
        required_memory=task_memory, 
        required_flops=task_flops
    )
