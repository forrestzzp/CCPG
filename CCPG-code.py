import networkx as nx
import torch
import logging
from typing import Dict, List, Tuple

# 配置安全与专业的日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [CCPG Engine] - %(levelname)s: %(message)s')

class ComputableComputePowerGraph:
    """
    可算力图谱 (CCPG) 核心计算引擎 Demo
    结合 NetworkX 的拓扑管理与 PyTorch 的张量计算
    """
    def __init__(self):
        # 初始化有向图，代表非对称的网络通信环境
        self.graph = nx.DiGraph()
        # 超参数定义 (拉格朗日乘子)，用于平衡延迟、经济成本与节点状态
        self.lambda_cost = torch.tensor(0.4)    # 经济成本权重
        self.lambda_latency = torch.tensor(0.6) # 网络延迟权重

    def add_compute_node(self, node_id: str, flops: float, memory: float, power_cost: float, utilization: float):
        """
        添加计算节点 (如 GPU 服务器)
        :param node_id: 节点唯一标识
        :param flops: 理论算力 (TFLOPS)
        :param memory: 可用显存 (GB)
        :param power_cost: 实时单位时间金钱/电量成本
        :param utilization: 当前利用率 (0.0 - 1.0)
        """
        # 严格的输入校验 (工程安全性)
        if not (0.0 <= utilization <= 1.0):
            raise ValueError("Utilization must be between 0.0 and 1.0")
            
        # 将节点属性张量化 (Tensorization)，转入 PyTorch 体系
        feature_vector = torch.tensor([flops, memory, power_cost, utilization], dtype=torch.float32)
        
        self.graph.add_node(
            node_id, 
            features=feature_vector,
            # 提取标量方便常规查询
            memory_available=memory * (1 - utilization),
            flops_available=flops * (1 - utilization),
            cost=power_cost
        )
        logging.debug(f"Added Compute Node {node_id}: {feature_vector.tolist()}")

    def add_network_edge(self, src: str, dst: str, bandwidth: float, latency: float):
        """
        添加网络链路
        :param src: 源节点
        :param dst: 目标节点
        :param bandwidth: 剩余带宽 (Gbps)
        :param latency: 链路延迟 (ms)
        """
        # 将边属性张量化
        edge_vector = torch.tensor([bandwidth, latency], dtype=torch.float32)
        self.graph.add_edge(src, dst, features=edge_vector, latency=latency)
        logging.debug(f"Added Network Edge {src} -> {dst}: {edge_vector.tolist()}")

    def _compute_dynamic_weights(self):
        """
        [核心理论体现] 
        利用 PyTorch 张量运算，动态计算全图的“可算力复合权重” (Composite Cost)
        公式: J = λ1 * Node_Cost + λ2 * Edge_Latency
        """
        for u, v, data in self.graph.edges(data=True):
            # 获取目标节点的算力成本张量 (索引 2 是 power_cost)
            node_cost = self.graph.nodes[v]['features'][2]
            # 获取链路的延迟张量 (索引 1 是 latency)
            edge_latency = data['features'][1]
            
            # 使用 PyTorch 计算复合目标函数
            # 这里利用了 autograd 机制的潜力，虽然在此 demo 中是正向传播
            dynamic_weight = (self.lambda_cost * node_cost) + (self.lambda_latency * edge_latency)
            
            # 将算出的张量标量化，存入图的权重属性中供寻优算法使用
            self.graph[u][v]['dynamic_weight'] = dynamic_weight.item()

    def optimize_task_routing(self, src_node: str, required_memory: float, required_flops: float) -> Tuple[List[str], float]:
        """
        约束寻优：在满足算力和显存约束下，计算最优的算力分发路径
        :param src_node: 任务发起的源节点 (如用户的网关)
        :param required_memory: 任务所需的最小显存 (GB)
        :param required_flops: 任务所需的最小算力 (TFLOPS)
        """
        # 1. 触发张量计算，更新全局最新的动态权重
        self._compute_dynamic_weights()
        
        # 2. 生成满足硬件约束的物理子图 (SubGraph)
        # 过滤掉算力或显存不足的节点，这体现了物理设施强约束性
        valid_nodes = [
            n for n, attr in self.graph.nodes(data=True)
            if (attr['memory_available'] >= required_memory and 
                attr['flops_available'] >= required_flops) or n == src_node
        ]
        subgraph = self.graph.subgraph(valid_nodes)
        
        # 3. 在过滤后的可算力子图上寻找全局最优路径
        best_path = []
        min_total_cost = float('inf')
        target_compute_node = None

        for target in subgraph.nodes():
            if target == src_node:
                continue
            
            try:
                # 基于动态张量权重 (dynamic_weight) 进行最短路径寻优
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
# 场景模拟：跨地域算力调度 (Demo 运行脚本)
# ==========================================
if __name__ == "__main__":
    logging.info("Initializing Computable Compute-Power Graph (CCPG)...")
    ccpg = ComputableComputePowerGraph()

    # 1. 构建物理节点 (例如：东数西算的三个智算中心)
    # 参数: ID, FLOPS, Memory(GB), Cost(美元/时), 负载率
    ccpg.add_compute_node("Gateway_Beijing", flops=10, memory=64, power_cost=2.0, utilization=0.9) # 拥挤且昂贵
    ccpg.add_compute_node("Node_Ningxia", flops=500, memory=1024, power_cost=0.5, utilization=0.2) # 便宜，空闲
    ccpg.add_compute_node("Node_Guizhou", flops=800, memory=2048, power_cost=0.4, utilization=0.4) # 更便宜，算力强

    # 2. 构建通信拓扑 (光纤网络)
    # 参数: SRC, DST, Bandwidth(Gbps), Latency(ms)
    ccpg.add_network_edge("Gateway_Beijing", "Node_Ningxia", bandwidth=100, latency=15.0)
    ccpg.add_network_edge("Gateway_Beijing", "Node_Guizhou", bandwidth=100, latency=35.0)
    ccpg.add_network_edge("Node_Ningxia", "Node_Guizhou", bandwidth=400, latency=10.0)

    # 3. 提交大型 AI 训练任务请求
    logging.info("Submitting AI Training Task (Requires: 500GB Memory, 300 TFLOPS)")
    task_memory = 500.0
    task_flops = 300.0

    # 4. 执行可算力图谱引擎计算
    best_path, cost = ccpg.optimize_task_routing(
        src_node="Gateway_Beijing", 
        required_memory=task_memory, 
        required_flops=task_flops
    )