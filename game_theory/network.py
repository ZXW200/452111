"""
交互网络模块 - 定义agents之间的交互结构
Interaction Network Module - Define interaction structures between agents
"""
import random
from typing import List, Tuple, Dict, Optional
from abc import ABC, abstractmethod

# 尝试导入networkx，如果没有则使用简单实现
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not installed. Using simple network implementation.")


class InteractionNetwork(ABC):
    """
    交互网络基类
    Base Interaction Network Class
    
    定义agents之间的交互拓扑结构
    """
    
    def __init__(self, agents: List[str]):
        """
        Args:
            agents: Agent名称列表
        """
        self.agents = agents
        self.n = len(agents)
        self._edges: List[Tuple[str, str]] = []
        self._adjacency: Dict[str, List[str]] = {a: [] for a in agents}
    
    @abstractmethod
    def _build_network(self):
        """构建网络结构"""
        pass
    
    def _add_edge(self, a1: str, a2: str):
        """添加边（无向）"""
        if a1 != a2 and a2 not in self._adjacency[a1]:
            self._edges.append((a1, a2))
            self._adjacency[a1].append(a2)
            self._adjacency[a2].append(a1)
    
    def get_neighbors(self, agent: str) -> List[str]:
        """
        获取agent的邻居（可交互对象）
        Get agent's neighbors (interaction partners)
        """
        return self._adjacency.get(agent, [])
    
    def get_all_edges(self) -> List[Tuple[str, str]]:
        """
        获取所有边（交互对）
        Get all edges (interaction pairs)
        """
        return self._edges.copy()
    
    def get_interaction_pairs(self) -> List[Tuple[str, str]]:
        """
        获取本轮的交互对
        Get interaction pairs for current round
        
        默认返回所有边，子类可覆盖实现更复杂的配对逻辑
        """
        return self.get_all_edges()
    
    def get_degree(self, agent: str) -> int:
        """获取agent的度（邻居数量）"""
        return len(self._adjacency.get(agent, []))
    
    def get_network_stats(self) -> Dict:
        """
        获取网络统计信息
        Get network statistics
        """
        degrees = [self.get_degree(a) for a in self.agents]
        return {
            "num_agents": self.n,
            "num_edges": len(self._edges),
            "avg_degree": sum(degrees) / self.n if self.n > 0 else 0,
            "min_degree": min(degrees) if degrees else 0,
            "max_degree": max(degrees) if degrees else 0,
        }
    
    def to_dict(self) -> Dict:
        """导出为字典格式（用于保存）"""
        return {
            "type": self.__class__.__name__,
            "agents": self.agents,
            "edges": self._edges,
            "adjacency": self._adjacency,
        }


# ============================================================
# 网络拓扑实现 / Network Topology Implementations
# ============================================================

class FullyConnectedNetwork(InteractionNetwork):
    """
    完全连接网络 - 所有agent两两可交互
    Fully Connected Network - All agents can interact with each other
    """
    name = "Fully Connected"
    
    def __init__(self, agents: List[str]):
        super().__init__(agents)
        self._build_network()
    
    def _build_network(self):
        for i, a1 in enumerate(self.agents):
            for a2 in self.agents[i+1:]:
                self._add_edge(a1, a2)


class RingNetwork(InteractionNetwork):
    """
    环形网络 - 每个agent只与相邻两个交互
    Ring Network - Each agent interacts with two neighbors
    """
    name = "Ring"
    
    def __init__(self, agents: List[str]):
        super().__init__(agents)
        self._build_network()
    
    def _build_network(self):
        for i in range(self.n):
            self._add_edge(self.agents[i], self.agents[(i+1) % self.n])


class GridNetwork(InteractionNetwork):
    """
    网格网络 - 2D网格结构
    Grid Network - 2D grid structure
    """
    name = "Grid"
    
    def __init__(self, agents: List[str], cols: int = None):
        self.cols = cols or int(len(agents) ** 0.5)
        self.rows = (len(agents) + self.cols - 1) // self.cols
        super().__init__(agents)
        self._build_network()
    
    def _build_network(self):
        for i, agent in enumerate(self.agents):
            row, col = i // self.cols, i % self.cols
            
            # 右邻居
            if col < self.cols - 1 and i + 1 < self.n:
                self._add_edge(agent, self.agents[i + 1])
            
            # 下邻居
            if row < self.rows - 1 and i + self.cols < self.n:
                self._add_edge(agent, self.agents[i + self.cols])


class StarNetwork(InteractionNetwork):
    """
    星形网络 - 一个中心节点连接所有其他节点
    Star Network - One central node connected to all others
    """
    name = "Star"
    
    def __init__(self, agents: List[str], center: str = None):
        self.center = center or agents[0]
        super().__init__(agents)
        self._build_network()
    
    def _build_network(self):
        for agent in self.agents:
            if agent != self.center:
                self._add_edge(self.center, agent)


class SmallWorldNetwork(InteractionNetwork):
    """
    小世界网络 - 基于Watts-Strogatz模型
    Small World Network - Based on Watts-Strogatz model
    
    特点：高聚类系数 + 短平均路径长度
    """
    name = "Small World"
    
    def __init__(self, agents: List[str], k: int = 4, p: float = 0.3):
        """
        Args:
            agents: Agent列表
            k: 每个节点的邻居数（必须是偶数）
            p: 重连概率
        """
        self.k = min(k, len(agents) - 1)
        if self.k % 2 == 1:
            self.k -= 1
        self.p = p
        super().__init__(agents)
        self._build_network()
    
    def _build_network(self):
        if HAS_NETWORKX:
            # 使用networkx生成
            G = nx.watts_strogatz_graph(self.n, self.k, self.p)
            for i, j in G.edges():
                self._add_edge(self.agents[i], self.agents[j])
        else:
            # 简单实现：先建环形，再随机重连
            # 1. 建立规则环形网络
            for i in range(self.n):
                for j in range(1, self.k // 2 + 1):
                    self._add_edge(self.agents[i], self.agents[(i + j) % self.n])
            
            # 2. 随机重连
            new_edges = []
            for a1, a2 in self._edges:
                if random.random() < self.p:
                    # 随机选择新的端点
                    candidates = [a for a in self.agents if a != a1 and a not in self._adjacency[a1]]
                    if candidates:
                        new_a2 = random.choice(candidates)
                        new_edges.append((a1, new_a2))
                    else:
                        new_edges.append((a1, a2))
                else:
                    new_edges.append((a1, a2))
            
            # 重建
            self._edges = []
            self._adjacency = {a: [] for a in self.agents}
            for a1, a2 in new_edges:
                self._add_edge(a1, a2)


class ScaleFreeNetwork(InteractionNetwork):
    """
    无标度网络 - 基于Barabási-Albert模型
    Scale-Free Network - Based on Barabási-Albert model
    
    特点：度分布服从幂律，存在少数高度连接的hub节点
    """
    name = "Scale Free"
    
    def __init__(self, agents: List[str], m: int = 2):
        """
        Args:
            agents: Agent列表
            m: 每个新节点连接的边数
        """
        self.m = min(m, len(agents) - 1)
        super().__init__(agents)
        self._build_network()
    
    def _build_network(self):
        if HAS_NETWORKX:
            G = nx.barabasi_albert_graph(self.n, self.m)
            for i, j in G.edges():
                self._add_edge(self.agents[i], self.agents[j])
        else:
            # 简单实现：优先连接
            # 初始完全图
            for i in range(self.m + 1):
                for j in range(i + 1, self.m + 1):
                    self._add_edge(self.agents[i], self.agents[j])
            
            # 逐步添加节点
            for i in range(self.m + 1, self.n):
                new_agent = self.agents[i]
                # 计算连接概率（基于度）
                total_degree = sum(self.get_degree(a) for a in self.agents[:i])
                if total_degree == 0:
                    targets = random.sample(self.agents[:i], min(self.m, i))
                else:
                    probs = [self.get_degree(a) / total_degree for a in self.agents[:i]]
                    targets = []
                    candidates = list(self.agents[:i])
                    for _ in range(min(self.m, i)):
                        if not candidates:
                            break
                        # 加权随机选择
                        r = random.random()
                        cumsum = 0
                        for idx, (cand, prob) in enumerate(zip(candidates, probs)):
                            cumsum += prob
                            if r <= cumsum:
                                targets.append(cand)
                                candidates.pop(idx)
                                probs.pop(idx)
                                # 重新归一化
                                if probs:
                                    s = sum(probs)
                                    probs = [p/s for p in probs]
                                break
                
                for target in targets:
                    self._add_edge(new_agent, target)


class RandomNetwork(InteractionNetwork):
    """
    随机网络 - Erdős-Rényi模型
    Random Network - Erdős-Rényi model
    """
    name = "Random (ER)"
    
    def __init__(self, agents: List[str], p: float = 0.3):
        """
        Args:
            agents: Agent列表
            p: 任意两个节点连接的概率
        """
        self.p = p
        super().__init__(agents)
        self._build_network()
    
    def _build_network(self):
        for i, a1 in enumerate(self.agents):
            for a2 in self.agents[i+1:]:
                if random.random() < self.p:
                    self._add_edge(a1, a2)


# ============================================================
# 网络注册表 / Network Registry
# ============================================================

NETWORK_REGISTRY = {
    "fully_connected": FullyConnectedNetwork,
    "ring": RingNetwork,
    "grid": GridNetwork,
    "star": StarNetwork,
    "small_world": SmallWorldNetwork,
    "scale_free": ScaleFreeNetwork,
    "random": RandomNetwork,
}


def create_network(network_type: str, agents: List[str], **kwargs) -> InteractionNetwork:
    """
    工厂函数：根据类型创建网络
    Factory function: Create network by type
    
    Args:
        network_type: 网络类型名称
        agents: Agent列表
        **kwargs: 网络参数
    
    Returns:
        网络实例
    """
    if network_type not in NETWORK_REGISTRY:
        raise ValueError(f"Unknown network type: {network_type}. "
                        f"Available: {list(NETWORK_REGISTRY.keys())}")
    
    return NETWORK_REGISTRY[network_type](agents, **kwargs)


# ============================================================
# 可视化（如果有matplotlib）/ Visualization
# ============================================================

def visualize_network(network: InteractionNetwork, output_path: str = None):
    """
    可视化网络结构
    Visualize network structure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, cannot visualize")
        return
    
    if HAS_NETWORKX:
        G = nx.Graph()
        G.add_nodes_from(network.agents)
        G.add_edges_from(network.get_all_edges())
        
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, k=2, iterations=50)
        nx.draw(G, pos, 
                with_labels=True, 
                node_color='lightblue',
                node_size=500,
                font_size=8,
                font_weight='bold',
                edge_color='gray',
                alpha=0.7)
        
        plt.title(f"{network.__class__.__name__}\n"
                 f"Nodes: {network.n}, Edges: {len(network._edges)}")
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Network visualization saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    else:
        print("networkx not installed, cannot visualize")
