"""
博弈仿真引擎 - 管理整个仿真流程
Game Simulation Engine - Manage the entire simulation process
"""
import json
import os
import concurrent.futures
from datetime import datetime
from typing import List, Dict, Callable, Optional, Tuple
from dataclasses import dataclass, field

# 针对 I/O 密集型 API 请求优化的线程池大小
# Python 默认是 min(32, CPU+4)，但 API 请求是"纯等待"任务，可以开更大
# 100 个线程足以支持大规模并行 API 调用
MAX_API_WORKERS = 100

from .games import GameConfig, Action, get_payoff, PRISONERS_DILEMMA
from .network import InteractionNetwork, FullyConnectedNetwork
from .strategies import Strategy, create_strategy


@dataclass
class AgentState:
    """
    Agent状态（简化版，替代原来的Persona）
    Agent State (simplified, replacing original Persona)
    """
    name: str
    strategy: Strategy
    description: str = ""
    personality: str = ""
    
    # 博弈相关状态
    game_history: List[Dict] = field(default_factory=list)
    total_payoff: float = 0.0
    opponent_models: Dict[str, str] = field(default_factory=dict)
    
    def record_game(self, opponent: str, my_action: Action, opp_action: Action, payoff: float):
        """记录一次博弈"""
        self.game_history.append({
            "round": len(self.game_history) + 1,
            "opponent": opponent,
            "my_action": my_action.value,
            "opp_action": opp_action.value,
            "payoff": payoff,
        })
        self.total_payoff += payoff
    
    def get_history_with(self, opponent: str) -> List[Tuple[Action, Action]]:
        """获取与特定对手的历史"""
        history = []
        for g in self.game_history:
            if g["opponent"] == opponent:
                my_act = Action(g["my_action"])
                opp_act = Action(g["opp_action"])
                history.append((my_act, opp_act))
        return history
    
    def get_cooperation_rate(self) -> float:
        """计算合作率"""
        if not self.game_history:
            return 0.0
        coop_count = sum(1 for g in self.game_history if g["my_action"] == "cooperate")
        return coop_count / len(self.game_history)
    
    def to_dict(self) -> Dict:
        """导出为字典"""
        return {
            "name": self.name,
            "strategy": self.strategy.name,
            "description": self.description,
            "total_payoff": self.total_payoff,
            "cooperation_rate": self.get_cooperation_rate(),
            "game_history": self.game_history,
        }


class GameSimulation:
    """
    博弈仿真主类
    Main Game Simulation Class
    """
    
    def __init__(self,
                 agents: Dict[str, AgentState],
                 game_config: GameConfig,
                 network: InteractionNetwork,
                 rounds: int = 100,
                 verbose: bool = True):
        """
        Args:
            agents: Agent字典 {name: AgentState}
            game_config: 博弈配置
            network: 交互网络
            rounds: 总轮数
            verbose: 是否打印详细信息
        """
        self.agents = agents
        self.game_config = game_config
        self.network = network
        self.total_rounds = rounds
        self.verbose = verbose
        
        self.current_round = 0
        self.round_results: List[Dict] = []
        
    def run(self, 
            round_callback: Callable = None,
            reflection_interval: int = 10) -> Dict:
        """
        运行完整仿真
        Run complete simulation
        
        Args:
            round_callback: 每轮结束后的回调函数 callback(round_num, round_data)
            reflection_interval: 反思间隔（每N轮触发一次）
        
        Returns:
            仿真结果
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Starting Game Theory Simulation")
            print(f"{'='*60}")
            print(f"Game: {self.game_config.name}")
            print(f"Agents: {list(self.agents.keys())}")
            print(f"Network: {self.network.__class__.__name__}")
            print(f"Total Rounds: {self.total_rounds}")
            print(f"{'='*60}\n")
        
        for round_num in range(1, self.total_rounds + 1):
            self.current_round = round_num
            round_data = self._run_single_round()
            self.round_results.append(round_data)
            
            if round_callback:
                round_callback(round_num, round_data)
            
            # 定期反思（策略调整）
            if round_num % reflection_interval == 0:
                self._trigger_reflection()
            
            if self.verbose and round_num % 10 == 0:
                self._print_progress(round_num)
        
        results = self._compile_results()
        
        if self.verbose:
            self._print_final_results(results)
        
        return results
    
    def _run_single_round(self) -> Dict:
        """
        执行单轮博弈（并行化版本）

        优化：使用 ThreadPoolExecutor 并行执行所有 Agent 的 choose_action 调用，
        将每轮耗时从 N * API延迟 缩短到 1 * API延迟
        """
        round_data = {
            "round": self.current_round,
            "interactions": [],
            "round_payoffs": {name: 0.0 for name in self.agents}
        }

        # 获取本轮交互对
        pairs = self.network.get_interaction_pairs()

        # 准备所有决策任务：收集 (agent, history, opponent_name) 元组
        decision_tasks = []
        for agent1_name, agent2_name in pairs:
            agent1 = self.agents[agent1_name]
            agent2 = self.agents[agent2_name]

            # 获取双方历史
            history1 = agent1.get_history_with(agent2_name)
            history2 = agent2.get_history_with(agent1_name)

            # 添加两个决策任务（每对交互需要两个决策）
            decision_tasks.append((agent1.strategy, history1, agent2_name))
            decision_tasks.append((agent2.strategy, history2, agent1_name))

        # 定义执行单个决策的函数
        def execute_decision(task):
            strategy, history, opponent_name = task
            return strategy.choose_action(history, opponent_name)

        # 并行执行所有决策
        # ThreadPoolExecutor 适合 I/O 密集型任务（如 API 调用）
        # 使用较大的线程池以确保所有 Agent 真正同时发起请求
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_API_WORKERS) as executor:
            actions = list(executor.map(execute_decision, decision_tasks))

        # 处理结果：将并行获取的动作与交互对匹配
        for i, (agent1_name, agent2_name) in enumerate(pairs):
            agent1 = self.agents[agent1_name]
            agent2 = self.agents[agent2_name]

            # 从并行结果中获取动作（每对交互占用两个连续的结果）
            action1 = actions[i * 2]
            action2 = actions[i * 2 + 1]

            # 计算收益
            payoff1, payoff2 = get_payoff(self.game_config, action1, action2)

            # 记录结果
            agent1.record_game(agent2_name, action1, action2, payoff1)
            agent2.record_game(agent1_name, action2, action1, payoff2)

            # 更新 LLM 策略的 total_payoff（如果有此方法）
            if hasattr(agent1.strategy, 'update_payoff'):
                agent1.strategy.update_payoff(payoff1)
            if hasattr(agent2.strategy, 'update_payoff'):
                agent2.strategy.update_payoff(payoff2)

            # 保存交互数据
            round_data["interactions"].append({
                "agent1": agent1_name,
                "agent2": agent2_name,
                "action1": action1.value,
                "action2": action2.value,
                "payoff1": payoff1,
                "payoff2": payoff2,
            })

            round_data["round_payoffs"][agent1_name] += payoff1
            round_data["round_payoffs"][agent2_name] += payoff2

        return round_data
    
    def _trigger_reflection(self):
        """
        触发策略反思
        TODO: Week 2-3 实现基于LLM的策略调整
        """
        pass
    
    def _compile_results(self) -> Dict:
        """汇总仿真结果"""
        final_payoffs = {
            name: agent.total_payoff 
            for name, agent in self.agents.items()
        }
        
        cooperation_rates = {
            name: agent.get_cooperation_rate()
            for name, agent in self.agents.items()
        }
        
        # 计算合作率随时间的变化
        cooperation_evolution = []
        for round_data in self.round_results:
            total_coop = 0
            total_actions = 0
            for interaction in round_data["interactions"]:
                if interaction["action1"] == "cooperate":
                    total_coop += 1
                if interaction["action2"] == "cooperate":
                    total_coop += 1
                total_actions += 2
            
            rate = total_coop / total_actions if total_actions > 0 else 0
            cooperation_evolution.append(rate)
        
        return {
            "config": {
                "game": self.game_config.name,
                "network": self.network.__class__.__name__,
                "total_rounds": self.total_rounds,
                "num_agents": len(self.agents),
            },
            "final_payoffs": final_payoffs,
            "cooperation_rates": cooperation_rates,
            "cooperation_evolution": cooperation_evolution,
            "agent_details": {
                name: agent.to_dict() 
                for name, agent in self.agents.items()
            },
            "round_history": self.round_results,
        }
    
    def _print_progress(self, round_num: int):
        """打印进度"""
        payoffs = [(name, agent.total_payoff) for name, agent in self.agents.items()]
        payoffs.sort(key=lambda x: x[1], reverse=True)
        
        # 计算当前整体合作率
        last_round = self.round_results[-1]
        coop_count = sum(
            (1 if i["action1"] == "cooperate" else 0) + 
            (1 if i["action2"] == "cooperate" else 0)
            for i in last_round["interactions"]
        )
        total_actions = len(last_round["interactions"]) * 2
        coop_rate = coop_count / total_actions if total_actions > 0 else 0
        
        print(f"Round {round_num:4d} | Coop Rate: {coop_rate:.1%} | "
              f"Top: {payoffs[0][0]}({payoffs[0][1]:.1f})")
    
    def _print_final_results(self, results: Dict):
        """打印最终结果"""
        print(f"\n{'='*60}")
        print("SIMULATION COMPLETE")
        print(f"{'='*60}")
        
        print("\n📊 Final Rankings:")
        payoffs = list(results["final_payoffs"].items())
        payoffs.sort(key=lambda x: x[1], reverse=True)
        for i, (name, payoff) in enumerate(payoffs, 1):
            coop_rate = results["cooperation_rates"][name]
            strategy = self.agents[name].strategy.name
            print(f"  {i}. {name:15s} | Payoff: {payoff:7.1f} | "
                  f"Coop: {coop_rate:.1%} | Strategy: {strategy}")
        
        print(f"\n📈 Overall Cooperation Rate: "
              f"{sum(results['cooperation_rates'].values())/len(results['cooperation_rates']):.1%}")
    
    def save_results(self, output_dir: str = "experiments/results") -> str:
        """保存结果到文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/sim_{self.game_config.name.replace(' ', '_')}_{timestamp}.json"
        
        results = self._compile_results()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {filename}")
        return filename


# ============================================================
# 快速创建仿真的辅助函数 / Helper Functions
# ============================================================

def create_simulation(
    num_agents: int = 5,
    strategy_config: Dict[str, str] = None,
    game_type: str = "prisoners_dilemma",
    network_type: str = "fully_connected",
    rounds: int = 100,
    **kwargs
) -> GameSimulation:
    """
    快速创建仿真实例
    Quickly create simulation instance
    
    Args:
        num_agents: Agent数量
        strategy_config: 策略配置 {agent_name: strategy_name} 或 None(全部用tit_for_tat)
        game_type: 博弈类型
        network_type: 网络类型
        rounds: 轮数
    
    Returns:
        GameSimulation实例
    """
    from .games import GAME_REGISTRY
    from .network import create_network
    
    # 创建agent名称
    agent_names = [f"Agent_{i}" for i in range(num_agents)]
    
    # 设置默认策略
    if strategy_config is None:
        strategy_config = {name: "tit_for_tat" for name in agent_names}
    
    # 创建agents
    agents = {}
    for name in agent_names:
        strategy_name = strategy_config.get(name, "tit_for_tat")
        strategy = create_strategy(strategy_name)
        agents[name] = AgentState(
            name=name,
            strategy=strategy,
            description=f"Agent using {strategy_name} strategy",
        )
    
    # 获取博弈配置
    game_config = GAME_REGISTRY.get(game_type, PRISONERS_DILEMMA)
    
    # 创建网络
    network = create_network(network_type, agent_names, **kwargs)
    
    return GameSimulation(
        agents=agents,
        game_config=game_config,
        network=network,
        rounds=rounds,
    )


def run_quick_experiment(
    strategies: List[str] = None,
    game_type: str = "prisoners_dilemma",
    network_type: str = "fully_connected",
    rounds: int = 50,
    verbose: bool = True,
) -> Dict:
    """
    快速运行实验
    Run quick experiment
    
    Args:
        strategies: 策略列表（每个agent一个策略）
        game_type: 博弈类型
        network_type: 网络类型
        rounds: 轮数
        verbose: 是否打印详细信息
    
    Returns:
        实验结果
    """
    if strategies is None:
        strategies = ["tit_for_tat", "always_cooperate", "always_defect", "random", "pavlov"]
    
    num_agents = len(strategies)
    agent_names = [f"Agent_{i}" for i in range(num_agents)]
    strategy_config = dict(zip(agent_names, strategies))
    
    sim = create_simulation(
        num_agents=num_agents,
        strategy_config=strategy_config,
        game_type=game_type,
        network_type=network_type,
        rounds=rounds,
    )
    sim.verbose = verbose
    
    return sim.run()
