"""
策略库 - 定义各种博弈策略
Strategy Library - Define various game strategies

注意: LLMStrategy 已移至独立模块 llm_strategy.py
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, TYPE_CHECKING
import random

from .games import Action

if TYPE_CHECKING:
    from persona.persona import Persona


class Strategy(ABC):
    """
    策略基类
    Base Strategy Class

    所有策略必须实现 choose_action 方法
    All strategies must implement choose_action method
    """
    name: str = "Base Strategy"
    description: str = "Base strategy class"
    description_cn: str = "策略基类"

    @abstractmethod
    def choose_action(self,
                      history: List[Tuple[Action, Action]],
                      opponent_name: str = None) -> Action:
        """
        根据历史选择动作
        Choose action based on history

        Args:
            history: 博弈历史 [(我的动作, 对手动作), ...]
            opponent_name: 对手名称（可选）

        Returns:
            选择的动作
        """
        pass

    def reset(self):
        """重置策略状态（如果有的话）"""
        pass


# ============================================================
# 固定策略 / Fixed Strategies
# ============================================================

class AlwaysCooperate(Strategy):
    """永远合作 / Always Cooperate"""
    name = "Always Cooperate"
    description = "Always choose to cooperate regardless of opponent's actions"
    description_cn = "无论对手如何，始终选择合作"

    def choose_action(self, history, opponent_name=None) -> Action:
        return Action.COOPERATE


class AlwaysDefect(Strategy):
    """永远背叛 / Always Defect"""
    name = "Always Defect"
    description = "Always choose to defect regardless of opponent's actions"
    description_cn = "无论对手如何，始终选择背叛"

    def choose_action(self, history, opponent_name=None) -> Action:
        return Action.DEFECT


class RandomStrategy(Strategy):
    """随机策略 / Random Strategy"""
    name = "Random"
    description = "Randomly choose cooperate or defect with equal probability"
    description_cn = "以相等概率随机选择合作或背叛"

    def __init__(self, coop_prob: float = 0.5):
        """
        Args:
            coop_prob: 合作概率，默认0.5
        """
        self.coop_prob = coop_prob

    def choose_action(self, history, opponent_name=None) -> Action:
        return Action.COOPERATE if random.random() < self.coop_prob else Action.DEFECT


# ============================================================
# 条件策略 / Conditional Strategies
# ============================================================

class TitForTat(Strategy):
    """
    以牙还牙 / Tit for Tat
    最著名的博弈策略之一，由Axelrod锦标赛中胜出
    """
    name = "Tit for Tat"
    description = "Start with cooperation, then copy opponent's last move"
    description_cn = "第一轮合作，之后模仿对手上一轮的动作"

    def choose_action(self, history, opponent_name=None) -> Action:
        if not history:
            return Action.COOPERATE
        # 返回对手上一轮的动作
        return history[-1][1]


class TitForTwoTats(Strategy):
    """
    以牙还两牙 / Tit for Two Tats
    更宽容的版本，对手连续背叛两次才报复
    """
    name = "Tit for Two Tats"
    description = "Defect only if opponent defected in last TWO rounds"
    description_cn = "只有对手连续两轮背叛才报复"

    def choose_action(self, history, opponent_name=None) -> Action:
        if len(history) < 2:
            return Action.COOPERATE
        # 检查对手最近两轮是否都背叛
        if history[-1][1] == Action.DEFECT and history[-2][1] == Action.DEFECT:
            return Action.DEFECT
        return Action.COOPERATE


class GrimTrigger(Strategy):
    """
    冷酷触发 / Grim Trigger
    一旦对手背叛，永远背叛
    """
    name = "Grim Trigger"
    description = "Cooperate until opponent defects once, then always defect"
    description_cn = "开始合作，一旦对手背叛就永远背叛"

    def __init__(self):
        self.triggered = False

    def choose_action(self, history, opponent_name=None) -> Action:
        if self.triggered:
            return Action.DEFECT

        # 检查对手是否曾经背叛
        for my_action, opp_action in history:
            if opp_action == Action.DEFECT:
                self.triggered = True
                return Action.DEFECT

        return Action.COOPERATE

    def reset(self):
        self.triggered = False


class Pavlov(Strategy):
    """
    巴甫洛夫策略 / Win-Stay, Lose-Shift
    如果上一轮收益高就重复，收益低就改变
    """
    name = "Pavlov"
    description = "Repeat last action if it gave good payoff, otherwise switch"
    description_cn = "如果上一轮结果好就重复，否则改变选择"

    def choose_action(self, history, opponent_name=None) -> Action:
        if not history:
            return Action.COOPERATE

        my_last, opp_last = history[-1]

        # 如果双方动作相同（CC或DD），保持当前策略
        # 如果动作不同（CD或DC），切换策略
        if my_last == opp_last:
            return my_last
        else:
            return Action.COOPERATE if my_last == Action.DEFECT else Action.DEFECT


class SuspiciousTitForTat(Strategy):
    """
    怀疑的以牙还牙 / Suspicious Tit for Tat
    先背叛，之后模仿对手
    """
    name = "Suspicious Tit for Tat"
    description = "Start with defection, then copy opponent's last move"
    description_cn = "第一轮背叛，之后模仿对手上一轮的动作"

    def choose_action(self, history, opponent_name=None) -> Action:
        if not history:
            return Action.DEFECT
        return history[-1][1]


class GradualStrategy(Strategy):
    """
    渐进策略 / Gradual
    对手每背叛一次，就连续报复N次（N=对手总背叛次数），然后合作两次示好
    """
    name = "Gradual"
    description = "Punish defection with increasing retaliation, then reconcile"
    description_cn = "对背叛进行递增报复，然后和解"

    def __init__(self):
        self.defect_count = 0      # 对手背叛总次数
        self.punish_remaining = 0  # 剩余惩罚轮数
        self.calm_remaining = 0    # 剩余冷静轮数

    def choose_action(self, history, opponent_name=None) -> Action:
        # 如果在惩罚阶段
        if self.punish_remaining > 0:
            self.punish_remaining -= 1
            return Action.DEFECT

        # 如果在冷静阶段
        if self.calm_remaining > 0:
            self.calm_remaining -= 1
            return Action.COOPERATE

        # 检查对手上一轮是否背叛
        if history and history[-1][1] == Action.DEFECT:
            self.defect_count += 1
            self.punish_remaining = self.defect_count - 1  # 本轮也算一次
            self.calm_remaining = 2
            return Action.DEFECT

        return Action.COOPERATE

    def reset(self):
        self.defect_count = 0
        self.punish_remaining = 0
        self.calm_remaining = 0


# ============================================================
# 基于概率的策略 / Probability-based Strategies
# ============================================================

class ProbabilisticCooperator(Strategy):
    """
    概率合作者
    根据对手历史合作率调整自己的合作概率
    """
    name = "Probabilistic Cooperator"
    description = "Match opponent's cooperation rate probabilistically"
    description_cn = "概率性地匹配对手的合作率"

    def __init__(self, base_coop: float = 0.5):
        self.base_coop = base_coop

    def choose_action(self, history, opponent_name=None) -> Action:
        if not history:
            return Action.COOPERATE if random.random() < self.base_coop else Action.DEFECT

        # 计算对手合作率
        opp_coop_count = sum(1 for _, opp in history if opp == Action.COOPERATE)
        opp_coop_rate = opp_coop_count / len(history)

        return Action.COOPERATE if random.random() < opp_coop_rate else Action.DEFECT


# ============================================================
# LLMStrategy 已移至 llm_strategy.py
# 使用方式: from game_theory.llm_strategy import LLMStrategy
# ============================================================


# ============================================================
# 策略注册表 / Strategy Registry
# ============================================================

STRATEGY_REGISTRY = {
    # 固定策略
    "always_cooperate": AlwaysCooperate,
    "always_defect": AlwaysDefect,
    "random": RandomStrategy,

    # 条件策略
    "tit_for_tat": TitForTat,
    "tit_for_two_tats": TitForTwoTats,
    "grim_trigger": GrimTrigger,
    "pavlov": Pavlov,
    "suspicious_tit_for_tat": SuspiciousTitForTat,
    "gradual": GradualStrategy,

    # 概率策略
    "probabilistic": ProbabilisticCooperator,

    # LLM策略 - 延迟导入避免循环
    # "llm": LLMStrategy,  # 使用时从 llm_strategy.py 单独导入
}


def create_strategy(strategy_name: str, **kwargs) -> Strategy:
    """
    工厂函数：根据名称创建策略实例
    Factory function: Create strategy instance by name

    Args:
        strategy_name: 策略名称
        **kwargs: 策略参数

    Returns:
        策略实例
    """
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy_name}. "
                        f"Available: {list(STRATEGY_REGISTRY.keys())}")

    return STRATEGY_REGISTRY[strategy_name](**kwargs)
