"""
博弈定义模块 - 定义各种博弈类型和Payoff矩阵
Game Theory Module - Define game types and payoff matrices
"""
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Tuple, List


class Action(Enum):
    """博弈动作 / Game Actions"""
    COOPERATE = "cooperate"
    DEFECT = "defect"


@dataclass
class GameConfig:
    """
    博弈配置类
    Game Configuration Class
    
    Attributes:
        name: 博弈名称
        payoff_matrix: 收益矩阵 {(action1, action2): (payoff1, payoff2)}
        description: 博弈描述
    """
    name: str
    payoff_matrix: Dict[Tuple[Action, Action], Tuple[float, float]]
    description: str
    description_cn: str  # 中文描述


# ============================================================
# 经典博弈定义 / Classic Game Definitions
# ============================================================

# 囚徒困境 Prisoner's Dilemma
# T > R > P > S, 2R > T + S
# T=5 (Temptation), R=3 (Reward), P=1 (Punishment), S=0 (Sucker)
PRISONERS_DILEMMA = GameConfig(
    name="Prisoner's Dilemma",
    payoff_matrix={
        (Action.COOPERATE, Action.COOPERATE): (3, 3),   # R, R - 双方合作
        (Action.COOPERATE, Action.DEFECT): (0, 5),      # S, T - 我合作，对方背叛
        (Action.DEFECT, Action.COOPERATE): (5, 0),      # T, S - 我背叛，对方合作
        (Action.DEFECT, Action.DEFECT): (1, 1),         # P, P - 双方背叛
    },
    description="Classic Prisoner's Dilemma: Mutual cooperation yields moderate reward, "
                "but defection tempts with higher individual payoff.",
    description_cn="经典囚徒困境：双方合作获得中等收益，但背叛诱惑着更高的个人收益。"
)

# 雪堆博弈 / 鹰鸽博弈 Snowdrift / Hawk-Dove Game
# T > R > S > P
SNOWDRIFT = GameConfig(
    name="Snowdrift Game",
    payoff_matrix={
        (Action.COOPERATE, Action.COOPERATE): (3, 3),
        (Action.COOPERATE, Action.DEFECT): (1, 5),
        (Action.DEFECT, Action.COOPERATE): (5, 1),
        (Action.DEFECT, Action.DEFECT): (0, 0),
    },
    description="Snowdrift/Hawk-Dove: Unlike PD, mutual defection is worst outcome. "
                "Better to cooperate if opponent defects.",
    description_cn="雪堆博弈：与囚徒困境不同，双方背叛是最差结果。如果对手背叛，合作反而更好。"
)

# 猎鹿博弈 Stag Hunt
# R > T > P > S
STAG_HUNT = GameConfig(
    name="Stag Hunt",
    payoff_matrix={
        (Action.COOPERATE, Action.COOPERATE): (5, 5),   # 一起猎鹿
        (Action.COOPERATE, Action.DEFECT): (0, 3),      # 我猎鹿，对方猎兔
        (Action.DEFECT, Action.COOPERATE): (3, 0),      # 我猎兔，对方猎鹿
        (Action.DEFECT, Action.DEFECT): (2, 2),         # 都猎兔
    },
    description="Stag Hunt: Cooperation yields highest reward but requires trust. "
                "Safe defection gives guaranteed but lower payoff.",
    description_cn="猎鹿博弈：合作产生最高收益但需要信任。安全的背叛给出有保障但较低的收益。"
)

# 和谐博弈 Harmony Game
# R > T, R > S > P (合作总是最优)
HARMONY = GameConfig(
    name="Harmony Game",
    payoff_matrix={
        (Action.COOPERATE, Action.COOPERATE): (4, 4),
        (Action.COOPERATE, Action.DEFECT): (2, 3),
        (Action.DEFECT, Action.COOPERATE): (3, 2),
        (Action.DEFECT, Action.DEFECT): (1, 1),
    },
    description="Harmony: Cooperation is always the best strategy regardless of opponent.",
    description_cn="和谐博弈：无论对手如何，合作总是最佳策略。"
)


# ============================================================
# 博弈注册表 / Game Registry
# ============================================================

GAME_REGISTRY = {
    "prisoners_dilemma": PRISONERS_DILEMMA,
    "snowdrift": SNOWDRIFT,
    "stag_hunt": STAG_HUNT,
    "harmony": HARMONY,
}


# ============================================================
# 工具函数 / Utility Functions
# ============================================================

def get_payoff(game: GameConfig, action1: Action, action2: Action) -> Tuple[float, float]:
    """
    计算双方收益
    Calculate payoffs for both players
    
    Args:
        game: 博弈配置
        action1: 玩家1的动作
        action2: 玩家2的动作
    
    Returns:
        (玩家1收益, 玩家2收益)
    """
    return game.payoff_matrix[(action1, action2)]


def get_payoff_description(game: GameConfig) -> str:
    """
    生成收益矩阵的文字描述（用于LLM prompt）
    Generate text description of payoff matrix (for LLM prompt)
    """
    cc = game.payoff_matrix[(Action.COOPERATE, Action.COOPERATE)]
    cd = game.payoff_matrix[(Action.COOPERATE, Action.DEFECT)]
    dc = game.payoff_matrix[(Action.DEFECT, Action.COOPERATE)]
    dd = game.payoff_matrix[(Action.DEFECT, Action.DEFECT)]
    
    return f"""Payoff Matrix for {game.name}:
- Both Cooperate: You get {cc[0]}, Opponent gets {cc[1]}
- You Cooperate, Opponent Defects: You get {cd[0]}, Opponent gets {cd[1]}
- You Defect, Opponent Cooperates: You get {dc[0]}, Opponent gets {dc[1]}
- Both Defect: You get {dd[0]}, Opponent gets {dd[1]}"""


def action_from_string(s: str) -> Action:
    """
    从字符串解析动作
    Parse action from string
    """
    s = s.lower().strip()
    if s in ["cooperate", "c", "合作"]:
        return Action.COOPERATE
    elif s in ["defect", "d", "背叛", "叛变"]:
        return Action.DEFECT
    else:
        raise ValueError(f"Unknown action: {s}")
