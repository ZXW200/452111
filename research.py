"""
博弈论 LLM 多智能体研究实验脚本 v9
Game Theory LLM Multi-Agent Research Experiments

实验列表:
1. Pure vs Hybrid - LLM自己分析 vs 代码辅助
2. 记忆视窗对比 - 5/10/20/全部历史
3. 多LLM对比 - DeepSeek vs GPT vs Gemini
4. Cheap Talk - 语言交流博弈
5. 群体动力学 - 多人混合群体
6. Baseline 对比 - LLM vs 经典策略

所有实验默认遍历三种博弈: 囚徒困境 / 雪堆博弈 / 猎鹿博弈
结果按 results/{时间戳}/{博弈类型}/ 分目录保存
"""

import json
import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 博弈论模块导入
from game_theory.games import (
    PRISONERS_DILEMMA, SNOWDRIFT, STAG_HUNT,
    Action, GameConfig, get_payoff, get_payoff_description, GAME_REGISTRY
)
from game_theory.llm_strategy import LLMStrategy
from game_theory.strategies import (
    TitForTat, AlwaysCooperate, AlwaysDefect,
    GrimTrigger, Pavlov, RandomStrategy
)
from game_theory.network import (
    FullyConnectedNetwork, SmallWorldNetwork, ScaleFreeNetwork, NETWORK_REGISTRY
)
from game_theory.simulation import AgentState, GameSimulation


# ============================================================
# 全局配置
# ============================================================

GAME_NAMES_CN = {
    "prisoners_dilemma": "囚徒困境",
    "snowdrift": "雪堆博弈",
    "stag_hunt": "猎鹿博弈",
}

NETWORK_NAMES_CN = {
    "fully_connected": "完全连接",
    "small_world": "小世界",
    "scale_free": "无标度",
}

# 默认实验参数
DEFAULT_CONFIG = {
    "n_repeats": 3,      # 重复次数（论文建议30次）
    "rounds": 20,        # 每次对局轮数
    "provider": "deepseek",  # 默认LLM
    "verbose": True,
}


# ============================================================
# 结果保存管理
# ============================================================

class ResultManager:
    """
    实验结果管理器

    目录结构:
    results/
    └── 20250121_143052/           # 时间戳
        ├── experiment_config.json  # 实验配置
        ├── details/                # 每次实验详细数据
        │   └── {实验名}_{模型名}_{次数}_{轮数}.json
        ├── summary/                # 汇总报告 (CSV 格式)
        │   └── {实验名}.csv
        ├── prisoners_dilemma/      # 博弈类型
        │   ├── pure_vs_hybrid.json
        │   └── pure_vs_hybrid.png
        ├── snowdrift/
        └── stag_hunt/
    """

    def __init__(self, base_dir: str = "results"):
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.root_dir = os.path.join(base_dir, self.timestamp)
        os.makedirs(self.root_dir, exist_ok=True)

        # 创建 details 和 summary 目录
        self.details_dir = os.path.join(self.root_dir, "details")
        self.summary_dir = os.path.join(self.root_dir, "summary")
        os.makedirs(self.details_dir, exist_ok=True)
        os.makedirs(self.summary_dir, exist_ok=True)

        # 为每个博弈创建子目录
        for game_name in GAME_REGISTRY.keys():
            game_dir = os.path.join(self.root_dir, game_name)
            os.makedirs(game_dir, exist_ok=True)

        print(f"Results dir: {self.root_dir}")

    def get_game_dir(self, game_name: str) -> str:
        """获取博弈类型目录"""
        return os.path.join(self.root_dir, game_name)

    def save_json(self, game_name: str, experiment_name: str, data: Dict) -> str:
        """保存 JSON 数据"""
        game_dir = self.get_game_dir(game_name)
        filepath = os.path.join(game_dir, f"{experiment_name}.json")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        print(f"  Saved: {filepath}")
        return filepath

    def save_figure(self, game_name: str, experiment_name: str, fig: plt.Figure) -> str:
        """保存图表"""
        game_dir = self.get_game_dir(game_name)
        filepath = os.path.join(game_dir, f"{experiment_name}.png")

        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  Saved: {filepath}")
        return filepath

    def save_config(self, config: Dict):
        """保存实验配置"""
        filepath = os.path.join(self.root_dir, "experiment_config.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"Config saved: {filepath}")

    def save_summary(self, all_results: Dict):
        """保存汇总报告到根目录"""
        filepath = os.path.join(self.root_dir, "summary.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
        print(f"Summary saved: {filepath}")

    def save_transcript(self, game_name: str, experiment_name: str, content: str) -> str:
        """保存易读的 transcript 文本文件"""
        game_dir = self.get_game_dir(game_name)
        filepath = os.path.join(game_dir, f"{experiment_name}_transcript.txt")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"  Saved: {filepath}")
        return filepath

    def save_detail(self, experiment_name: str, provider: str, trial: int, rounds: int, data: Dict) -> str:
        """保存单次实验详细数据到 details 目录"""
        filename = f"{experiment_name}_{provider}_{trial}_{rounds}.json"
        filepath = os.path.join(self.details_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        return filepath

    def save_round_records(self, experiment_name: str, game_name: str, provider: str, records: List[Dict]) -> str:
        """
        保存每轮记录到单个文件

        Args:
            experiment_name: 实验名称
            game_name: 博弈类型
            provider: LLM提供商
            records: 每轮记录列表
        """
        filename = f"{experiment_name}_{game_name}_{provider}_rounds.json"
        filepath = os.path.join(self.details_dir, filename)

        data = {
            "experiment": experiment_name,
            "game": game_name,
            "provider": provider,
            "total_records": len(records),
            "records": records,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        print(f"  Rounds: {filepath}")
        return filepath

    def save_experiment_summary(self, experiment_name: str, data: Dict) -> str:
        """保存实验汇总到 summary 目录 (CSV 格式)"""
        filepath = os.path.join(self.summary_dir, f"{experiment_name}.csv")

        rows = self._flatten_summary_to_rows(experiment_name, data)
        if rows:
            fieldnames = ["experiment", "game", "condition", "payoff", "coop_rate"]
            with open(filepath, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        print(f"  Summary: {filepath}")
        return filepath

    def _flatten_summary_to_rows(self, experiment_name: str, data: Dict) -> List[Dict]:
        """将嵌套的实验数据展平为 CSV 行"""
        rows = []

        for game_name, game_data in data.items():
            if isinstance(game_data, dict):
                for key, stats in game_data.items():
                    if isinstance(stats, dict) and "payoff" in stats:
                        row = self._make_summary_row(experiment_name, game_name, key, stats)
                        rows.append(row)
                    elif isinstance(stats, dict) and "payoffs" in stats:
                        # group_dynamics 结构: payoffs/coop_rates/rankings
                        rows.extend(self._make_group_summary_rows(experiment_name, game_name, key, stats))
                    elif isinstance(stats, dict):
                        for sub_key, sub_stats in stats.items():
                            if isinstance(sub_stats, dict) and "payoff" in sub_stats:
                                row = self._make_summary_row(experiment_name, game_name, f"{key}_{sub_key}", sub_stats)
                                rows.append(row)

        return rows

    def _make_group_summary_rows(self, experiment: str, game: str, network: str, stats: Dict) -> List[Dict]:
        """生成 group_dynamics 的 summary 行"""
        rows = []
        payoffs = stats.get("payoffs", {})
        coop_rates = stats.get("coop_rates", {})

        for agent_id, payoff in payoffs.items():
            coop = coop_rates.get(agent_id, 0)
            rows.append({
                "experiment": experiment,
                "game": game,
                "condition": f"{network}_{agent_id}",
                "payoff": f"{payoff:.1f}",
                "coop_rate": f"{coop * 100:.1f}%",
            })

        return rows

    def _make_summary_row(self, experiment: str, game: str, condition: str, stats: Dict) -> Dict:
        """生成单行 summary 数据"""
        payoff = stats.get("payoff", {})
        coop = stats.get("coop_rate", {})

        payoff_str = f"{payoff.get('mean', 0):.1f} ± {payoff.get('std', 0):.1f}"
        coop_str = f"{coop.get('mean', 0) * 100:.1f}%"

        return {
            "experiment": experiment,
            "game": game,
            "condition": condition,
            "payoff": payoff_str,
            "coop_rate": coop_str,
        }


# ============================================================
# 统计工具
# ============================================================

def compute_statistics(values: List[float]) -> Dict:
    """计算统计量 + 95% 置信区间"""
    if not values:
        return {"mean": 0, "std": 0, "ci_low": 0, "ci_high": 0, "n": 0}

    arr = np.array(values)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1) if len(arr) > 1 else 0
    n = len(arr)

    if n > 1:
        se = std / np.sqrt(n)
        ci_low = mean - 1.96 * se
        ci_high = mean + 1.96 * se
    else:
        ci_low = ci_high = mean

    return {
        "mean": round(mean, 3),
        "std": round(std, 3),
        "ci_low": round(ci_low, 3),
        "ci_high": round(ci_high, 3),
        "n": n
    }


def compute_cooperation_rate(history: List[Action]) -> float:
    """计算合作率"""
    if not history:
        return 0.0
    cooperations = sum(1 for a in history if a == Action.COOPERATE)
    return cooperations / len(history)


def make_history_tuples(my_history: List[Action], opp_history: List[Action]) -> List[Tuple[Action, Action]]:
    """
    将两个独立的历史列表转换为元组列表
    用于兼容传统策略的接口

    Args:
        my_history: 我的动作历史
        opp_history: 对手动作历史

    Returns:
        [(我的动作, 对手动作), ...]
    """
    return list(zip(my_history, opp_history))


def print_separator(title: str = "", char: str = "=", width: int = 60):
    """打印分隔线"""
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"\n{char * padding} {title} {char * padding}")
    else:
        print(char * width)


def print_game_header(game_name: str):
    """打印博弈类型标题"""
    cn_name = GAME_NAMES_CN.get(game_name, game_name)
    print(f"\n{'─' * 50}")
    print(f"  Game: {cn_name}")
    print(f"{'─' * 50}")


# ============================================================
# 可视化工具
# ============================================================

def plot_comparison_bar(
    data: Dict[str, Dict],
    title: str,
    ylabel: str = "得分",
    game_name: str = "",
) -> plt.Figure:
    """绘制对比柱状图"""

    fig, ax = plt.subplots(figsize=(10, 6))

    labels = list(data.keys())
    means = [d["payoff"]["mean"] for d in data.values()]
    stds = [d["payoff"]["std"] for d in data.values()]

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color='steelblue', alpha=0.8)

    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} - {GAME_NAMES_CN.get(game_name, game_name)}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # 添加数值标签
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def plot_cooperation_comparison(
    data: Dict[str, Dict],
    title: str,
    game_name: str = "",
) -> plt.Figure:
    """绘制得分和合作率对比图"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    labels = list(data.keys())

    # 得分图
    means = [d["payoff"]["mean"] for d in data.values()]
    stds = [d["payoff"]["std"] for d in data.values()]
    x = np.arange(len(labels))
    bars1 = ax1.bar(x, means, yerr=stds, capsize=5, color='steelblue', alpha=0.8)
    ax1.set_ylabel("得分")
    ax1.set_title("得分对比")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')

    for bar, mean in zip(bars1, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=9)

    # 合作率图
    coop_means = [d["coop_rate"]["mean"] * 100 for d in data.values()]
    coop_stds = [d["coop_rate"]["std"] * 100 for d in data.values()]
    bars2 = ax2.bar(x, coop_means, yerr=coop_stds, capsize=5, color='forestgreen', alpha=0.8)
    ax2.set_ylabel("合作率 (%)")
    ax2.set_title("合作率对比")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_ylim(0, 105)

    for bar, mean in zip(bars2, coop_means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=9)

    fig.suptitle(f"{title} - {GAME_NAMES_CN.get(game_name, game_name)}", fontsize=14)
    plt.tight_layout()
    return fig


# ============================================================
# 实验1: Pure vs Hybrid
# ============================================================

def experiment_pure_vs_hybrid(
    result_manager: ResultManager,
    provider: str = DEFAULT_CONFIG["provider"],
    n_repeats: int = DEFAULT_CONFIG["n_repeats"],
    rounds: int = DEFAULT_CONFIG["rounds"],
    games: List[str] = None,
) -> Dict:
    """
    对比 Pure 和 Hybrid 模式
    """

    if games is None:
        games = list(GAME_REGISTRY.keys())

    print_separator("实验1: Pure vs Hybrid LLM")
    print("Pure:   LLM 自己从历史分析对手")
    print("Hybrid: 代码分析好告诉 LLM")
    print(f"Provider: {provider} | Repeats: {n_repeats} | Rounds: {rounds}")

    all_results = {}

    for game_name in games:
        game_config = GAME_REGISTRY[game_name]
        print_game_header(game_name)

        results = {"pure": [], "hybrid": []}
        coop_rates = {"pure": [], "hybrid": []}
        all_round_records = []  # 收集所有轮次记录

        for mode in ["pure", "hybrid"]:
            print(f"\n  Mode: {mode.upper()}")

            for trial in range(n_repeats):
                print(f"    Trial {trial + 1}/{n_repeats}...", end=" ", flush=True)

                try:
                    llm_strategy = LLMStrategy(
                        provider=provider,
                        mode=mode,
                        game_config=game_config,
                    )

                    opponent = TitForTat()

                    llm_payoff = 0
                    llm_history = []
                    opp_history = []

                    for r in range(rounds):
                        llm_action = llm_strategy.choose_action(llm_history, opp_history)
                        opp_action = opponent.choose_action(make_history_tuples(opp_history, llm_history))

                        payoff, _ = get_payoff(game_config, llm_action, opp_action)
                        llm_payoff += payoff

                        # 记录每轮数据
                        llm_response = llm_strategy.raw_responses[-1] if llm_strategy.raw_responses else ""
                        all_round_records.append({
                            "mode": mode,
                            "trial": trial + 1,
                            "round": r + 1,
                            "llm_response": llm_response,
                            "llm_action": llm_action.name,
                            "opp_action": opp_action.name,
                            "payoff": payoff,
                            "cumulative_payoff": llm_payoff,
                        })

                        llm_history.append(llm_action)
                        opp_history.append(opp_action)

                    coop_rate = compute_cooperation_rate(llm_history)
                    results[mode].append(llm_payoff)
                    coop_rates[mode].append(coop_rate)

                    # 获取解析质量（兼容不同版本的 LLMStrategy）
                    if hasattr(llm_strategy, 'get_parse_quality'):
                        parse_quality = llm_strategy.get_parse_quality()
                        success_rate = parse_quality.get('success_rate', 0)
                    elif hasattr(llm_strategy, 'parser'):
                        parse_quality = llm_strategy.parser.get_stats()
                        success_rate = parse_quality.get('success_rate', 0)
                    else:
                        success_rate = 0

                    # 保存详细数据（包含LLM思考过程）
                    detail_data = {
                        "experiment": "pure_vs_hybrid",
                        "game": game_name,
                        "mode": mode,
                        "trial": trial + 1,
                        "rounds": rounds,
                        "payoff": llm_payoff,
                        "coop_rate": coop_rate,
                        "parse_success_rate": success_rate,
                        "llm_history": [a.name for a in llm_history],
                        "opp_history": [a.name for a in opp_history],
                        "llm_responses": llm_strategy.raw_responses.copy(),  # LLM思考过程
                    }
                    result_manager.save_detail(f"pure_vs_hybrid_{game_name}_{mode}", provider, trial + 1, rounds, detail_data)

                    print(f"Payoff: {llm_payoff:.1f}, Coop rate: {coop_rate:.1%}, Parse: {success_rate:.0%}")

                except Exception as e:
                    print(f"Error: {e}")
                    continue

        # 统计当前博弈结果
        game_results = {
            "pure": {
                "payoff": compute_statistics(results["pure"]),
                "coop_rate": compute_statistics(coop_rates["pure"]),
            },
            "hybrid": {
                "payoff": compute_statistics(results["hybrid"]),
                "coop_rate": compute_statistics(coop_rates["hybrid"]),
            },
        }

        all_results[game_name] = game_results

        # 保存当前博弈结果
        result_manager.save_json(game_name, "pure_vs_hybrid", game_results)

        # 保存每轮记录
        result_manager.save_round_records("pure_vs_hybrid", game_name, provider, all_round_records)

        # 生成并保存图表
        fig = plot_cooperation_comparison(game_results, "Pure vs Hybrid", game_name)
        result_manager.save_figure(game_name, "pure_vs_hybrid", fig)

    # 打印汇总
    _print_pure_vs_hybrid_summary(all_results)

    # 保存实验汇总
    result_manager.save_experiment_summary("pure_vs_hybrid", all_results)

    return all_results


def _print_pure_vs_hybrid_summary(results: Dict):
    """打印 Pure vs Hybrid 汇总"""
    print_separator("汇总: Pure vs Hybrid")
    print(f"{'Game':<12} {'Pure Payoff':<18} {'Hybrid Payoff':<18} {'Pure Coop':<14} {'Hybrid Coop':<14}")
    print("-" * 76)

    for game_name, stats in results.items():
        cn_name = GAME_NAMES_CN.get(game_name, game_name)

        pure_pay = stats["pure"]["payoff"]
        hybrid_pay = stats["hybrid"]["payoff"]
        pure_coop = stats["pure"]["coop_rate"]
        hybrid_coop = stats["hybrid"]["coop_rate"]

        pure_str = f"{pure_pay['mean']:.1f} ± {pure_pay['std']:.1f}"
        hybrid_str = f"{hybrid_pay['mean']:.1f} ± {hybrid_pay['std']:.1f}"
        pure_coop_str = f"{pure_coop['mean']:.1%}"
        hybrid_coop_str = f"{hybrid_coop['mean']:.1%}"

        print(f"{cn_name:<12} {pure_str:<18} {hybrid_str:<18} {pure_coop_str:<14} {hybrid_coop_str:<14}")


# ============================================================
# 实验2: 记忆视窗对比
# ============================================================

def experiment_memory_window(
    result_manager: ResultManager,
    provider: str = DEFAULT_CONFIG["provider"],
    n_repeats: int = DEFAULT_CONFIG["n_repeats"],
    rounds: int = 30,
    windows: List[Optional[int]] = [5, 10, 20, None],
    games: List[str] = None,
) -> Dict:
    """记忆视窗对比实验"""

    if games is None:
        games = list(GAME_REGISTRY.keys())

    print_separator("实验2: 记忆视窗对比")
    print(f"测试不同历史记忆长度: {windows}")
    print(f"Provider: {provider} | Repeats: {n_repeats} | Rounds: {rounds}")

    all_results = {}

    for game_name in games:
        game_config = GAME_REGISTRY[game_name]
        print_game_header(game_name)

        window_results = {}
        all_round_records = []  # 收集所有轮次记录

        for window in windows:
            window_label = str(window) if window else "全部"
            print(f"\n  Window: {window_label}")

            payoffs = []
            coop_rates = []

            for trial in range(n_repeats):
                print(f"    Trial {trial + 1}/{n_repeats}...", end=" ", flush=True)

                try:
                    llm_strategy = LLMStrategy(
                        provider=provider,
                        mode="pure",
                        game_config=game_config,
                        history_window=window,
                    )

                    opponent = GrimTrigger()

                    llm_payoff = 0
                    llm_history = []
                    opp_history = []

                    for r in range(rounds):
                        llm_action = llm_strategy.choose_action(llm_history, opp_history)
                        opp_action = opponent.choose_action(make_history_tuples(opp_history, llm_history))

                        payoff, _ = get_payoff(game_config, llm_action, opp_action)
                        llm_payoff += payoff

                        # 记录每轮数据
                        llm_response = llm_strategy.raw_responses[-1] if llm_strategy.raw_responses else ""
                        all_round_records.append({
                            "window": window_label,
                            "trial": trial + 1,
                            "round": r + 1,
                            "llm_response": llm_response,
                            "llm_action": llm_action.name,
                            "opp_action": opp_action.name,
                            "payoff": payoff,
                            "cumulative_payoff": llm_payoff,
                        })

                        llm_history.append(llm_action)
                        opp_history.append(opp_action)

                    coop_rate = compute_cooperation_rate(llm_history)
                    payoffs.append(llm_payoff)
                    coop_rates.append(coop_rate)

                    print(f"Payoff: {llm_payoff:.1f}, Coop rate: {coop_rate:.1%}")

                except Exception as e:
                    print(f"Error: {e}")
                    continue

            window_results[window_label] = {
                "payoff": compute_statistics(payoffs),
                "coop_rate": compute_statistics(coop_rates),
            }

        all_results[game_name] = window_results

        # 保存结果
        result_manager.save_json(game_name, "memory_window", window_results)

        # 保存每轮记录
        result_manager.save_round_records("memory_window", game_name, provider, all_round_records)

        # 生成图表
        fig = plot_cooperation_comparison(window_results, "记忆视窗对比", game_name)
        result_manager.save_figure(game_name, "memory_window", fig)

    _print_window_summary(all_results)

    # 保存实验汇总
    result_manager.save_experiment_summary("memory_window", all_results)

    return all_results


def _print_window_summary(results: Dict):
    """打印记忆视窗汇总"""
    print_separator("汇总: 记忆视窗对比")

    for game_name, window_stats in results.items():
        cn_name = GAME_NAMES_CN.get(game_name, game_name)
        print(f"\n{cn_name}:")
        print(f"  {'Window':<8} {'Payoff':<18} {'Coop Rate':<12}")
        print(f"  {'-' * 38}")

        for window, stats in window_stats.items():
            pay = stats["payoff"]
            coop = stats["coop_rate"]
            pay_str = f"{pay['mean']:.1f} ± {pay['std']:.1f}"
            coop_str = f"{coop['mean']:.1%}"
            print(f"  {window:<8} {pay_str:<18} {coop_str:<12}")


# ============================================================
# 实验3: 多 LLM 对比
# ============================================================

def experiment_multi_llm(
    result_manager: ResultManager,
    providers: List[str] = ["deepseek", "openai", "gemini"],
    n_repeats: int = DEFAULT_CONFIG["n_repeats"],
    rounds: int = DEFAULT_CONFIG["rounds"],
    games: List[str] = None,
) -> Dict:
    """多 LLM 对比实验"""

    if games is None:
        games = list(GAME_REGISTRY.keys())

    print_separator("实验3: 多 LLM 对比")
    print(f"对比 LLM: {providers}")
    print(f"Repeats: {n_repeats} | Rounds: {rounds}")

    all_results = {}

    for game_name in games:
        game_config = GAME_REGISTRY[game_name]
        print_game_header(game_name)

        provider_results = {}
        all_round_records = []  # 收集所有轮次记录

        for provider in providers:
            print(f"\n  Provider: {provider.upper()}")

            payoffs = []
            coop_rates = []

            for trial in range(n_repeats):
                print(f"    Trial {trial + 1}/{n_repeats}...", end=" ", flush=True)

                try:
                    llm_strategy = LLMStrategy(
                        provider=provider,
                        mode="hybrid",
                        game_config=game_config,
                    )

                    opponent = TitForTat()

                    llm_payoff = 0
                    llm_history = []
                    opp_history = []

                    for r in range(rounds):
                        llm_action = llm_strategy.choose_action(llm_history, opp_history)
                        opp_action = opponent.choose_action(make_history_tuples(opp_history, llm_history))

                        payoff, _ = get_payoff(game_config, llm_action, opp_action)
                        llm_payoff += payoff

                        # 记录每轮数据
                        llm_response = llm_strategy.raw_responses[-1] if llm_strategy.raw_responses else ""
                        all_round_records.append({
                            "provider": provider,
                            "trial": trial + 1,
                            "round": r + 1,
                            "llm_response": llm_response,
                            "llm_action": llm_action.name,
                            "opp_action": opp_action.name,
                            "payoff": payoff,
                            "cumulative_payoff": llm_payoff,
                        })

                        llm_history.append(llm_action)
                        opp_history.append(opp_action)

                    coop_rate = compute_cooperation_rate(llm_history)
                    payoffs.append(llm_payoff)
                    coop_rates.append(coop_rate)

                    print(f"Payoff: {llm_payoff:.1f}, Coop rate: {coop_rate:.1%}")

                except Exception as e:
                    print(f"Error: {e}")
                    continue

            provider_results[provider] = {
                "payoff": compute_statistics(payoffs),
                "coop_rate": compute_statistics(coop_rates),
            }

        all_results[game_name] = provider_results

        # 保存结果
        result_manager.save_json(game_name, "multi_llm", provider_results)

        # 保存每轮记录
        result_manager.save_round_records("multi_llm", game_name, "all", all_round_records)

        # 生成图表
        fig = plot_cooperation_comparison(provider_results, "多 LLM 对比", game_name)
        result_manager.save_figure(game_name, "multi_llm", fig)

    _print_multi_llm_summary(all_results)

    # 保存实验汇总
    result_manager.save_experiment_summary("multi_llm", all_results)

    return all_results


def _print_multi_llm_summary(results: Dict):
    """打印多 LLM 对比汇总"""
    print_separator("汇总: 多 LLM 对比")

    for game_name, provider_stats in results.items():
        cn_name = GAME_NAMES_CN.get(game_name, game_name)
        print(f"\n{cn_name}:")
        print(f"  {'LLM':<12} {'Payoff':<18} {'Coop Rate':<12}")
        print(f"  {'-' * 42}")

        sorted_providers = sorted(
            provider_stats.items(),
            key=lambda x: x[1]["payoff"]["mean"],
            reverse=True
        )

        for provider, stats in sorted_providers:
            pay = stats["payoff"]
            coop = stats["coop_rate"]
            pay_str = f"{pay['mean']:.1f} ± {pay['std']:.1f}"
            coop_str = f"{coop['mean']:.1%}"
            print(f"  {provider:<12} {pay_str:<18} {coop_str:<12}")


# ============================================================
# 实验4: Cheap Talk (语言交流)
# ============================================================

def experiment_cheap_talk(
    result_manager: ResultManager,
    provider1: str = DEFAULT_CONFIG["provider"],
    provider2: str = None,
    n_repeats: int = DEFAULT_CONFIG["n_repeats"],
    rounds: int = DEFAULT_CONFIG["rounds"],
    games: List[str] = None,
) -> Dict:
    """Cheap Talk 实验 - LLM vs LLM 双向交流，支持不同 provider 对战"""

    if provider2 is None:
        provider2 = provider1

    if games is None:
        games = list(GAME_REGISTRY.keys())

    print_separator("实验4: Cheap Talk (语言交流)")
    print("对比: 无交流 vs 有语言交流 (LLM vs LLM)")
    if provider1 == provider2:
        print(f"Provider: {provider1} vs {provider2} | Repeats: {n_repeats} | Rounds: {rounds}")
    else:
        print(f"Provider: {provider1} vs {provider2} (跨模型对战) | Repeats: {n_repeats} | Rounds: {rounds}")

    all_results = {}

    for game_name in games:
        game_config = GAME_REGISTRY[game_name]
        print_game_header(game_name)

        results = {"no_talk": [], "cheap_talk": []}
        coop_rates = {"no_talk": {"player1": [], "player2": []}, "cheap_talk": {"player1": [], "player2": []}}
        promise_kept = {"player1": [], "player2": []}
        all_round_records = []  # 收集所有轮次记录

        # 详细记录：保存每轮的 message 和 action
        detailed_trials = {"no_talk": [], "cheap_talk": []}

        for mode in ["no_talk", "cheap_talk"]:
            print(f"\n  Mode: {mode}")

            for trial in range(n_repeats):
                print(f"    Trial {trial + 1}/{n_repeats}...", end=" ", flush=True)

                try:
                    use_cheap_talk = (mode == "cheap_talk")

                    # 创建两个 LLM 策略（可以是不同 provider）
                    llm1 = LLMStrategy(
                        provider=provider1,
                        mode="hybrid",
                        game_config=game_config,
                        enable_cheap_talk=use_cheap_talk,
                        agent_name=f"Player1({provider1})",
                    )

                    llm2 = LLMStrategy(
                        provider=provider2,
                        mode="hybrid",
                        game_config=game_config,
                        enable_cheap_talk=use_cheap_talk,
                        agent_name=f"Player2({provider2})",
                    )

                    total_payoff_1 = 0
                    total_payoff_2 = 0
                    history_1 = []
                    history_2 = []
                    messages_1 = []
                    messages_2 = []

                    # 记录每轮的详细数据
                    round_details = []

                    for r in range(rounds):
                        # 双方生成消息（如果启用cheap talk）
                        msg1 = ""
                        msg2 = ""
                        if use_cheap_talk:
                            if hasattr(llm1, 'generate_message'):
                                msg1 = llm1.generate_message(history_1, history_2, "Player2")
                                messages_1.append(msg1)
                            if hasattr(llm2, 'generate_message'):
                                msg2 = llm2.generate_message(history_2, history_1, "Player1")
                                messages_2.append(msg2)

                        # 双方选择动作，传入对方的消息
                        action1 = llm1.choose_action(history_1, history_2, "Player2", opponent_message=msg2)
                        action2 = llm2.choose_action(history_2, history_1, "Player1", opponent_message=msg1)

                        payoff1, payoff2 = get_payoff(game_config, action1, action2)
                        total_payoff_1 += payoff1
                        total_payoff_2 += payoff2

                        history_1.append(action1)
                        history_2.append(action2)

                        # 保存本轮详细记录
                        llm1_response = llm1.raw_responses[-1] if llm1.raw_responses else ""
                        llm2_response = llm2.raw_responses[-1] if llm2.raw_responses else ""
                        round_details.append({
                            "round": r + 1,
                            "player1_message": msg1,
                            "player2_message": msg2,
                            "player1_action": action1.name,
                            "player2_action": action2.name,
                            "player1_payoff": payoff1,
                            "player2_payoff": payoff2,
                            "player1_cumulative": total_payoff_1,
                            "player2_cumulative": total_payoff_2,
                        })

                        # 记录到总记录
                        all_round_records.append({
                            "mode": mode,
                            "trial": trial + 1,
                            "round": r + 1,
                            "player1_message": msg1,
                            "player2_message": msg2,
                            "player1_response": llm1_response,
                            "player2_response": llm2_response,
                            "player1_action": action1.name,
                            "player2_action": action2.name,
                            "player1_payoff": payoff1,
                            "player2_payoff": payoff2,
                        })

                    coop_rate_1 = compute_cooperation_rate(history_1)
                    coop_rate_2 = compute_cooperation_rate(history_2)
                    total_payoff = total_payoff_1 + total_payoff_2  # 社会总收益

                    results[mode].append(total_payoff)
                    coop_rates[mode]["player1"].append(coop_rate_1)
                    coop_rates[mode]["player2"].append(coop_rate_2)

                    # 保存本次 trial 的完整记录
                    trial_record = {
                        "trial": trial + 1,
                        "player1_payoff": total_payoff_1,
                        "player2_payoff": total_payoff_2,
                        "total_payoff": total_payoff,
                        "player1_coop_rate": coop_rate_1,
                        "player2_coop_rate": coop_rate_2,
                        "rounds": round_details,
                    }

                    if use_cheap_talk:
                        if messages_1:
                            kept1 = _analyze_promise_keeping(messages_1, history_1)
                            promise_kept["player1"].append(kept1)
                            trial_record["player1_promise_keeping"] = kept1
                        if messages_2:
                            kept2 = _analyze_promise_keeping(messages_2, history_2)
                            promise_kept["player2"].append(kept2)
                            trial_record["player2_promise_keeping"] = kept2

                    detailed_trials[mode].append(trial_record)

                    # 保存详细数据（包含LLM思考过程）
                    detail_data = {
                        "experiment": "cheap_talk",
                        "game": game_name,
                        "mode": mode,
                        "trial": trial + 1,
                        "rounds": rounds,
                        "player1_payoff": total_payoff_1,
                        "player2_payoff": total_payoff_2,
                        "total_payoff": total_payoff,
                        "player1_coop_rate": coop_rate_1,
                        "player2_coop_rate": coop_rate_2,
                        "player1_messages": messages_1 if use_cheap_talk else [],
                        "player2_messages": messages_2 if use_cheap_talk else [],
                        "player1_history": [a.name for a in history_1],
                        "player2_history": [a.name for a in history_2],
                        "player1_responses": llm1.raw_responses.copy(),
                        "player2_responses": llm2.raw_responses.copy(),
                    }
                    provider_label = f"{provider1}_vs_{provider2}" if provider1 != provider2 else provider1
                    result_manager.save_detail(f"cheap_talk_{game_name}_{mode}", provider_label, trial + 1, rounds, detail_data)

                    avg_coop = (coop_rate_1 + coop_rate_2) / 2
                    print(f"Total: {total_payoff:.1f}, Avg coop: {avg_coop:.1%}")

                except Exception as e:
                    print(f"Error: {e}")
                    continue

        # 计算统计结果
        game_results = {
            "no_talk": {
                "total_payoff": compute_statistics(results["no_talk"]),
                "player1_coop_rate": compute_statistics(coop_rates["no_talk"]["player1"]),
                "player2_coop_rate": compute_statistics(coop_rates["no_talk"]["player2"]),
            },
            "cheap_talk": {
                "total_payoff": compute_statistics(results["cheap_talk"]),
                "player1_coop_rate": compute_statistics(coop_rates["cheap_talk"]["player1"]),
                "player2_coop_rate": compute_statistics(coop_rates["cheap_talk"]["player2"]),
                "player1_promise_kept": compute_statistics(promise_kept["player1"]) if promise_kept["player1"] else None,
                "player2_promise_kept": compute_statistics(promise_kept["player2"]) if promise_kept["player2"] else None,
            },
        }

        all_results[game_name] = game_results

        # 保存统计结果
        result_manager.save_json(game_name, "cheap_talk", game_results)

        # 保存每轮记录
        provider_label = f"{provider1}_vs_{provider2}" if provider1 != provider2 else provider1
        result_manager.save_round_records("cheap_talk", game_name, provider_label, all_round_records)

        # 生成易读的 transcript 文本文件
        transcript = _generate_cheap_talk_transcript(game_name, provider1, provider2, detailed_trials)
        result_manager.save_transcript(game_name, "cheap_talk", transcript)

        # 生成图表 - 转换数据格式以适配 plot_cooperation_comparison
        plot_data = {}
        for mode_key in ["no_talk", "cheap_talk"]:
            mode_stats = game_results[mode_key]
            # 计算平均合作率
            avg_coop_mean = (mode_stats["player1_coop_rate"]["mean"] + mode_stats["player2_coop_rate"]["mean"]) / 2
            avg_coop_std = (mode_stats["player1_coop_rate"]["std"] + mode_stats["player2_coop_rate"]["std"]) / 2
            plot_data[mode_key] = {
                "payoff": mode_stats["total_payoff"],
                "coop_rate": {"mean": avg_coop_mean, "std": avg_coop_std},
            }
        fig = plot_cooperation_comparison(plot_data, "Cheap Talk 对比 (LLM vs LLM)", game_name)
        result_manager.save_figure(game_name, "cheap_talk", fig)

    _print_cheap_talk_summary(all_results)

    # 保存实验汇总
    result_manager.save_experiment_summary("cheap_talk", all_results)

    return all_results


def _generate_cheap_talk_transcript(game_name: str, provider1: str, provider2: str, detailed_trials: Dict) -> str:
    """生成易读的 Cheap Talk 交互记录 (LLM vs LLM)"""
    cn_name = GAME_NAMES_CN.get(game_name, game_name)

    lines = []
    lines.append("=" * 70)
    lines.append(f"CHEAP TALK 实验记录 - {cn_name}")
    lines.append(f"Player1: {provider1} | Player2: {provider2}")
    lines.append(f"对战模式: LLM vs LLM (双向交流)")
    lines.append("=" * 70)
    lines.append("")

    for mode in ["no_talk", "cheap_talk"]:
        mode_name = "无交流模式 (No Talk)" if mode == "no_talk" else "有交流模式 (Cheap Talk)"
        lines.append("-" * 70)
        lines.append(f"【{mode_name}】")
        lines.append("-" * 70)

        for trial_data in detailed_trials[mode]:
            trial_num = trial_data["trial"]
            p1_payoff = trial_data.get("player1_payoff", 0)
            p2_payoff = trial_data.get("player2_payoff", 0)
            total_payoff = trial_data.get("total_payoff", p1_payoff + p2_payoff)
            p1_coop = trial_data.get("player1_coop_rate", 0)
            p2_coop = trial_data.get("player2_coop_rate", 0)

            lines.append("")
            lines.append(f">>> Trial {trial_num}")
            lines.append(f"    社会总收益: {total_payoff:.1f} | P1得分: {p1_payoff:.1f} | P2得分: {p2_payoff:.1f}")
            lines.append(f"    P1合作率: {p1_coop:.1%} | P2合作率: {p2_coop:.1%}")

            if "player1_promise_keeping" in trial_data:
                lines.append(f"    P1承诺遵守率: {trial_data['player1_promise_keeping']:.1%}")
            if "player2_promise_keeping" in trial_data:
                lines.append(f"    P2承诺遵守率: {trial_data['player2_promise_keeping']:.1%}")

            lines.append("")

            for rd in trial_data["rounds"]:
                round_num = rd["round"]
                p1_msg = rd.get("player1_message", "")
                p2_msg = rd.get("player2_message", "")
                p1_action = rd.get("player1_action", "")
                p2_action = rd.get("player2_action", "")
                p1_payoff_rd = rd.get("player1_payoff", 0)
                p2_payoff_rd = rd.get("player2_payoff", 0)

                lines.append(f"  Round {round_num:2d}:")

                # 显示双方消息
                if p1_msg:
                    lines.append(f"    P1 says: \"{p1_msg}\"")
                if p2_msg:
                    lines.append(f"    P2 says: \"{p2_msg}\"")

                # 动作符号
                p1_symbol = "合作" if p1_action == "COOPERATE" else "背叛"
                p2_symbol = "合作" if p2_action == "COOPERATE" else "背叛"

                lines.append(f"    P1: {p1_symbol} | P2: {p2_symbol}")
                lines.append(f"    得分: P1={p1_payoff_rd}, P2={p2_payoff_rd}")
                lines.append("")

        lines.append("")

    lines.append("=" * 70)
    lines.append("记录结束")
    lines.append("=" * 70)

    return "\n".join(lines)


def _analyze_promise_keeping(messages: List[str], actions: List[Action]) -> float:
    """分析承诺遵守率"""
    if not messages or not actions:
        return 0.0

    kept_count = 0
    promise_count = 0

    cooperation_keywords = ["合作", "cooperate", "trust", "信任", "一起"]

    for msg, action in zip(messages, actions):
        if msg and any(kw in msg.lower() for kw in cooperation_keywords):
            promise_count += 1
            if action == Action.COOPERATE:
                kept_count += 1

    return kept_count / promise_count if promise_count > 0 else 1.0


def _print_cheap_talk_summary(results: Dict):
    """打印 Cheap Talk 汇总 (LLM vs LLM)"""
    print_separator("汇总: Cheap Talk (LLM vs LLM)")

    for game_name, stats in results.items():
        cn_name = GAME_NAMES_CN.get(game_name, game_name)
        print(f"\n{cn_name}:")

        no_talk = stats["no_talk"]
        cheap_talk = stats["cheap_talk"]

        # 计算平均合作率
        no_talk_avg_coop = (no_talk['player1_coop_rate']['mean'] + no_talk['player2_coop_rate']['mean']) / 2
        cheap_talk_avg_coop = (cheap_talk['player1_coop_rate']['mean'] + cheap_talk['player2_coop_rate']['mean']) / 2

        print(f"  No talk:    Total Payoff {no_talk['total_payoff']['mean']:.1f} ± {no_talk['total_payoff']['std']:.1f}, "
              f"Avg Coop {no_talk_avg_coop:.1%}")
        print(f"  Cheap talk: Total Payoff {cheap_talk['total_payoff']['mean']:.1f} ± {cheap_talk['total_payoff']['std']:.1f}, "
              f"Avg Coop {cheap_talk_avg_coop:.1%}")

        if cheap_talk.get("player1_promise_kept"):
            print(f"  P1 Promise kept: {cheap_talk['player1_promise_kept']['mean']:.1%}")
        if cheap_talk.get("player2_promise_kept"):
            print(f"  P2 Promise kept: {cheap_talk['player2_promise_kept']['mean']:.1%}")


# ============================================================
# 实验5: 群体动力学
# ============================================================

def experiment_group_dynamics(
        result_manager: ResultManager,
        n_agents: int = 10,
        provider: str = DEFAULT_CONFIG["provider"],
        n_repeats: int = DEFAULT_CONFIG["n_repeats"],  # <--- [新增] 重复次数参数
        rounds: int = DEFAULT_CONFIG["rounds"],
        games: List[str] = None,
        networks: List[str] = None,
) -> Dict:
    """
    群体动力学实验（单 Provider）

    修复版：支持动态 n_agents，支持 n_repeats 重复实验取平均
    """

    if games is None:
        games = list(GAME_REGISTRY.keys())
    if networks is None:
        networks = ["fully_connected", "small_world"]

    print_separator("实验5: 群体动力学 (单 Provider)")
    print(f"Agent数量: {n_agents} | Provider: {provider}")
    print(f"网络: {networks} | Repeats: {n_repeats} | Rounds: {rounds}")

    all_results = {}

    for game_name in games:
        game_config = GAME_REGISTRY[game_name]
        print_game_header(game_name)

        network_results = {}
        all_round_records = []  # 收集所有轮次记录

        for network_name in networks:
            network_cn = NETWORK_NAMES_CN.get(network_name, network_name)
            print(f"\n  网络: {network_cn}")

            try:
                # 用于存储所有重复实验的数据
                all_trials_payoffs = defaultdict(list)
                all_trials_coop_rates = defaultdict(list)

                # === 核心循环：执行 n_repeats 次 ===
                for i in range(n_repeats):
                    print(f"    Repeat {i + 1}/{n_repeats}...", end=" ", flush=True)

                    # 1. 动态生成策略列表 (每次循环重新生成，确保状态重置)
                    strategies = []

                    # 设定 LLM 数量 (至少2个，或占20%)
                    n_llm = max(2, int(n_agents * 0.2))
                    n_classic = n_agents - n_llm

                    # 创建 LLM Agents
                    for k in range(n_llm):
                        strategies.append((
                            f"LLM_{k + 1}",
                            LLMStrategy(provider=provider, mode="hybrid", game_config=game_config)
                        ))

                    # 创建传统策略 Agents
                    classic_classes = [
                        TitForTat, AlwaysCooperate, AlwaysDefect,
                        Pavlov, GrimTrigger, RandomStrategy
                    ]
                    for k in range(n_classic):
                        StrategyClass = classic_classes[k % len(classic_classes)]
                        strategies.append((
                            f"{StrategyClass.__name__}_{k + 1}",
                            StrategyClass()
                        ))

                    # 2. 运行仿真
                    agent_names = [name for name, _ in strategies]
                    NetworkClass = NETWORK_REGISTRY[network_name]
                    network = NetworkClass(agent_names)

                    agents = {}
                    for name, strategy in strategies:
                        agents[name] = AgentState(name=name, strategy=strategy)

                    sim = GameSimulation(
                        agents=agents,
                        network=network,
                        game_config=game_config,
                        rounds=rounds,
                        verbose=False
                    )

                    sim.run()

                    # 3. 收集单次数据
                    trial_payoffs = {}
                    trial_coop_rates = {}
                    llm_responses = {}  # 收集各LLM的思考过程
                    for aid, agent in agents.items():
                        all_trials_payoffs[aid].append(agent.total_payoff)
                        trial_payoffs[aid] = agent.total_payoff

                        history = agent.game_history
                        if history:
                            actions = [Action(h["my_action"]) for h in history]
                            rate = compute_cooperation_rate(actions)
                        else:
                            rate = 0.0
                        all_trials_coop_rates[aid].append(rate)
                        trial_coop_rates[aid] = rate

                        # 收集LLM响应
                        if hasattr(agent.strategy, 'raw_responses'):
                            llm_responses[aid] = agent.strategy.raw_responses.copy()

                    # 构建每轮记录
                    for aid, agent in agents.items():
                        if hasattr(agent.strategy, 'raw_responses') and agent.game_history:
                            responses = agent.strategy.raw_responses
                            for r_idx, hist in enumerate(agent.game_history):
                                llm_response = responses[r_idx] if r_idx < len(responses) else ""
                                all_round_records.append({
                                    "network": network_name,
                                    "trial": i + 1,
                                    "round": r_idx + 1,
                                    "agent": aid,
                                    "llm_response": llm_response,
                                    "my_action": hist.get("my_action", ""),
                                    "opponent": hist.get("opponent", ""),
                                    "opp_action": hist.get("opp_action", ""),
                                    "payoff": hist.get("payoff", 0),
                                })

                    print("Done")

                # 4. 计算平均值
                final_payoffs = {k: np.mean(v) for k, v in all_trials_payoffs.items()}
                coop_rates = {k: np.mean(v) for k, v in all_trials_coop_rates.items()}

                network_results[network_name] = {
                    "payoffs": final_payoffs,
                    "coop_rates": coop_rates,
                    "rankings": sorted(final_payoffs.items(), key=lambda x: x[1], reverse=True),
                }

                # 打印前 5 名
                print(f"    Avg ranking (Top 5):")
                for rank, (aid, payoff) in enumerate(network_results[network_name]["rankings"][:5], 1):
                    coop = coop_rates.get(aid, 0)
                    marker = "[LLM]" if aid.startswith("LLM") else "[Classic]"
                    print(f"      {marker} {rank}. {aid}: {payoff:.1f} (Coop: {coop:.1%})")

            except Exception as e:
                print(f"    Error: {e}")
                import traceback
                traceback.print_exc()
                network_results[network_name] = {"error": str(e)}

        all_results[game_name] = network_results
        result_manager.save_json(game_name, "group_dynamics", network_results)

        # 保存每轮记录
        result_manager.save_round_records("group_dynamics", game_name, provider, all_round_records)

        fig = _plot_group_rankings(network_results, game_name)
        if fig:
            result_manager.save_figure(game_name, "group_dynamics", fig)

    # 保存实验汇总
    result_manager.save_experiment_summary("group_dynamics", all_results)

    return all_results


def experiment_group_dynamics_multi_provider(
        result_manager: ResultManager,
        n_agents: int = 10,
        providers: List[str] = None,
        n_repeats: int = DEFAULT_CONFIG["n_repeats"],  # <--- [新增] 重复次数参数
        rounds: int = DEFAULT_CONFIG["rounds"],
        games: List[str] = None,
        networks: List[str] = None,
) -> Dict:
    """
    群体动力学实验（多 Provider 对比）

    修复版：支持动态 n_agents，支持 n_repeats 重复实验取平均
    """

    if providers is None:
        providers = ["deepseek", "openai", "gemini"]
    if games is None:
        games = list(GAME_REGISTRY.keys())
    if networks is None:
        networks = ["fully_connected", "small_world"]

    print_separator("实验5b: 多 Provider 群体动力学")
    print(f"Agent数量: {n_agents} | Providers: {providers}")
    print(f"网络: {networks} | Repeats: {n_repeats} | Rounds: {rounds}")

    all_results = {}

    for game_name in games:
        game_config = GAME_REGISTRY[game_name]
        print_game_header(game_name)

        network_results = {}
        all_round_records = []  # 收集所有轮次记录

        for network_name in networks:
            network_cn = NETWORK_NAMES_CN.get(network_name, network_name)
            print(f"\n  网络: {network_cn}")

            try:
                # 用于存储所有重复实验的数据
                all_trials_payoffs = defaultdict(list)
                all_trials_coop_rates = defaultdict(list)

                # === 核心循环：执行 n_repeats 次 ===
                for i in range(n_repeats):
                    print(f"    Repeat {i + 1}/{n_repeats}...", end=" ", flush=True)

                    # 1. 动态生成策略列表
                    strategies = []

                    # 设定 LLM 总数
                    min_llms = len(providers)
                    n_llm_total = max(min_llms, int(n_agents * 0.2))
                    n_classic = n_agents - n_llm_total

                    # 均匀分配 Provider
                    base_count = n_llm_total // len(providers)
                    remainder = n_llm_total % len(providers)
                    llm_counts = [base_count + 1 if k < remainder else base_count for k in range(len(providers))]

                    # 创建 LLM Agents
                    current_llm_idx = 1
                    for provider, count in zip(providers, llm_counts):
                        for _ in range(count):
                            strategies.append((
                                f"LLM_{provider}_{current_llm_idx}",
                                LLMStrategy(provider=provider, mode="hybrid", game_config=game_config)
                            ))
                            current_llm_idx += 1

                    # 创建传统策略 Agents
                    classic_classes = [
                        TitForTat, AlwaysCooperate, AlwaysDefect,
                        Pavlov, GrimTrigger, RandomStrategy
                    ]
                    for k in range(n_classic):
                        StrategyClass = classic_classes[k % len(classic_classes)]
                        strategies.append((
                            f"{StrategyClass.__name__}_{k + 1}",
                            StrategyClass()
                        ))

                    # 2. 运行仿真
                    agent_names = [name for name, _ in strategies]
                    NetworkClass = NETWORK_REGISTRY[network_name]
                    network = NetworkClass(agent_names)

                    agents = {}
                    for name, strategy in strategies:
                        agents[name] = AgentState(name=name, strategy=strategy)

                    sim = GameSimulation(
                        agents=agents,
                        network=network,
                        game_config=game_config,
                        rounds=rounds,
                        verbose=False
                    )

                    sim.run()

                    # 3. 收集单次数据
                    trial_payoffs = {}
                    trial_coop_rates = {}
                    llm_responses = {}  # 收集各LLM的思考过程
                    for aid, agent in agents.items():
                        all_trials_payoffs[aid].append(agent.total_payoff)
                        trial_payoffs[aid] = agent.total_payoff

                        history = agent.game_history
                        if history:
                            actions = [Action(h["my_action"]) for h in history]
                            rate = compute_cooperation_rate(actions)
                        else:
                            rate = 0.0
                        all_trials_coop_rates[aid].append(rate)
                        trial_coop_rates[aid] = rate

                        # 收集LLM响应
                        if hasattr(agent.strategy, 'raw_responses'):
                            llm_responses[aid] = agent.strategy.raw_responses.copy()

                    # 构建每轮记录
                    for aid, agent in agents.items():
                        if hasattr(agent.strategy, 'raw_responses') and agent.game_history:
                            responses = agent.strategy.raw_responses
                            for r_idx, hist in enumerate(agent.game_history):
                                llm_response = responses[r_idx] if r_idx < len(responses) else ""
                                all_round_records.append({
                                    "network": network_name,
                                    "trial": i + 1,
                                    "round": r_idx + 1,
                                    "agent": aid,
                                    "llm_response": llm_response,
                                    "my_action": hist.get("my_action", ""),
                                    "opponent": hist.get("opponent", ""),
                                    "opp_action": hist.get("opp_action", ""),
                                    "payoff": hist.get("payoff", 0),
                                })

                    # 保存详细数据（包含LLM思考过程）
                    detail_data = {
                        "experiment": "group_dynamics_multi",
                        "game": game_name,
                        "network": network_name,
                        "providers": providers,
                        "trial": i + 1,
                        "rounds": rounds,
                        "n_agents": n_agents,
                        "payoffs": trial_payoffs,
                        "coop_rates": trial_coop_rates,
                        "llm_responses": llm_responses,  # 各LLM智能体的思考过程
                    }
                    result_manager.save_detail(f"group_multi_{game_name}_{network_name}", "multi", i + 1, rounds, detail_data)

                    print("Done")

                # 4. 计算平均值
                final_payoffs = {k: np.mean(v) for k, v in all_trials_payoffs.items()}
                coop_rates = {k: np.mean(v) for k, v in all_trials_coop_rates.items()}

                # 分类统计
                llm_results = {k: v for k, v in final_payoffs.items() if k.startswith("LLM_")}
                traditional_results = {k: v for k, v in final_payoffs.items() if not k.startswith("LLM_")}

                network_results[network_name] = {
                    "payoffs": final_payoffs,
                    "coop_rates": coop_rates,
                    "rankings": sorted(final_payoffs.items(), key=lambda x: x[1], reverse=True),
                    "llm_comparison": llm_results,
                    "traditional_comparison": traditional_results,
                }

                print(f"    LLM Avg ranking (Top 5):")
                llm_ranked = sorted(llm_results.items(), key=lambda x: x[1], reverse=True)
                for rank, (aid, payoff) in enumerate(llm_ranked[:5], 1):
                    coop = coop_rates.get(aid, 0)
                    print(f"      {rank}. {aid}: {payoff:.1f} (Coop: {coop:.1%})")

            except Exception as e:
                print(f"    Error: {e}")
                import traceback
                traceback.print_exc()
                network_results[network_name] = {"error": str(e)}

        all_results[game_name] = network_results
        result_manager.save_json(game_name, "group_dynamics_multi_provider", network_results)

        # 保存每轮记录
        result_manager.save_round_records("group_dynamics_multi", game_name, "multi", all_round_records)

        fig = _plot_multi_provider_comparison(network_results, game_name, providers)
        if fig:
            result_manager.save_figure(game_name, "group_dynamics_multi_provider", fig)

    # 保存实验汇总
    result_manager.save_experiment_summary("group_dynamics_multi_provider", all_results)

    return all_results


def _plot_multi_provider_comparison(network_results: Dict, game_name: str, providers: List[str]) -> Optional[plt.Figure]:
    """绘制多 Provider 对比图"""

    valid_networks = [n for n in network_results if "error" not in network_results[n]]
    if not valid_networks:
        return None

    n_networks = len(valid_networks)
    fig, axes = plt.subplots(1, n_networks, figsize=(7 * n_networks, 6))
    if n_networks == 1:
        axes = [axes]

    # 为不同 provider 设置颜色
    provider_colors = {
        "deepseek": "#4CAF50",  # 绿色
        "openai": "#2196F3",    # 蓝色
        "gemini": "#FF9800",    # 橙色
    }

    for ax, network_name in zip(axes, valid_networks):
        data = network_results[network_name]
        rankings = data["rankings"]
        coop_rates = data["coop_rates"]

        names = [r[0] for r in rankings]
        payoffs = [r[1] for r in rankings]

        # 设置颜色
        colors = []
        for name in names:
            if name.startswith("LLM_"):
                provider = name.replace("LLM_", "")
                colors.append(provider_colors.get(provider, "#9C27B0"))
            else:
                colors.append("#757575")  # 灰色表示传统策略

        bars = ax.barh(range(len(names)), payoffs, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("总得分")
        ax.set_title(f"{NETWORK_NAMES_CN.get(network_name, network_name)}")
        ax.invert_yaxis()

        # 在柱子上显示合作率
        for i, (name, payoff) in enumerate(zip(names, payoffs)):
            coop = coop_rates.get(name, 0)
            ax.text(payoff + 0.5, i, f"{coop:.0%}", va='center', fontsize=8)

    # 添加图例
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, facecolor=provider_colors.get(p, "#9C27B0"), label=f"LLM_{p}")
        for p in providers
    ]
    legend_elements.append(plt.Rectangle((0,0), 1, 1, facecolor="#757575", label="传统策略"))
    fig.legend(handles=legend_elements, loc='upper center', ncol=len(providers)+1, bbox_to_anchor=(0.5, 1.02))

    game_cn = GAME_NAMES_CN.get(game_name, game_name)
    fig.suptitle(f"多 Provider 群体动力学 - {game_cn}", fontsize=14, y=1.08)

    plt.tight_layout()
    return fig


def _plot_group_rankings(network_results: Dict, game_name: str) -> Optional[plt.Figure]:
    """绘制群体动力学排名图"""

    valid_networks = [n for n in network_results if "error" not in network_results[n]]
    if not valid_networks:
        return None

    n_networks = len(valid_networks)
    fig, axes = plt.subplots(1, n_networks, figsize=(6 * n_networks, 5))
    if n_networks == 1:
        axes = [axes]

    for ax, network_name in zip(axes, valid_networks):
        data = network_results[network_name]
        rankings = data["rankings"]

        names = [r[0] for r in rankings]
        payoffs = [r[1] for r in rankings]
        colors = ['steelblue' if 'LLM' in n else 'gray' for n in names]

        ax.barh(range(len(names)), payoffs, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("总得分")
        ax.set_title(f"{NETWORK_NAMES_CN.get(network_name, network_name)}")
        ax.invert_yaxis()

    fig.suptitle(f"群体动力学 - {GAME_NAMES_CN.get(game_name, game_name)}", fontsize=14)
    plt.tight_layout()
    return fig


# ============================================================
# 实验6: Baseline 对比
# ============================================================

def experiment_baseline_comparison(
    result_manager: ResultManager,
    providers: List[str] = ["deepseek", "openai", "gemini"],
    n_repeats: int = DEFAULT_CONFIG["n_repeats"],
    rounds: int = DEFAULT_CONFIG["rounds"],
    games: List[str] = None,
) -> Dict:
    """Baseline 对比实验（多 Provider 版本）"""

    if games is None:
        games = list(GAME_REGISTRY.keys())

    baselines = {
        "TitForTat": TitForTat,
        "AlwaysCooperate": AlwaysCooperate,
        "AlwaysDefect": AlwaysDefect,
        "GrimTrigger": GrimTrigger,
        "Pavlov": Pavlov,
        "Random": RandomStrategy,
    }

    print_separator("实验6: Baseline 对比")
    print(f"LLM Providers: {providers}")
    print(f"LLM vs 经典策略: {list(baselines.keys())}")
    print(f"Repeats: {n_repeats} | Rounds: {rounds}")

    all_results = {}

    for game_name in games:
        game_config = GAME_REGISTRY[game_name]
        print_game_header(game_name)

        game_results = {}
        all_round_records = []  # 收集所有轮次记录

        for provider in providers:
            print(f"\n  Provider: {provider.upper()}")

            baseline_results = {}

            for baseline_name, BaselineClass in baselines.items():
                print(f"\n    vs {baseline_name}")

                payoffs = []
                coop_rates = []

                for trial in range(n_repeats):
                    print(f"      Trial {trial + 1}/{n_repeats}...", end=" ", flush=True)

                    try:
                        llm_strategy = LLMStrategy(
                            provider=provider,
                            mode="hybrid",
                            game_config=game_config,
                        )

                        opponent = BaselineClass()

                        llm_payoff = 0
                        llm_history = []
                        opp_history = []

                        for r in range(rounds):
                            llm_action = llm_strategy.choose_action(llm_history, opp_history)
                            opp_action = opponent.choose_action(make_history_tuples(opp_history, llm_history))

                            payoff, _ = get_payoff(game_config, llm_action, opp_action)
                            llm_payoff += payoff

                            # 记录每轮数据
                            llm_response = llm_strategy.raw_responses[-1] if llm_strategy.raw_responses else ""
                            all_round_records.append({
                                "provider": provider,
                                "baseline": baseline_name,
                                "trial": trial + 1,
                                "round": r + 1,
                                "llm_response": llm_response,
                                "llm_action": llm_action.name,
                                "opp_action": opp_action.name,
                                "payoff": payoff,
                                "cumulative_payoff": llm_payoff,
                            })

                            llm_history.append(llm_action)
                            opp_history.append(opp_action)

                        coop_rate = compute_cooperation_rate(llm_history)
                        payoffs.append(llm_payoff)
                        coop_rates.append(coop_rate)

                        # 保存详细数据（包含LLM思考过程）
                        detail_data = {
                            "experiment": "baseline",
                            "game": game_name,
                            "provider": provider,
                            "baseline": baseline_name,
                            "trial": trial + 1,
                            "rounds": rounds,
                            "payoff": llm_payoff,
                            "coop_rate": coop_rate,
                            "llm_history": [a.name for a in llm_history],
                            "opp_history": [a.name for a in opp_history],
                            "llm_responses": llm_strategy.raw_responses.copy(),  # LLM思考过程
                        }
                        result_manager.save_detail(f"baseline_{game_name}_{baseline_name}", provider, trial + 1, rounds, detail_data)

                        print(f"Payoff: {llm_payoff:.1f}, Coop rate: {coop_rate:.1%}")

                    except Exception as e:
                        print(f"Error: {e}")
                        continue

                baseline_results[baseline_name] = {
                    "payoff": compute_statistics(payoffs),
                    "coop_rate": compute_statistics(coop_rates),
                }

            game_results[provider] = baseline_results

        all_results[game_name] = game_results

        # 保存结果
        result_manager.save_json(game_name, "baseline", game_results)

        # 保存每轮记录
        result_manager.save_round_records("baseline", game_name, "all", all_round_records)

        # 生成图表
        fig = _plot_baseline_multi_provider(game_results, game_name, providers, baselines)
        if fig:
            result_manager.save_figure(game_name, "baseline", fig)

    _print_baseline_summary_multi_provider(all_results, providers)

    # 保存实验汇总
    result_manager.save_experiment_summary("baseline", all_results)

    return all_results


def _plot_baseline_multi_provider(
    game_results: Dict,
    game_name: str,
    providers: List[str],
    baselines: Dict
) -> Optional[plt.Figure]:
    """绘制多 Provider Baseline 对比图"""

    n_providers = len(providers)
    n_baselines = len(baselines)

    fig, axes = plt.subplots(1, n_providers, figsize=(6 * n_providers, 6))
    if n_providers == 1:
        axes = [axes]

    # 为不同 provider 设置颜色
    provider_colors = {
        "deepseek": "#4CAF50",
        "openai": "#2196F3",
        "gemini": "#FF9800",
    }

    baseline_names = list(baselines.keys())

    for ax, provider in zip(axes, providers):
        if provider not in game_results:
            continue

        baseline_data = game_results[provider]
        means = [baseline_data[b]["payoff"]["mean"] for b in baseline_names]
        stds = [baseline_data[b]["payoff"]["std"] for b in baseline_names]

        x = np.arange(len(baseline_names))
        color = provider_colors.get(provider, "#9C27B0")
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=color, alpha=0.8)

        ax.set_ylabel("LLM 得分")
        ax.set_title(f"{provider.upper()}")
        ax.set_xticks(x)
        ax.set_xticklabels(baseline_names, rotation=45, ha='right')

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{mean:.1f}', ha='center', va='bottom', fontsize=9)

    game_cn = GAME_NAMES_CN.get(game_name, game_name)
    fig.suptitle(f"LLM vs Baselines (多模型对比) - {game_cn}", fontsize=14)
    plt.tight_layout()
    return fig


def _print_baseline_summary_multi_provider(results: Dict, providers: List[str]):
    """打印多 Provider Baseline 对比汇总"""
    print_separator("汇总: LLM vs Baselines (多模型)")

    for game_name, provider_stats in results.items():
        cn_name = GAME_NAMES_CN.get(game_name, game_name)
        print(f"\n{cn_name}:")

        for provider in providers:
            if provider not in provider_stats:
                continue

            print(f"\n  {provider.upper()}:")
            print(f"    {'Opponent':<16} {'Payoff':<18} {'Coop Rate':<12}")
            print(f"    {'-' * 46}")

            baseline_data = provider_stats[provider]
            for baseline, stats in baseline_data.items():
                pay = stats["payoff"]
                coop = stats["coop_rate"]
                pay_str = f"{pay['mean']:.1f} ± {pay['std']:.1f}"
                coop_str = f"{coop['mean']:.1%}"
                print(f"    {baseline:<16} {pay_str:<18} {coop_str:<12}")


# ============================================================
# 主函数
# ============================================================

def print_usage():
    """打印使用说明"""
    print("""
博弈论 LLM 研究实验脚本 v9
==========================

用法:
  python research.py <experiment> [options]

实验列表:
  pure_hybrid   - 实验1: Pure vs Hybrid LLM
  window        - 实验2: 记忆视窗对比
  multi_llm     - 实验3: 多 LLM 对比
  cheap_talk    - 实验4: Cheap Talk 语言交流
  group         - 实验5: 群体动力学（DeepSeek/OpenAI/Gemini 三模型）
  group_single  - 实验5: 群体动力学（单 Provider，需指定 --provider）
  baseline      - 实验6: Baseline 对比（DeepSeek/OpenAI/Gemini 三模型）
  all           - 运行全部实验

选项:
  --provider    LLM 提供商 (deepseek/openai/gemini)    [默认: deepseek]
  --provider1   cheap_talk 实验的 Player1 模型         [默认: 同 --provider]
  --provider2   cheap_talk 实验的 Player2 模型         [默认: 同 --provider]
  --repeats     重复次数                               [默认: 3]
  --rounds      每次轮数                               [默认: 20]
  --games       指定博弈 (pd/snowdrift/stag_hunt/harmony/all) [默认: all]
  --n_agents    群体动力学实验的智能体数量             [默认: 10]

帮助:
  -h, --help    显示此帮助信息

结果目录结构:
  results/{时间戳}/
  ├── experiment_config.json
  ├── summary.json
  ├── details/                    # 每次实验详细数据
  │   ├── {实验名}_{模型名}_{次数}_{轮数}.json
  │   └── {实验名}_{博弈}_{模型}_rounds.json  # 每轮记录
  ├── summary/                    # 各实验汇总 (CSV 格式)
  │   └── {实验名}.csv
  ├── prisoners_dilemma/
  │   ├── pure_vs_hybrid.json
  │   └── pure_vs_hybrid.png
  ├── snowdrift/
  └── stag_hunt/

示例:
  python research.py pure_hybrid
  python research.py group --rounds 30 --n_agents 15
  python research.py all --provider openai --repeats 5
  python research.py baseline --games pd
  python research.py cheap_talk --provider1 openai --provider2 gemini
""")


def main():
    # 默认跑全部实验

    if len(sys.argv) < 2:
        experiment = "all"
        print("未指定实验，默认运行全部实验...")
    else:
        experiment = sys.argv[1].lower()

        # 如果是帮助命令
        if experiment in ["-h", "--help", "help"]:
            print_usage()
            return

    # 解析参数
    provider = DEFAULT_CONFIG["provider"]
    n_repeats = DEFAULT_CONFIG["n_repeats"]
    rounds = DEFAULT_CONFIG["rounds"]
    games = None
    n_agents = 10
    provider1 = None  # cheap_talk 专用
    provider2 = None  # cheap_talk 专用

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--provider" and i + 1 < len(sys.argv):
            provider = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--provider1" and i + 1 < len(sys.argv):
            provider1 = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--provider2" and i + 1 < len(sys.argv):
            provider2 = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--repeats" and i + 1 < len(sys.argv):
            n_repeats = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--rounds" and i + 1 < len(sys.argv):
            rounds = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--n_agents" and i + 1 < len(sys.argv):
            n_agents = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--games" and i + 1 < len(sys.argv):
            game_arg = sys.argv[i + 1].lower()
            if game_arg == "all":
                games = None
            elif game_arg == "pd":
                games = ["prisoners_dilemma"]
            elif game_arg == "snowdrift":
                games = ["snowdrift"]
            elif game_arg == "stag_hunt":
                games = ["stag_hunt"]
            else:
                games = [game_arg]
            i += 2
        else:
            i += 1

    # 创建结果管理器
    result_manager = ResultManager()

    # 保存实验配置
    config = {
        "experiment": experiment,
        "provider": provider,
        "n_repeats": n_repeats,
        "rounds": rounds,
        "games": games or list(GAME_REGISTRY.keys()),
        "timestamp": result_manager.timestamp,
    }
    result_manager.save_config(config)

    # 运行实验
    all_results = {}

    if experiment in ["pure_hybrid", "all"]:
        results = experiment_pure_vs_hybrid(
            result_manager, provider=provider, n_repeats=n_repeats, rounds=rounds, games=games
        )
        all_results["pure_vs_hybrid"] = results

    if experiment in ["window", "all"]:
        results = experiment_memory_window(
            result_manager, provider=provider, n_repeats=n_repeats, rounds=max(30, rounds), games=games
        )
        all_results["memory_window"] = results

    if experiment in ["multi_llm", "all"]:
        results = experiment_multi_llm(
            result_manager, n_repeats=n_repeats, rounds=rounds, games=games
        )
        all_results["multi_llm"] = results

    if experiment in ["cheap_talk", "all"]:
        # cheap_talk 支持 --provider1 和 --provider2 指定两个不同的 LLM
        p1 = provider1 if provider1 else provider
        p2 = provider2 if provider2 else provider
        results = experiment_cheap_talk(
            result_manager, provider1=p1, provider2=p2, n_repeats=n_repeats, rounds=rounds, games=games
        )
        all_results["cheap_talk"] = results

    if experiment in ["group", "group_multi", "all"]:
        # 群体动力学实验默认使用三模型
        results = experiment_group_dynamics_multi_provider(
            result_manager,
            n_agents=n_agents,
            n_repeats=n_repeats,
            providers=["deepseek", "openai", "gemini"],
            rounds=rounds,
            games=games
        )
        all_results["group_dynamics_multi_provider"] = results

    if experiment in ["group_single"]:
        # 单 Provider 群体动力学实验
        results = experiment_group_dynamics(
            result_manager,
            n_agents=n_agents,
            n_repeats=n_repeats,
            provider=provider,
            rounds=rounds,
            games=games
        )
        all_results["group_dynamics"] = results

    if experiment in ["baseline", "all"]:
        results = experiment_baseline_comparison(
            result_manager,
            providers=["deepseek", "openai", "gemini"],
            n_repeats=n_repeats,
            rounds=rounds,
            games=games
        )
        all_results["baseline"] = results

    if experiment not in ["pure_hybrid", "window", "multi_llm", "cheap_talk", "group", "group_multi", "group_single", "baseline", "all"]:
        print(f"未知实验: {experiment}")
        print_usage()
        return

    # 保存汇总
    result_manager.save_summary(all_results)

    print_separator("实验完成")
    print(f"Results dir: {result_manager.root_dir}")
    print(f"Total experiments: {len(all_results)}")


if __name__ == "__main__":
    main()
