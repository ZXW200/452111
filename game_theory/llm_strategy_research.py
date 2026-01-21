"""
LLM 博弈策略 (科研版)
LLM Game Theory Strategy (Research Version)

修复科研硬伤：
1. 记忆视窗可配置（不再硬编码5轮）
2. Pure LLM 模式（不作弊式分析）
3. Cheap Talk 语言通道
4. 支持多次重复实验
"""

from typing import List, Tuple, Dict, Optional
from .games import Action, GameConfig, PRISONERS_DILEMMA


class LLMStrategy:
    """科研级 LLM 博弈策略"""

    name = "LLM Dynamic"
    description = "Research-grade LLM strategy"
    description_cn = "科研级LLM策略"

    # ============================================================
    # Prompt 模板
    # ============================================================

    # Pure LLM 模式：只给原始历史，不给分析
    PROMPT_PURE = """You are {agent_name} playing a repeated {game_name} game.
{personality_prompt}

PAYOFF MATRIX:
- Both Cooperate (C,C): {cc}, {cc}
- You Cooperate, Opponent Defects (C,D): {cd}, {dc}
- You Defect, Opponent Cooperates (D,C): {dc}, {cd}
- Both Defect (D,D): {dd}, {dd}

OPPONENT: {opponent_name}

COMPLETE INTERACTION HISTORY:
{history_text}

YOUR STATUS:
- Current round: {current_round}
- Your total payoff: {total_payoff}

TASK: Analyze the opponent's strategy pattern yourself, then decide your action.

Format:
ANALYSIS: [Your analysis of opponent's behavior pattern]
REASONING: [Your strategic reasoning]
ACTION: [COOPERATE or DEFECT]"""

    # Hybrid 模式：给分析辅助（对比用）
    PROMPT_HYBRID = """You are {agent_name} playing a repeated {game_name} game.
{personality_prompt}

PAYOFF MATRIX:
- Both Cooperate (C,C): {cc}, {cc}
- You Cooperate, Opponent Defects (C,D): {cd}, {dc}
- You Defect, Opponent Cooperates (D,C): {dc}, {cd}
- Both Defect (D,D): {dd}, {dd}

OPPONENT: {opponent_name}
- Cooperation rate: {opp_coop_rate}%
- Detected pattern: {opp_pattern}

RECENT HISTORY (last {history_len} rounds):
{history_text}

YOUR STATUS:
- Current round: {current_round}
- Your total payoff: {total_payoff}
- Your cooperation rate: {my_coop_rate}%

TASK: Decide your action.

Format:
REASONING: [Your strategic reasoning]
ACTION: [COOPERATE or DEFECT]"""

    # Cheap Talk 模式：先交流再决策
    PROMPT_CHEAP_TALK = """You are {agent_name} playing a repeated {game_name} game with communication.
{personality_prompt}

PAYOFF MATRIX:
- Both Cooperate (C,C): {cc}, {cc}
- You Cooperate, Opponent Defects (C,D): {cd}, {dc}
- You Defect, Opponent Cooperates (D,C): {dc}, {cd}
- Both Defect (D,D): {dd}, {dd}

OPPONENT: {opponent_name}

HISTORY:
{history_text}

OPPONENT'S MESSAGE THIS ROUND: "{opponent_message}"

YOUR STATUS:
- Current round: {current_round}
- Your total payoff: {total_payoff}

TASK: 
1. Send a message to your opponent (can be honest or deceptive)
2. Decide your actual action (which may or may not match your message)

Format:
MESSAGE: [Your message to opponent, max 20 words]
REASONING: [Your private reasoning]
ACTION: [COOPERATE or DEFECT]"""

    def __init__(self,
                 agent_name: str = "LLM_Agent",
                 game_config: GameConfig = None,
                 provider: str = None,
                 personality: str = "neutral",
                 # 科研参数
                 mode: str = "pure",  # pure / hybrid / cheap_talk
                 history_window: int = None,  # None = 全部历史
                 temperature: float = 0.3):

        self.agent_name = agent_name
        self.game_config = game_config or PRISONERS_DILEMMA
        self.provider = provider
        self.personality = personality

        # 科研参数
        self.mode = mode
        self.history_window = history_window  # None 表示完整历史
        self.temperature = temperature

        # 状态
        self.total_payoff = 0
        self.decision_log: List[Dict] = []

        # Cheap Talk
        self.last_message: str = ""
        self.received_messages: List[str] = []

        # LLM
        self.llm = None
        try:
            from .llm_api import LLMClient
            self.llm = LLMClient(provider=provider)
        except Exception as e:
            print(f"LLM 初始化失败: {e}")

    def _get_personality_prompt(self) -> str:
        prompts = {
            "neutral": "You are rational and adaptive.",
            "cooperative": "You value long-term cooperation and trust.",
            "competitive": "You focus on maximizing your own payoff.",
            "cautious": "You are risk-averse and protective.",
            "deceptive": "You may use deception strategically.",
        }
        return prompts.get(self.personality, prompts["neutral"])

    def choose_action(self,
                      history: List[Tuple[Action, Action]],
                      opponent_name: str = "Opponent",
                      opponent_message: str = None) -> Action:
        """选择动作"""
        if self.llm is None:
            return self._fallback(history)

        try:
            if self.mode == "cheap_talk":
                prompt = self._build_cheap_talk_prompt(history, opponent_name, opponent_message)
            elif self.mode == "hybrid":
                prompt = self._build_hybrid_prompt(history, opponent_name)
            else:  # pure
                prompt = self._build_pure_prompt(history, opponent_name)

            response = self.llm.chat(prompt, temperature=self.temperature, max_tokens=300)
            action, analysis, reasoning, message = self._parse_response(response)

            # 记录日志
            self.decision_log.append({
                "round": len(history) + 1,
                "opponent": opponent_name,
                "mode": self.mode,
                "analysis": analysis,
                "reasoning": reasoning,
                "message": message,
                "action": action.value,
                "total_payoff": self.total_payoff,
                "opponent_message": opponent_message,
                "raw_response": response,
            })

            self.last_message = message

            return action

        except Exception as e:
            print(f"LLM Error: {e}")
            return self._fallback(history)

    def get_message(self) -> str:
        """获取上一次的 Cheap Talk 消息"""
        return self.last_message

    def update_payoff(self, payoff: float):
        self.total_payoff += payoff

    def _fallback(self, history: List[Tuple[Action, Action]]) -> Action:
        if not history:
            return Action.COOPERATE
        return history[-1][1]

    # ============================================================
    # Prompt 构建
    # ============================================================

    def _build_pure_prompt(self, history: List[Tuple[Action, Action]], opponent_name: str) -> str:
        """Pure 模式：只给原始历史"""
        pm = self.game_config.payoff_matrix

        # 历史文本（可配置窗口）
        if self.history_window and len(history) > self.history_window:
            show_history = history[-self.history_window:]
            prefix = f"[Showing last {self.history_window} of {len(history)} rounds]\n"
        else:
            show_history = history
            prefix = ""

        history_text = prefix + self._format_history_raw(show_history, len(history) - len(show_history))

        return self.PROMPT_PURE.format(
            agent_name=self.agent_name,
            game_name=self.game_config.name,
            personality_prompt=self._get_personality_prompt(),
            cc=pm[(Action.COOPERATE, Action.COOPERATE)][0],
            cd=pm[(Action.COOPERATE, Action.DEFECT)][0],
            dc=pm[(Action.DEFECT, Action.COOPERATE)][0],
            dd=pm[(Action.DEFECT, Action.DEFECT)][0],
            opponent_name=opponent_name,
            history_text=history_text,
            current_round=len(history) + 1,
            total_payoff=self.total_payoff,
        )

    def _build_hybrid_prompt(self, history: List[Tuple[Action, Action]], opponent_name: str) -> str:
        """Hybrid 模式：给分析辅助"""
        pm = self.game_config.payoff_matrix
        n = len(history) if history else 1

        # 统计
        my_coop = sum(1 for h in history if h[0] == Action.COOPERATE) if history else 0
        opp_coop = sum(1 for h in history if h[1] == Action.COOPERATE) if history else 0

        # 历史
        window = self.history_window or 10
        show_history = history[-window:] if history else []
        history_text = self._format_history_raw(show_history, max(0, len(history) - window))

        return self.PROMPT_HYBRID.format(
            agent_name=self.agent_name,
            game_name=self.game_config.name,
            personality_prompt=self._get_personality_prompt(),
            cc=pm[(Action.COOPERATE, Action.COOPERATE)][0],
            cd=pm[(Action.COOPERATE, Action.DEFECT)][0],
            dc=pm[(Action.DEFECT, Action.COOPERATE)][0],
            dd=pm[(Action.DEFECT, Action.DEFECT)][0],
            opponent_name=opponent_name,
            opp_coop_rate=round(100 * opp_coop / n, 1),
            opp_pattern=self._analyze_opponent(history),
            history_len=len(show_history),
            history_text=history_text,
            current_round=len(history) + 1,
            total_payoff=self.total_payoff,
            my_coop_rate=round(100 * my_coop / n, 1),
        )

    def _build_cheap_talk_prompt(self, history: List[Tuple[Action, Action]],
                                 opponent_name: str, opponent_message: str) -> str:
        """Cheap Talk 模式"""
        pm = self.game_config.payoff_matrix

        history_text = self._format_history_with_messages(history)

        return self.PROMPT_CHEAP_TALK.format(
            agent_name=self.agent_name,
            game_name=self.game_config.name,
            personality_prompt=self._get_personality_prompt(),
            cc=pm[(Action.COOPERATE, Action.COOPERATE)][0],
            cd=pm[(Action.COOPERATE, Action.DEFECT)][0],
            dc=pm[(Action.DEFECT, Action.COOPERATE)][0],
            dd=pm[(Action.DEFECT, Action.DEFECT)][0],
            opponent_name=opponent_name,
            history_text=history_text,
            opponent_message=opponent_message or "(no message)",
            current_round=len(history) + 1,
            total_payoff=self.total_payoff,
        )

    def _format_history_raw(self, history: List[Tuple[Action, Action]], offset: int = 0) -> str:
        """格式化历史（原始数据，不加分析）"""
        if not history:
            return "No previous interactions."

        lines = []
        for i, (my_act, opp_act) in enumerate(history):
            round_num = offset + i + 1
            my_str = "C" if my_act == Action.COOPERATE else "D"
            opp_str = "C" if opp_act == Action.COOPERATE else "D"
            lines.append(f"Round {round_num}: You={my_str}, Opponent={opp_str}")

        return "\n".join(lines)

    def _format_history_with_messages(self, history: List[Tuple[Action, Action]]) -> str:
        """格式化历史（包含消息）"""
        if not history:
            return "No previous interactions."

        lines = []
        for i, (my_act, opp_act) in enumerate(history):
            my_str = "C" if my_act == Action.COOPERATE else "D"
            opp_str = "C" if opp_act == Action.COOPERATE else "D"

            # 获取该轮的消息（如果有）
            msg_info = ""
            if i < len(self.decision_log):
                my_msg = self.decision_log[i].get("message", "")
                opp_msg = self.decision_log[i].get("opponent_message", "")
                if my_msg or opp_msg:
                    msg_info = f' [You said: "{my_msg}", They said: "{opp_msg}"]'

            lines.append(f"Round {i + 1}: You={my_str}, Opponent={opp_str}{msg_info}")

        return "\n".join(lines[-10:])  # 最近10轮

    def _analyze_opponent(self, history: List[Tuple[Action, Action]]) -> str:
        """分析对手（仅 Hybrid 模式用）"""
        if not history:
            return "Unknown"
        if len(history) < 3:
            return "Insufficient data"

        recent = history[-5:]
        opp_actions = [h[1] for h in recent]

        if all(a == Action.COOPERATE for a in opp_actions):
            return "Always Cooperate pattern"
        if all(a == Action.DEFECT for a in opp_actions):
            return "Always Defect pattern"

        # TFT 检测
        is_tft = all(history[i][1] == history[i - 1][0] for i in range(1, min(5, len(history))))
        if is_tft:
            return "Tit-for-Tat pattern"

        coop_rate = sum(1 for a in opp_actions if a == Action.COOPERATE) / len(opp_actions)
        return f"Mixed ({coop_rate:.0%} cooperation)"

    def _parse_response(self, response: str) -> Tuple[Action, str, str, str]:
        """解析响应，返回 (action, analysis, reasoning, message)"""
        response_text = response.strip()
        analysis = ""
        reasoning = ""
        message = ""

        # 提取 ANALYSIS
        if "ANALYSIS:" in response_text.upper():
            try:
                start = response_text.upper().find("ANALYSIS:") + len("ANALYSIS:")
                end = response_text.upper().find("REASONING:")
                if end == -1:
                    end = response_text.upper().find("ACTION:")
                if end > start:
                    analysis = response_text[start:end].strip()
            except:
                pass

        # 提取 REASONING
        if "REASONING:" in response_text.upper():
            try:
                start = response_text.upper().find("REASONING:") + len("REASONING:")
                end = response_text.upper().find("ACTION:")
                if end > start:
                    reasoning = response_text[start:end].strip()
            except:
                pass

        # 提取 MESSAGE (Cheap Talk)
        if "MESSAGE:" in response_text.upper():
            try:
                start = response_text.upper().find("MESSAGE:") + len("MESSAGE:")
                end = response_text.upper().find("REASONING:")
                if end == -1:
                    end = response_text.upper().find("ACTION:")
                if end > start:
                    message = response_text[start:end].strip().strip('"')
            except:
                pass

        # 提取 ACTION
        action = Action.COOPERATE
        if "ACTION:" in response_text.upper():
            after = response_text.upper().split("ACTION:")[-1].strip()
            if after.startswith("DEFECT") or "DEFECT" in after.split()[0]:
                action = Action.DEFECT
        elif "DEFECT" in response_text.upper() and "COOPERATE" not in response_text.upper():
            action = Action.DEFECT

        return action, analysis, reasoning, message

    def get_decision_log(self) -> List[Dict]:
        return self.decision_log

    def reset(self):
        self.total_payoff = 0
        self.decision_log = []
        self.last_message = ""
        self.received_messages = []


# ============================================================
# 预设性格
# ============================================================

class PureLLM(LLMStrategy):
    """Pure 模式 - 不给任何分析"""
    name = "LLM Pure"

    def __init__(self, **kwargs):
        super().__init__(mode="pure", **kwargs)


class HybridLLM(LLMStrategy):
    """Hybrid 模式 - 给分析辅助"""
    name = "LLM Hybrid"

    def __init__(self, **kwargs):
        super().__init__(mode="hybrid", **kwargs)


class CheapTalkLLM(LLMStrategy):
    """Cheap Talk 模式 - 可交流"""
    name = "LLM CheapTalk"

    def __init__(self, **kwargs):
        super().__init__(mode="cheap_talk", **kwargs)


class CooperativeLLM(LLMStrategy):
    name = "LLM Cooperative"

    def __init__(self, **kwargs):
        super().__init__(personality="cooperative", **kwargs)


class CompetitiveLLM(LLMStrategy):
    name = "LLM Competitive"

    def __init__(self, **kwargs):
        super().__init__(personality="competitive", **kwargs)


class DeceptiveLLM(LLMStrategy):
    """欺骗型 - 专门用于 Cheap Talk 研究"""
    name = "LLM Deceptive"

    def __init__(self, **kwargs):
        super().__init__(personality="deceptive", mode="cheap_talk", **kwargs)
