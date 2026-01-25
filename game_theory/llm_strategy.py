"""
LLM 博弈策略模块
Game Theory LLM Strategy Module

修复版 v2 - 解决格式霸权 + Token 截断问题
"""

import re
import random
import os
from enum import Enum
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field


# ============================================================
# 模板加载
# ============================================================

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "prompts")
_TEMPLATE_CACHE: Dict[str, str] = {}


def _load_template(name: str) -> str:
    """加载提示模板"""
    if name not in _TEMPLATE_CACHE:
        path = os.path.join(_TEMPLATE_DIR, f"{name}.txt")
        with open(path, "r", encoding="utf-8") as f:
            _TEMPLATE_CACHE[name] = f.read()
    return _TEMPLATE_CACHE[name]


# ============================================================
# 解析状态枚举
# ============================================================

class ParseStatus(Enum):
    """解析结果状态"""
    SUCCESS = "success"           # 高置信度匹配
    FALLBACK = "fallback"         # 低置信度匹配
    AMBIGUOUS = "ambiguous"       # 矛盾信号
    FAILED = "failed"             # 解析失败


# ============================================================
# 解析结果数据类
# ============================================================

@dataclass
class ParseResult:
    """LLM 响应解析结果"""
    action: Optional[Any] = None  # Action 枚举，延迟导入
    status: ParseStatus = ParseStatus.FAILED
    confidence: float = 0.0
    matched_pattern: str = ""
    raw_response: str = ""

    # 调试信息
    cooperate_signals: List[Tuple[str, float]] = field(default_factory=list)
    defect_signals: List[Tuple[str, float]] = field(default_factory=list)


# ============================================================
# 响应解析器
# ============================================================

class ResponseParser:
    """
    LLM 响应解析器

    特点:
    1. 支持20+种格式变体
    2. 置信度评分
    3. 矛盾检测
    4. 解析失败时随机选择（消除偏差）
    """

    # 合作信号 (按优先级排序)
    COOPERATE_PATTERNS = [
        # 精确格式
        (r"ACTION:\s*COOPERATE", 1.0),
        (r"ACTION:\s*C\b", 0.95),

        # 宽松格式 (大小写不敏感)
        (r"action:\s*cooperate", 0.9),
        (r"my\s+action\s*(?:is)?:?\s*cooperate", 0.85),
        (r"i\s+(?:will\s+)?(?:choose|select|pick)\s+(?:to\s+)?cooperate", 0.85),
        (r"decision:\s*cooperate", 0.85),
        (r"choice:\s*cooperate", 0.85),

        # 自然语言
        (r"i(?:'ll|\s+will)\s+cooperate", 0.8),
        (r"let(?:'s|s)\s+cooperate", 0.75),
        (r"cooperating\s+(?:is|seems)\s+(?:the\s+)?(?:best|better|right)", 0.7),
        (r"(?:strategy|best|optimal)\s+(?:is\s+)?to\s+cooperate", 0.7),
        (r"to\s+cooperate\s+(?:this|now|here)", 0.65),

        # 中文
        (r"(?:我)?(?:选择|决定)?合作", 0.85),
        (r"动作[：:]\s*合作", 0.9),
        (r"行动[：:]\s*合作", 0.9),
        (r"选择[：:]\s*合作", 0.85),

        # 单字母 (低置信度)
        (r"\b(?:action|choice|decision)\s*[：:=]\s*C\b", 0.7),

        # 兜底
        (r"\bcooperate\b", 0.5),
    ]

    # 背叛信号 (按优先级排序)
    DEFECT_PATTERNS = [
        # 精确格式
        (r"ACTION:\s*DEFECT", 1.0),
        (r"ACTION:\s*D\b", 0.95),

        # 宽松格式
        (r"action:\s*defect", 0.9),
        (r"my\s+action\s*(?:is)?:?\s*defect", 0.85),
        (r"i\s+(?:will\s+)?(?:choose|select|pick)\s+(?:to\s+)?defect", 0.85),
        (r"decision:\s*defect", 0.85),
        (r"choice:\s*defect", 0.85),

        # 自然语言
        (r"i(?:'ll|\s+will)\s+defect", 0.8),
        (r"i\s+(?:must|should|have\s+to)\s+defect", 0.75),
        (r"defecting\s+(?:is|seems)\s+(?:the\s+)?(?:best|better|right|optimal)", 0.7),
        (r"(?:have|need)\s+to\s+defect", 0.7),
        (r"(?:strategy|best|optimal)\s+(?:is\s+)?to\s+defect", 0.7),
        (r"to\s+defect\s+(?:this|now|here)", 0.65),

        # 同义词
        (r"i(?:'ll|\s+will)\s+betray", 0.8),
        (r"i\s+(?:choose|select)\s+(?:to\s+)?betray", 0.8),

        # 中文
        (r"(?:我)?(?:选择|决定)?背叛", 0.85),
        (r"(?:我)?(?:选择|决定)?不合作", 0.8),
        (r"动作[：:]\s*背叛", 0.9),
        (r"行动[：:]\s*背叛", 0.9),
        (r"选择[：:]\s*背叛", 0.85),
        (r"动作[：:]\s*D\b", 0.85),

        # 单字母 (低置信度)
        (r"\b(?:action|choice|decision)\s*[：:=]\s*D\b", 0.7),

        # 兜底
        (r"\bdefect\b(?!\s*(?:ion|ive|or))", 0.5),
    ]

    def __init__(self):
        self.stats = {
            "total": 0,
            "success": 0,
            "fallback": 0,
            "ambiguous": 0,
            "failed": 0,
            "forced_cooperate": 0,
            "forced_defect": 0,
        }
        self._Action = None  # 延迟导入

    @property
    def Action(self):
        if self._Action is None:
            try:
                from .games import Action
            except ImportError:
                from games import Action
            self._Action = Action
        return self._Action

    def parse(self, response: str) -> ParseResult:
        """解析 LLM 响应"""
        self.stats["total"] += 1

        if not response or not response.strip():
            self.stats["failed"] += 1
            return self._random_fallback("", "empty response")

        text = response.strip()

        # 扫描合作信号
        coop_signals = self._scan_patterns(text, self.COOPERATE_PATTERNS)
        # 扫描背叛信号
        defect_signals = self._scan_patterns(text, self.DEFECT_PATTERNS)

        # 计算最高置信度
        max_coop = max([conf for _, conf in coop_signals], default=0)
        max_defect = max([conf for _, conf in defect_signals], default=0)

        result = ParseResult(
            raw_response=response,
            cooperate_signals=coop_signals,
            defect_signals=defect_signals,
        )

        # 决策逻辑
        if max_coop == 0 and max_defect == 0:
            # 无信号 -> 随机
            self.stats["failed"] += 1
            return self._random_fallback(response, "no signal")

        if max_coop > 0 and max_defect > 0:
            # 有矛盾信号
            diff = abs(max_coop - max_defect)
            if diff < 0.2:
                # 置信度接近 -> 矛盾
                self.stats["ambiguous"] += 1
                return self._random_fallback(response, "ambiguous")
            # 否则取高的

        if max_coop > max_defect:
            result.action = self.Action.COOPERATE
            result.confidence = max_coop
            result.matched_pattern = coop_signals[0][0] if coop_signals else ""
            result.status = ParseStatus.SUCCESS if max_coop >= 0.7 else ParseStatus.FALLBACK
        else:
            result.action = self.Action.DEFECT
            result.confidence = max_defect
            result.matched_pattern = defect_signals[0][0] if defect_signals else ""
            result.status = ParseStatus.SUCCESS if max_defect >= 0.7 else ParseStatus.FALLBACK

        self.stats["success" if result.status == ParseStatus.SUCCESS else "fallback"] += 1
        return result

    def _scan_patterns(self, text: str, patterns: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """扫描匹配模式"""
        matches = []
        for pattern, confidence in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches.append((pattern, confidence))
        # 按置信度排序
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def _random_fallback(self, response: str, reason: str) -> ParseResult:
        """随机选择（消除偏差）"""
        action = random.choice([self.Action.COOPERATE, self.Action.DEFECT])
        if action == self.Action.COOPERATE:
            self.stats["forced_cooperate"] += 1
        else:
            self.stats["forced_defect"] += 1

        return ParseResult(
            action=action,
            status=ParseStatus.FAILED,
            confidence=0.0,
            matched_pattern=f"random ({reason})",
            raw_response=response,
        )

    def get_stats(self) -> Dict:
        """获取解析统计"""
        total = self.stats["total"]
        if total == 0:
            return self.stats

        return {
            **self.stats,
            "success_rate": self.stats["success"] / total,
            "fallback_rate": self.stats["fallback"] / total,
            "ambiguous_rate": self.stats["ambiguous"] / total,
            "failure_rate": self.stats["failed"] / total,
        }


# ============================================================
# LLM 策略主类
# ============================================================

class LLMStrategy:
    """
    基于 LLM 的博弈策略

    模式:
    - pure: LLM 自己分析历史
    - hybrid: 代码预处理历史，告诉 LLM 统计信息

    特点:
    1. 宽松解析，支持多种输出格式
    2. 解析失败时随机选择（无偏差）
    3. max_tokens=1000 防止截断
    """

    DEFAULT_MAX_TOKENS = 1000

    def __init__(self,
                 provider: str = "deepseek",
                 mode: str = "hybrid",
                 game_config = None,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 temperature: float = 0.7,
                 persona_prompt: str = None,
                 history_window: int = None,
                 enable_cheap_talk: bool = False,
                 agent_name: str = "Player",
                 personality: str = "rational and analytical",
                 strategy_tendency: str = "balanced"):
        """
        Args:
            provider: LLM 提供商 (deepseek/openai/gemini)
            mode: 策略模式 (pure/hybrid)
            game_config: 博弈配置
            max_tokens: 最大 token 数
            temperature: 温度参数
            persona_prompt: 自定义人格提示
            history_window: 历史窗口大小 (None 表示使用全部历史)
            enable_cheap_talk: 是否启用 cheap talk 消息功能
            agent_name: 智能体名称
            personality: 性格描述
            strategy_tendency: 策略倾向
        """
        self.provider = provider
        self.mode = mode
        self.game_config = game_config
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.persona_prompt = persona_prompt
        self.history_window = history_window
        self.enable_cheap_talk = enable_cheap_talk
        self.agent_name = agent_name
        self.personality = personality
        self.strategy_tendency = strategy_tendency

        self.parser = ResponseParser()
        self._client = None

        # 调试信息
        self.raw_responses: List[str] = []
        self.total_payoff = 0.0
        self.current_round = 0
        self.last_message: str = ""  # 最后生成的消息

    @property
    def name(self) -> str:
        """策略名称，用于显示和日志"""
        return f"LLM ({self.provider}/{self.mode})"

    @property
    def client(self):
        """延迟加载 LLM 客户端"""
        if self._client is None:
            try:
                from .llm_api import LLMClient
            except ImportError:
                from llm_api import LLMClient
            self._client = LLMClient(provider=self.provider)
        return self._client

    @property
    def Action(self):
        """延迟导入 Action"""
        try:
            from .games import Action
        except ImportError:
            from games import Action
        return Action

    def choose_action(self,
                      my_history: List,
                      opponent_history: List,
                      opponent_name: str = "Opponent",
                      opponent_message: str = None) -> Any:
        """
        选择动作

        Args:
            my_history: 我的动作历史 [Action, ...]
            opponent_history: 对手动作历史 [Action, ...]
            opponent_name: 对手名称
            opponent_message: 对手发送的消息 (cheap talk)

        Returns:
            Action.COOPERATE 或 Action.DEFECT
        """
        prompt = self._build_prompt(my_history, opponent_history, opponent_name, opponent_message)

        try:
            response = self.client.chat(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            self.raw_responses.append(response)

            result = self.parser.parse(response)
            return result.action

        except Exception as e:
            # API 错误时随机选择
            return random.choice([self.Action.COOPERATE, self.Action.DEFECT])

    def _build_prompt(self,
                      my_history: List,
                      opponent_history: List,
                      opponent_name: str,
                      opponent_message: str = None) -> str:
        """构建 LLM 提示"""

        # 获取博弈描述
        if self.game_config:
            try:
                from .games import get_payoff_description
            except ImportError:
                from games import get_payoff_description
            game_desc = get_payoff_description(self.game_config)
        else:
            game_desc = "Standard Prisoner's Dilemma"

        if self.mode == "pure":
            return self._build_pure_prompt(my_history, opponent_history, opponent_name, game_desc, opponent_message)
        else:
            return self._build_hybrid_prompt(my_history, opponent_history, opponent_name, game_desc, opponent_message)

    def _build_pure_prompt(self,
                           my_history: List,
                           opponent_history: List,
                           opponent_name: str,
                           game_desc: str,
                           opponent_message: str = None) -> str:
        """Pure 模式提示 - LLM 自己分析"""
        self.current_round = len(my_history) + 1
        history_str = self._format_history(my_history, opponent_history)

        my_coop_rate = "0%" if not my_history else f"{sum(1 for a in my_history if self._get_action_value(a) == 'cooperate') / len(my_history):.0%}"
        opp_coop_rate = "N/A" if not opponent_history else f"{sum(1 for a in opponent_history if self._get_action_value(a) == 'cooperate') / len(opponent_history):.0%}"

        template = _load_template("strategy_select")
        prompt = template.format(
            opponent_name=opponent_name,
            game_desc=game_desc,
            current_round=self.current_round,
            total_payoff=f"{self.total_payoff:.1f}",
            my_coop_rate=my_coop_rate,
            opp_coop_rate=opp_coop_rate,
            opp_pattern=self._analyze_pattern(opponent_history),
            history_summary=history_str if history_str else "No history yet (first round)",
        )

        # 如果有对手消息，添加到提示中
        if opponent_message:
            message_section = f"\n=== OPPONENT MESSAGE ===\n{opponent_name} says: \"{opponent_message}\"\n"
            # 插入到 YOUR TASK 之前
            prompt = prompt.replace("=== YOUR TASK ===", message_section + "=== YOUR TASK ===")

        return prompt

    def _build_hybrid_prompt(self,
                             my_history: List,
                             opponent_history: List,
                             opponent_name: str,
                             game_desc: str,
                             opponent_message: str = None) -> str:
        """Hybrid 模式提示 - 代码预处理统计"""
        self.current_round = len(my_history) + 1
        rounds_played = len(opponent_history)

        if rounds_played == 0:
            my_coop_rate_str = "0%"
            opp_coop_rate_str = "N/A"
            history_summary = "No history yet (first round)"
        else:
            window = self.history_window if self.history_window else rounds_played
            windowed_opp = opponent_history[-window:]
            windowed_my = my_history[-window:]
            window_size = len(windowed_opp)

            opp_coop = sum(1 for a in windowed_opp if self._get_action_value(a) == "cooperate")
            opp_coop_rate = opp_coop / window_size
            my_coop = sum(1 for a in windowed_my if self._get_action_value(a) == "cooperate")
            my_coop_rate = my_coop / window_size if window_size > 0 else 0

            my_coop_rate_str = f"{my_coop_rate:.0%}"
            opp_coop_rate_str = f"{opp_coop_rate:.0%} ({opp_coop}/{window_size})"

            window_info = f" (last {window} rounds)" if self.history_window else ""
            last_action = self._get_action_value(opponent_history[-1])
            history_summary = f"Rounds: {rounds_played}{window_info}, Last opponent action: {last_action}"

        template = _load_template("strategy_select")
        prompt = template.format(
            opponent_name=opponent_name,
            game_desc=game_desc,
            current_round=self.current_round,
            total_payoff=f"{self.total_payoff:.1f}",
            my_coop_rate=my_coop_rate_str,
            opp_coop_rate=opp_coop_rate_str,
            opp_pattern=self._analyze_pattern(opponent_history),
            history_summary=history_summary,
        )

        # 如果有对手消息，添加到提示中
        if opponent_message:
            message_section = f"\n=== OPPONENT MESSAGE ===\n{opponent_name} says: \"{opponent_message}\"\n"
            # 插入到 YOUR TASK 之前
            prompt = prompt.replace("=== YOUR TASK ===", message_section + "=== YOUR TASK ===")

        return prompt

    def _analyze_pattern(self, opponent_history: List) -> str:
        """分析对手行为模式"""
        if len(opponent_history) < 3:
            return "Not enough data"

        recent = opponent_history[-10:] if len(opponent_history) >= 10 else opponent_history
        coop_count = sum(1 for a in recent if self._get_action_value(a) == "cooperate")
        coop_rate = coop_count / len(recent)

        if coop_rate >= 0.8:
            return "Highly cooperative"
        elif coop_rate >= 0.5:
            return "Mixed/Conditional cooperator"
        elif coop_rate >= 0.2:
            return "Mostly defects"
        else:
            return "Always defects"

    def _format_history(self, my_history: List, opponent_history: List) -> str:
        """格式化历史记录"""
        if not my_history:
            return ""

        # 使用 history_window 限制历史长度，默认为全部
        window = self.history_window if self.history_window else len(my_history)

        lines = []
        for i, (my_act, opp_act) in enumerate(zip(my_history, opponent_history), 1):
            my_str = my_act.value if hasattr(my_act, 'value') else str(my_act)
            opp_str = opp_act.value if hasattr(opp_act, 'value') else str(opp_act)
            lines.append(f"Round {i}: You={my_str}, Opponent={opp_str}")

        return "\n".join(lines[-window:])

    def _get_action_value(self, action) -> str:
        """安全获取动作的字符串值，兼容 Action 枚举和字符串"""
        if hasattr(action, 'value'):
            return action.value
        return str(action).lower()

    def _get_payoffs(self) -> Dict[str, int]:
        """从 game_config 提取收益矩阵数值"""
        if not self.game_config:
            return {"cc": 3, "cd": 0, "dc": 5, "dd": 1}  # 默认囚徒困境
        matrix = self.game_config.payoff_matrix
        return {
            "cc": matrix[(self.Action.COOPERATE, self.Action.COOPERATE)][0],
            "cd": matrix[(self.Action.COOPERATE, self.Action.DEFECT)][0],
            "dc": matrix[(self.Action.DEFECT, self.Action.COOPERATE)][0],
            "dd": matrix[(self.Action.DEFECT, self.Action.DEFECT)][0],
        }

    def generate_message(self,
                         my_history: List,
                         opponent_history: List,
                         opponent_name: str = "Opponent") -> str:
        """
        生成 Cheap Talk 消息

        Args:
            my_history: 我的动作历史
            opponent_history: 对手动作历史
            opponent_name: 对手名称

        Returns:
            要发送给对手的消息
        """
        if not self.enable_cheap_talk:
            return ""

        rounds_played = len(opponent_history)

        # 构建消息生成提示
        prompt = f"""You are playing an iterated game against {opponent_name}.

Rounds played so far: {rounds_played}

You can send a short message to your opponent before making your decision.
This message could be used to signal your intentions, build trust, or strategize.

Generate a brief message (1-2 sentences max) to send to your opponent.
The message should be strategic and relevant to the game.

MESSAGE:"""

        try:
            response = self.client.chat(
                prompt,
                max_tokens=100,
                temperature=self.temperature,
            )

            # 提取消息
            message = response.strip()
            if "MESSAGE:" in message:
                message = message.split("MESSAGE:")[-1].strip()

            self.last_message = message
            return message

        except Exception as e:
            return ""

    def update_payoff(self, payoff: float):
        """更新累计收益"""
        self.total_payoff += payoff

    def get_debug_info(self) -> Dict:
        """获取调试信息"""
        return {
            "provider": self.provider,
            "mode": self.mode,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "total_calls": len(self.raw_responses),
            "parse_quality": self.parser.get_stats(),
            "last_raw_response": self.raw_responses[-1] if self.raw_responses else None,
        }

    def reset(self):
        """重置状态"""
        self.raw_responses = []
        self.total_payoff = 0.0
        self.last_message = ""
        self.parser = ResponseParser()


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("解析器测试")
    print("=" * 60)

    parser = ResponseParser()

    test_cases = [
        ("ACTION: COOPERATE", "COOPERATE", ParseStatus.SUCCESS),
        ("ACTION: DEFECT", "DEFECT", ParseStatus.SUCCESS),
        ("action: cooperate", "COOPERATE", ParseStatus.SUCCESS),
        ("action: defect", "DEFECT", ParseStatus.SUCCESS),
        ("I will cooperate.", "COOPERATE", ParseStatus.SUCCESS),
        ("I choose to defect.", "DEFECT", ParseStatus.SUCCESS),
        ("I'll defect this round.", "DEFECT", ParseStatus.SUCCESS),
        ("Let's cooperate for mutual benefit.", "COOPERATE", ParseStatus.SUCCESS),
        ("我选择合作", "COOPERATE", ParseStatus.SUCCESS),
        ("我选择背叛", "DEFECT", ParseStatus.SUCCESS),
        ("After careful analysis, I think the best strategy is to defect this round.",
         "DEFECT", ParseStatus.SUCCESS),
        ("The optimal strategy is to cooperate here.",
         "COOPERATE", ParseStatus.SUCCESS),
        ("Hello world", None, ParseStatus.FAILED),
        ("The weather is nice today.", None, ParseStatus.FAILED),
    ]

    passed = 0
    for text, expected_action, expected_status in test_cases:
        result = parser.parse(text)

        if expected_action is None:
            success = result.status == expected_status
        else:
            success = (result.action.value.upper() == expected_action and
                      result.status in [ParseStatus.SUCCESS, ParseStatus.FALLBACK])

        icon = "✅" if success else "❌"
        action_str = result.action.value.upper() if result.action else "None"
        status_str = "[success]" if success else "[failed]"

        print(f"{icon} '{text[:40]:40s}' -> {action_str:10s} {status_str}")

        if success:
            passed += 1

    print(f"通过: {passed}/{len(test_cases)}")
    print(f"统计: {parser.get_stats()}")
