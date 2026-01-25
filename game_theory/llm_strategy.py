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
# 模板路径
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
                 enable_cheap_talk: bool = False):
        """
        Args:
            provider: LLM 提供商 (deepseek/openai/claude)
            mode: 策略模式 (pure/hybrid)
            game_config: 博弈配置
            max_tokens: 最大 token 数
            temperature: 温度参数
            persona_prompt: 自定义人格提示
            history_window: 历史窗口大小 (None 表示使用全部历史)
            enable_cheap_talk: 是否启用 cheap talk 消息功能
        """
        self.provider = provider
        self.mode = mode
        self.game_config = game_config
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.persona_prompt = persona_prompt
        self.history_window = history_window
        self.enable_cheap_talk = enable_cheap_talk

        self.parser = ResponseParser()
        self._client = None

        # 调试信息
        self.raw_responses: List[str] = []
        self.total_payoff = 0.0
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
                      opponent_name: str = "Opponent") -> Any:
        """
        选择动作

        Args:
            my_history: 我的动作历史 [Action, ...]
            opponent_history: 对手动作历史 [Action, ...]
            opponent_name: 对手名称

        Returns:
            Action.COOPERATE 或 Action.DEFECT
        """
        prompt = self._build_prompt(my_history, opponent_history, opponent_name)

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
                      opponent_name: str) -> str:
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
            return self._build_pure_prompt(my_history, opponent_history, opponent_name, game_desc)
        else:
            return self._build_hybrid_prompt(my_history, opponent_history, opponent_name, game_desc)

    def _build_pure_prompt(self,
                           my_history: List,
                           opponent_history: List,
                           opponent_name: str,
                           game_desc: str) -> str:
        """Pure 模式提示 - LLM 自己分析"""
        history_str = self._format_history(my_history, opponent_history)
        context = f"Game History:\n{history_str}" if history_str else "This is the first round."

        template = _load_template("strategy_select")
        return template.format(
            opponent_name=opponent_name,
            game_desc=game_desc,
            context=context
        )

    def _build_hybrid_prompt(self,
                             my_history: List,
                             opponent_history: List,
                             opponent_name: str,
                             game_desc: str) -> str:
        """Hybrid 模式提示 - 代码预处理统计"""
        rounds_played = len(opponent_history)

        if rounds_played == 0:
            context = "This is the first round. No history yet."
        else:
            # 应用历史窗口限制
            window = self.history_window if self.history_window else rounds_played
            windowed_opp = opponent_history[-window:]
            windowed_my = my_history[-window:]
            window_size = len(windowed_opp)

            # 计算对手统计（窗口内）
            opp_coop = sum(1 for a in windowed_opp if self._get_action_value(a) == "cooperate")
            opp_coop_rate = opp_coop / window_size

            # 最近趋势
            recent = windowed_opp[-5:] if len(windowed_opp) >= 5 else windowed_opp
            recent_coop = sum(1 for a in recent if self._get_action_value(a) == "cooperate")
            recent_rate = recent_coop / len(recent)

            # 我的统计（窗口内）
            my_coop = sum(1 for a in windowed_my if self._get_action_value(a) == "cooperate")
            my_coop_rate = my_coop / window_size if window_size > 0 else 0

            # 获取最后动作的字符串值
            last_action = self._get_action_value(opponent_history[-1]) if opponent_history else 'N/A'
            window_info = f" (window: last {window} rounds)" if self.history_window else ""

            context = f"""Rounds played: {rounds_played}{window_info}

Opponent Statistics:
- Overall cooperation rate: {opp_coop_rate:.1%} ({opp_coop}/{window_size})
- Recent 5 rounds cooperation: {recent_rate:.1%}
- Last action: {last_action}

Your Statistics:
- Your cooperation rate: {my_coop_rate:.1%}
- Your total payoff so far: {self.total_payoff:.1f}"""

        template = _load_template("strategy_select")
        return template.format(
            opponent_name=opponent_name,
            game_desc=game_desc,
            context=context
        )

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