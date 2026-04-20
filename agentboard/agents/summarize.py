"""
TrajectorySummarizer：HiAgent 论文 §3.3 (Observation Summarization) 模块的最小可用复刻。

背景：HiAgent 官方仓库 (HiAgent2024/HiAgent) 的 agentboard/agents/ 目录下缺失 summarize.py，
issue #3 与 #4 均反映该模块缺失但作者尚未修复。本文件依据论文 §3.3 给出的 prompt 模板
以及 cme_final.py 中对 TrajectorySummarizer 的调用（cme_final.py:160-182）复刻。

调用（来源：cme_final.py）：
    summarizer = TrajectorySummarizer(self.llm_model)
    summary = summarizer.generate_summary([trajectory], [subgoal])[0]

其中：
- trajectory 是双重列表：List[List[(role, content)]]，每个内层 list 通常是
  [('Action', ...), ('Observation', ...)]，调用前已剔除 "check valid actions" 项。
- subgoal 是 ('Subgoal', '<子目标文本>') 形式的元组。
- 返回值是与输入等长的字符串列表，每个元素是该 (trajectory, subgoal) 的浓缩摘要，
  会被填入 cme_final.py 中作为压缩后的 Observation。

LLM 接口签名（来源：agentboard/llm/openai_gpt.py:74 等）：
    llm_model.generate(system_message: str, prompt: str) -> Tuple[bool, str]
"""

from typing import List, Sequence, Tuple


# 论文 §3.3 给出的 prompt 模板。原文中存在 {example} few-shot 占位符，但论文与官方仓库均未
# 提供具体的示例内容，此处保持 zero-shot 模式，不渲染该占位符。
_PROMPT_TEMPLATE = """You are an advanced AI system tasked with summarizing and analyzing a series of action-observation pairs (trajectories) and determining whether a specific subgoal has been met.

Your goal is to create a summary that captures all essential information, decisions, and outcomes from the given trajectories, and indicate whether the subgoal has been met based on the summarized observations.
If there are no valid actions taken, you need to analyze the reason.

### Instructions:
1. Provide a summarized observation related to the subgoal in a concise manner.
2. Determine whether the subgoal has been met.
3. Do not output anything except whether summary and subgoal are met. Your output should be only one line. Do not output things like '##Summary', '##Summary and Analysis'.

##Trajectory
{formatted_trajectory}

##Subgoal
{subgoal}

###Output:"""


_SYSTEM_MESSAGE = "You are a helpful assistant."


class TrajectorySummarizer:
    """对一个已完成子目标对应的 action-observation 轨迹进行 LLM 摘要。"""

    def __init__(self, llm_model):
        self.llm_model = llm_model

    @staticmethod
    def _format_trajectory(trajectory: Sequence[Sequence[Tuple[str, str]]]) -> str:
        """把 cme_final.py 中的双重列表轨迹格式化成 'Action: xxx\nObservation: yyy' 的字符串。

        与 cme_final.py:vanilla_serialize_history 中的拼接逻辑保持一致，确保 LLM 看到的
        轨迹格式与 working memory 中其他位置一致。
        """
        lines = []
        for chunk in trajectory:
            for role, content in chunk:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _format_subgoal(subgoal) -> str:
        """从 ('Subgoal', '<text>') 元组中取出子目标文本；对异常输入做兜底处理。"""
        if isinstance(subgoal, tuple) and len(subgoal) >= 2:
            text = subgoal[1]
            return text.strip() if isinstance(text, str) else str(text)
        return str(subgoal)

    @staticmethod
    def _last_observation(trajectory: Sequence[Sequence[Tuple[str, str]]]) -> str:
        """LLM 调用失败时的降级摘要：取轨迹中的最后一个 Observation 作为代替。"""
        for chunk in reversed(trajectory):
            for role, content in reversed(chunk):
                if role == "Observation":
                    return content
        return ""

    def generate_summary(
        self,
        trajectories: Sequence[Sequence[Sequence[Tuple[str, str]]]],
        subgoals: Sequence[Tuple[str, str]],
    ) -> List[str]:
        """对一批 (trajectory, subgoal) 生成压缩摘要。

        Args:
            trajectories: 轨迹列表，每个轨迹是 List[List[(role, content)]]。
            subgoals: 与 trajectories 等长的子目标元组列表。

        Returns:
            字符串列表，与输入等长，每项是单行的浓缩摘要。
        """
        summaries: List[str] = []
        for trajectory, subgoal in zip(trajectories, subgoals):
            formatted_trajectory = self._format_trajectory(trajectory)
            subgoal_text = self._format_subgoal(subgoal)
            prompt = _PROMPT_TEMPLATE.format(
                formatted_trajectory=formatted_trajectory,
                subgoal=subgoal_text,
            )

            success, completion = self.llm_model.generate(_SYSTEM_MESSAGE, prompt)

            if success and completion:
                # 论文 instruction 3 要求输出仅一行；此处取第一行非空内容作为最终摘要。
                summary = ""
                for line in completion.strip().splitlines():
                    line = line.strip()
                    if line:
                        summary = line
                        break
                if not summary:
                    summary = self._last_observation(trajectory)
            else:
                # LLM 调用失败：退化为「最后一个 Observation」，与 cme_final.py 中
                # summarization=False 分支的行为对齐，避免训练流程中断。
                summary = self._last_observation(trajectory)

            summaries.append(summary)
        return summaries
