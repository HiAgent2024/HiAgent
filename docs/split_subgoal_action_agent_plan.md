# 拆分 Subgoal-Action Agent 计划

## 目标

实现一个新的 HiAgent 变体，把 subgoal 生成和 action 生成拆开，同时不改变原始 HiAgent 的运行逻辑。

原始 `ContextEfficientAgentV2` 目前会让 LLM 在一次回复里同时生成 subgoal 和第一个 action：

```text
Subgoal: clear b6
Action: unstack b3 b6
```

新的变体应该尽量保留原始 prompt。第一步仍然使用原始 HiAgent prompt，让模型判断是否需要新 subgoal，并生成该 subgoal。如果回复里包含新 subgoal，agent 保存这个 subgoal，但丢弃同一次回复里的第一个 action，然后再进行第二次 LLM 调用，把保存下来的 subgoal grounding 成可执行 action。

因此拆分方式是：

```text
Stage 1: 原始 HiAgent prompt -> Subgoal + draft Action，或 Action
Stage 2: 如果 Stage 1 生成了 Subgoal，则基于该 Subgoal 重新生成 Action
```

这个设计用于支持云边协同实验：云端大模型负责生成高层 subgoal，本地或边缘模型负责把 subgoal grounding 成可执行 action。

## 设计原则

- 不改变原始 `ContextEfficientAgentV2` 的执行逻辑。
- 新行为放在新文件里实现。
- 只有显式指定新的 agent 名称时，才启用新逻辑。
- 尽量复用原始 HiAgent 的 memory、history serialization、解析逻辑和 prompt 结构。
- subgoal 生成条件尽量和原代码保持一致。
- 第一次 `Subgoal + Action` 回复里的 action 视为 draft action，用第二次 grounding 调用替换它。
- 第一版不加入额外 verifier 或 LLM judge，避免和原始实现差异过大。

## 拟新增/修改文件

```text
agentboard/agents/cme_split.py
agentboard/agents/__init__.py
```

可选：

```text
eval_configs/hiagent/blocksworld.yaml
```

## 新 Agent

新增 agent 类：

```python
SplitSubgoalActionAgent
```

推荐实现方式：

```python
from agents.cme_final import ContextEfficientAgentV2

class SplitSubgoalActionAgent(ContextEfficientAgentV2):
    ...
```

这样新 agent 可以复用：

- `reset()`
- `update()`
- `make_prompt()` 的组件
- memory 结构
- action parser
- retrieve 相关逻辑
- 现有 PDDL prompt 上下文

## 运行流程

原始 HiAgent 流程：

```text
agent.run()
  -> make_prompt()
  -> LLM 生成 Subgoal + Action 或 Action
  -> 如果有 Subgoal，则解析并保存
  -> 返回 action 给环境
```

新的 split-agent 流程：

```text
agent.run()
  -> 使用原始 HiAgent prompt 结构调用 make_prompt()
  -> 第一次调用 LLM，和 ContextEfficientAgentV2 一致
  -> 如果回复只包含 Action:
       按原始 agent 逻辑解析并返回 action
  -> 如果回复包含 Subgoal + Action:
       解析并保存 Subgoal 到 memory
       丢弃第一次回复里的 draft Action
       调用 generate_action_for_subgoal()
       解析并返回重新生成的 action
  -> 返回 action 给环境
```

## LLM 调用

### 1. 原始 HiAgent 决策调用

这一步应尽量复用原始 `make_prompt()` 输出。

期望输出格式不变：

```text
Subgoal: clear b6
Action: unstack b3 b6
```

或者：

```text
Action: putdown b1
```

这样可以保留原模型关于“是否需要新 subgoal”的判断方式。

### 2. 新 Subgoal 的 Action 重新生成

只有当第一次调用产生 `Subgoal + Action` 时，才执行这一步。

第一次调用得到的 subgoal 会被保存，但第一次 action 被视为 draft，不执行。

期望输出：

```text
Action: unstack b3 b6
```

第二次 prompt 应包含：

- final goal
- current observation
- current subgoal
- relevant history
- 可用辅助命令，例如 `check valid actions`
- 在可行情况下，保留和原 prompt 一致的 domain instruction 与 examples

第二次 prompt 应要求模型只输出一个 action，并且该 action 要直接服务于当前 subgoal。

如果第一次调用只产生 `Action`，则不进行第二次调用。

## Subgoal-Action 一致性

第一版只使用 prompt-level 的一致性约束，保持和原代码风格接近。

示例 action prompt 约束：

```text
Current Subgoal: clear b6

You must output one executable action that directly helps achieve the current subgoal.
Do not pursue another subgoal.
If the current action cannot be determined, use:
Action: check valid actions
```

第一版不加入额外的规则检查器或 LLM judge。

## Subgoal 刷新逻辑

为了贴近原始行为，新 agent 第一版不加入显式的 subgoal completion verifier。

第一次 LLM 调用仍然使用原始 HiAgent prompt，由它判断是否需要新 subgoal：

- 如果第一次回复包含 `Subgoal`，保存 subgoal，并重新生成对应 action。
- 如果第一次回复只包含 `Action`，直接执行该 action。

这样可以避免引入新的 `Subgoal complete` 信号，保持 subgoal 刷新行为接近原始实现。

## 日志

原始 agent 的日志保持不变。

对于新 agent，可以选择在终端 debug 输出中区分两次调用：

```text
-------------Subgoal Response---------
Subgoal: clear b6
---------------[END]------------

-------------Action Response---------
Action: unstack b3 b6
---------------[END]------------
```

运行日志应使用单独路径，例如：

```text
logs/hiagent/blocksworld_split_qwen3_5_4b
```

## 测试计划

先做 smoke test：

```bash
OPENAI_API_BASE=http://127.0.0.1:8000/v1 \
OPENAI_API_KEY=dummy \
EVALTASK=blocksworld \
python agentboard/eval_main.py \
  --cfg-path eval_configs/hiagent/blocksworld.yaml \
  --tasks pddl \
  --model qwen3_5_4b_vllm_server \
  --agent SplitSubgoalActionAgent \
  --max_num_steps 5 \
  --memory_size 100 \
  --log_path ./logs/hiagent/smoke_blocksworld_split_qwen3_5_4b
```

正式测试：

```bash
OPENAI_API_BASE=http://127.0.0.1:8000/v1 \
OPENAI_API_KEY=dummy \
EVALTASK=blocksworld \
python agentboard/eval_main.py \
  --cfg-path eval_configs/hiagent/blocksworld.yaml \
  --tasks pddl \
  --model qwen3_5_4b_vllm_server \
  --agent SplitSubgoalActionAgent \
  --max_num_steps 30 \
  --memory_size 100 \
  --log_path ./logs/hiagent/blocksworld_split_qwen3_5_4b
```

对比对象：

```text
ContextEfficientAgentV2
VanillaAgent
```

## 风险

1. LLM 调用次数会增加，尤其是在频繁生成新 subgoal 时。
2. 拆分 subgoal 和 action 生成后，二者之间的语义绑定可能变弱。
3. action generator 可能输出合法但对当前 subgoal 没有帮助的 action。
4. 第一次调用仍然会生成 draft action，但这部分输出会被丢弃。
5. 第二次 action-grounding prompt 必须尽量贴近原始上下文，否则行为可能偏离原始 HiAgent。

## 预期收益

该拆分设计可以分别评估：

- 高层规划 / subgoal 生成能力
- 低层 action grounding 能力

它也为云边协同提供清晰接口：

```text
Cloud LLM: generate subgoal
Edge/local LLM: generate executable action
Environment: execute action and return observation
```

原始 HiAgent 仍然通过下面命令保持不变：

```bash
--agent ContextEfficientAgentV2
```

拆分版本只通过下面参数启用：

```bash
--agent SplitSubgoalActionAgent
```
