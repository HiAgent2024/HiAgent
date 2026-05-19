# PlugMem trajectory API initial_observation 修复方案

## 背景

PlugMem 原生 `Memory` 的正确使用方式是显式传入初始观察，然后逐步追加动作后的观察：

```python
memory = Memory(goal=goal, observation=obs_0)
memory.append(action_t0=action_1, observation_t1=obs_1)
memory.append(action_t0=action_2, observation_t1=obs_2)
memory.close()
graph.insert(memory)
```

语义是：

```text
obs_0 --action_1--> obs_1
obs_1 --action_2--> obs_2
```

`Memory.append()` 内部会使用当前 `self.observation_t0` 和 `action_t0` 抽取 subgoal/reward/state，然后把 `self.observation_t0` 更新为 `observation_t1`。

## 当前 HTTP trajectory API 的问题

当前 `PlugMem/plugmem/api/routes/memories.py` 的 trajectory mode 使用：

```python
mem = Memory(
    goal=body.goal,
    observation=body.steps[0].observation,
    llm=llm,
    embedder=embedder,
    time=graph.semantic_time,
    session_id=body.session_id,
)
for step in body.steps:
    mem.append(action_t0=step.action, observation_t1=step.observation)
mem.close()
graph.insert(mem)
```

因为 request schema 只有 `steps[].observation` 和 `steps[].action`，没有单独的初始观察字段，所以 `steps[0].observation` 会同时被当作：

- 初始 observation
- 第一步 action 执行后的 observation

这会让第一步状态转移变成：

```text
obs_1 --action_1--> obs_1
```

而不是期望的：

```text
obs_0 --action_1--> obs_1
```

## 推荐改动

在 `PlugMem/plugmem/api/schemas.py` 的 `MemoryInsertRequest` 中新增可选字段：

```python
initial_observation: Optional[str] = None
```

然后在 `PlugMem/plugmem/api/routes/memories.py` 的 `_insert_trajectory()` 中优先使用该字段：

```python
initial_observation = body.initial_observation or body.steps[0].observation

mem = Memory(
    goal=body.goal,
    observation=initial_observation,
    llm=llm,
    embedder=embedder,
    time=graph.semantic_time,
    session_id=body.session_id,
)
for step in body.steps:
    mem.append(action_t0=step.action, observation_t1=step.observation)
mem.close()
graph.insert(mem)
```

这样旧请求仍然兼容；新请求可以严格对齐 PlugMem 原生 `Memory(initial_obs) + append(action, next_obs)` 逻辑。

## 新 payload 格式

HiAgent 上传 trajectory 时应使用：

```json
{
  "mode": "trajectory",
  "goal": "current task goal",
  "initial_observation": "obs_0",
  "steps": [
    {
      "action": "action_1",
      "observation": "obs_1"
    },
    {
      "action": "action_2",
      "observation": "obs_2"
    }
  ],
  "session_id": "optional-run-id"
}
```

服务端解释为：

```text
Memory(goal, observation=obs_0)
append(action_1, obs_1)
append(action_2, obs_2)
```

这与 PlugMem 原始逻辑一致。

## HiAgent 上传侧转换

HiAgent 的 `self.memory` 通常类似：

```python
[
    [("Observation", obs_0)],
    [("Action", action_1), ("Observation", obs_1)],
    [("Action", action_2), ("Observation", obs_2)],
]
```

转换时应拆成 `initial_observation` 和 `steps`：

```python
def memory_to_plugmem_payload(goal, memory, session_id=None):
    observations = []
    actions = []

    for turn in memory:
        for key, value in turn:
            if key == "Observation" and value:
                observations.append(value)
            elif key == "Action" and value:
                actions.append(value)

    if not observations or not actions:
        return None

    steps = [
        {
            "action": actions[i],
            "observation": observations[i + 1],
        }
        for i in range(min(len(actions), len(observations) - 1))
    ]

    if not steps:
        return None

    return {
        "mode": "trajectory",
        "goal": goal,
        "initial_observation": observations[0],
        "steps": steps,
        "session_id": session_id,
    }
```

## 兼容策略

- `initial_observation` 是可选字段，旧 payload 不传时仍 fallback 到 `body.steps[0].observation`。
- 新的 HiAgent 集成应使用 `initial_observation`，不要再使用 synthetic no-op step。
- 不要同时使用 `initial_observation` 和 no-op step；二者是替代方案，叠加会产生一条多余的假轨迹。

## 风险

改动范围较小，主要影响：

- `PlugMem/plugmem/api/schemas.py`
- `PlugMem/plugmem/api/routes/memories.py`
- `PlugMem/tests/test_api_memories.py`

潜在风险：

- 旧调用方如果已经用 no-op step 绕过问题，升级后不应再传 `initial_observation`，否则会保留多余 no-op。
- 空 `steps` 仍不应允许；即使提供 `initial_observation`，没有 action 也无法形成可结构化 trajectory。
- 旧格式请求仍是旧语义；严格对齐只对新格式请求成立。
