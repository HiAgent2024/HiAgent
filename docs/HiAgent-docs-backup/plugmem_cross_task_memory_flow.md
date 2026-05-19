# PlugMem 插件版为 HiAgent 提供跨任务记忆的全流程

本文说明如何复用 PlugMem 插件版背后的服务能力，为 HiAgent 提供跨任务长期记忆。这里的关键点是：PlugMem 的 OpenClaw 插件不能直接安装到 HiAgent，但插件使用的 PlugMem HTTP 服务可以被 HiAgent 的 Python agent 调用。

## 1. 总体输入输出关系

完整链路如下：

```text
HiAgent 当前任务输入
  ├─ goal
  ├─ init observation
  ├─ step observation
  └─ action-observation trajectory
        │
        ▼
Python PlugMem bridge
  ├─ recall: 用当前 goal/observation 查询历史记忆
  └─ remember/upload: 将当前任务轨迹上传到 PlugMem
        │
        ▼
PlugMem 服务
  ├─ LLM: 结构化轨迹、推理召回结果
  ├─ Embedding: 建立向量索引
  └─ Chroma: 持久化 graph 记忆
        │
        ▼
HiAgent prompt 增强
  ├─ 当前任务原始上下文
  ├─ 当前任务短期 working memory
  └─ PlugMem 跨任务 recall 结果
        │
        ▼
HiAgent 输出 action
```

跨任务记忆的核心输入是每个任务执行后的轨迹，核心输出是下一个任务开始或执行中可召回的经验。

## 2. 构建 PlugMem 服务

### 输入

需要准备：

- Python 3.10+
- `uv`
- OpenAI-compatible LLM endpoint
- OpenAI-compatible embedding endpoint，或直接使用 `OPENAI_API_KEY` 走 OpenAI embedding fallback
- 一个 PlugMem 服务 API key
- 一个 Chroma 持久化目录

建议在 `PlugMem` 目录下配置环境变量：

```powershell
cd PlugMem

$env:LLM_BASE_URL="https://api.openai.com/v1"
$env:LLM_API_KEY="你的 LLM API key"
$env:LLM_MODEL="gpt-4o-mini"

# 如果没有单独 embedding 服务，可以使用 OpenAI fallback
$env:OPENAI_API_KEY="你的 OpenAI API key"

$env:CHROMA_MODE="persistent"
$env:CHROMA_PATH="./data/chroma"

$env:PLUGMEM_API_KEY="dev-key-change-me"
```

### 处理

安装 PlugMem 依赖：

```powershell
uv sync
```

启动服务：

```powershell
uv run uvicorn plugmem.api.app:app --host 0.0.0.0 --port 8080
```

### 输出

服务启动后应暴露：

- `http://localhost:8080/api/v1/health`
- `http://localhost:8080/api/v1/graphs`
- `http://localhost:8080/api/v1/graphs/{graph_id}/memories`
- `http://localhost:8080/api/v1/graphs/{graph_id}/reason`
- `http://localhost:8080/inspector/`

健康检查：

```powershell
curl http://localhost:8080/api/v1/health
```

期望输出中：

```json
{
  "status": "ok",
  "llm_available": true,
  "embedding_available": true,
  "chroma_available": true
}
```

如果状态是 `degraded`，说明 LLM、embedding 或 Chroma 至少有一个不可用。

## 3. 创建跨任务记忆 graph

### 输入

需要一个跨任务共享的 graph id。建议：

```text
hiagent-cross-task
```

这个 graph 是 PlugMem 中的记忆命名空间。所有任务都可以写入它，也可以从它召回经验。

### 处理

创建 graph：

```powershell
curl -X POST http://localhost:8080/api/v1/graphs `
  -H "X-API-Key: dev-key-change-me" `
  -H "Content-Type: application/json" `
  -d "{\"graph_id\":\"hiagent-cross-task\"}"
```

### 输出

期望输出：

```json
{
  "graph_id": "hiagent-cross-task",
  "stats": {}
}
```

后续所有跨任务 remember 和 recall 都围绕这个 graph 进行。

## 4. 构建 HiAgent 到 PlugMem 的 Python 桥接层

### 输入

HiAgent agent 在运行时已经有这些信息：

- `self.goal`
- `self.init_obs`
- `self.memory`
- 当前 action
- 当前 observation/state

PlugMem API 需要的主要输入是：

召回输入：

```json
{
  "observation": "当前观察或问题",
  "goal": "当前任务目标",
  "task_type": "任务类型，例如 blockworld"
}
```

上传输入：

```json
{
  "mode": "trajectory",
  "goal": "当前任务目标",
  "steps": [
    {
      "observation": "某一步观察",
      "action": "该步执行动作"
    }
  ],
  "session_id": "可选，用于区分运行实例"
}
```

### 处理

建议新增一个 Python client，例如：

```python
import requests


class PlugMemClient:
    def __init__(self, base_url, api_key, graph_id):
        self.base_url = base_url.rstrip("/")
        self.graph_id = graph_id
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json",
        }

    def recall(self, observation, goal=None, task_type=""):
        response = requests.post(
            f"{self.base_url}/api/v1/graphs/{self.graph_id}/reason",
            headers=self.headers,
            json={
                "observation": observation,
                "goal": goal,
                "task_type": task_type,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["reasoning"]

    def upload_trajectory(self, goal, steps, session_id=None):
        response = requests.post(
            f"{self.base_url}/api/v1/graphs/{self.graph_id}/memories",
            headers=self.headers,
            json={
                "mode": "trajectory",
                "goal": goal,
                "steps": steps,
                "session_id": session_id,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
```

### 输出

桥接层输出两类结果：

- `recall()` 输出一段自然语言记忆推理结果，用于加入 HiAgent prompt。
- `upload_trajectory()` 输出 PlugMem 写入结果和 graph 统计信息。

## 5. 构建 PlugMem 版 HiAgent agent

### 输入

建议以现有 `ContextEfficientAgentV2` 为基础，因为它已经实现：

- 任务内 working memory
- subgoal/action 机制
- prompt 构建
- action 解析

新增 agent 可以命名为：

```text
PlugMemContextEfficientAgent
```

配置输入建议包括：

```yaml
agent:
  name: PlugMemContextEfficientAgent
  memory_size: 100
  need_goal: True
  use_parser: True
  plugmem:
    enabled: True
    base_url: http://localhost:8080
    api_key: dev-key-change-me
    graph_id: hiagent-cross-task
    recall_on_reset: True
    upload_on_finish: True
```

### 处理

在新 agent 中做三件事：

1. `reset(goal, init_obs)`：
   - 接收当前任务 goal 和初始 observation。
   - 调用 PlugMem recall。
   - 将召回结果保存在 `self.plugmem_context`。

2. `make_prompt(...)`：
   - 保留原始 HiAgent prompt。
   - 将 `self.plugmem_context` 拼接到 prompt 中。
   - 明确告诉 LLM：这部分是跨任务历史经验，只能作为参考。

3. `remember_current_task(...)`：
   - 将 `self.memory` 转换成 PlugMem trajectory steps。
   - 调用 PlugMem `/memories` 上传。

### 输出

新 agent 输出：

- 与原 agent 一样的 `(success, action)`。
- 额外产生 PlugMem 上传结果。
- 后续任务可以从 PlugMem graph 中召回这次任务经验。

## 6. 运行 HiAgent

### 输入

运行前需要确认：

- PlugMem 服务已经运行在 `http://localhost:8080`
- `hiagent-cross-task` graph 已经创建
- HiAgent 的 `.env` 或 shell 环境中有 `PROJECT_PATH`
- 目标配置文件存在，例如 `eval_configs/hiagent/blocksworld.yaml`
- 新 agent 已经注册到 `agentboard/agents/__init__.py`

### 处理

运行示例：

```powershell
python agentboard/eval_main.py `
  --cfg-path eval_configs/hiagent/blocksworld.yaml `
  --tasks pddl `
  --model gpt-4-turbo `
  --agent PlugMemContextEfficientAgent `
  --memory_size 100 `
  --max_num_steps 50 `
  --log_path ./logs/hiagent/plugmem_cross_task `
  --project_name none `
  --baseline_dir ./data/baseline_results
```

### 输出

HiAgent 输出：

- 每一步 action
- 每一步 observation
- 成功率、进度率、grounding accuracy 等评估结果
- 日志目录中的 trajectory 和 prompt 记录

PlugMem 输出：

- 新增 episodic memory
- 新增 semantic memory
- 新增 procedural memory
- 可在 `/inspector/` 查看 graph、节点和 session

## 7. 上传当前任务经验到 PlugMem

这里的“上传”指把 HiAgent 的任务执行轨迹写入 PlugMem 服务。

### 输入

HiAgent 原始 memory 结构通常类似：

```python
[
    [("Observation", init_obs)],
    [("Action", action_1), ("Observation", obs_1)],
    [("Action", action_2), ("Observation", obs_2)],
]
```

需要转换为 PlugMem trajectory：

```python
[
    {
        "observation": init_obs,
        "action": action_1,
    },
    {
        "observation": obs_1,
        "action": action_2,
    }
]
```

### 处理

在每个环境结束时调用：

```python
if hasattr(self.agent, "remember_current_task"):
    self.agent.remember_current_task(task_type=game_name)
```

PDDL 中适合放在 `evaluate_env()` 的成功返回前和失败返回前，确保无论任务成功或失败都能上传经验。

### 输出

PlugMem 会返回：

```json
{
  "status": "ok",
  "stats": {
    "episodic": 10,
    "semantic": 5,
    "procedural": 3
  }
}
```

这些数量代表当前 graph 中已持久化的记忆节点。

## 8. 召回跨任务记忆并注入 prompt

### 输入

下一次任务开始时，使用当前任务上下文作为查询：

```json
{
  "observation": "当前任务初始状态",
  "goal": "当前任务目标",
  "task_type": "blockworld"
}
```

### 处理

调用：

```text
POST /api/v1/graphs/hiagent-cross-task/reason
```

PlugMem 会：

1. 根据 observation/goal 选择记忆类型。
2. 从 Chroma 中检索相关节点。
3. 用 LLM 对检索结果做推理总结。
4. 返回可直接放入 prompt 的文本。

### 输出

返回值示例：

```text
Past relevant memory:
- Similar blockworld tasks often require clearing the target block first.
- Invalid move attempts should be followed by checking valid actions.
- When stacking blocks, preserve already satisfied tower order.
```

建议注入 prompt 时使用单独区域：

```text
Past task hints:
{plugmem_context}

These hints come from previous tasks and may help with the current task.
Use relevant strategies and patterns when choosing the next Subgoal or Action.
```

## 9. 多任务共享策略

最简单策略：

```text
所有任务写入 hiagent-cross-task
所有任务从 hiagent-cross-task 召回
```

输入输出关系：

```text
blockworld trajectory ─┐
gripper trajectory    ├─ upload ─▶ hiagent-cross-task ─▶ recall ─▶ tyreworld prompt
tyreworld trajectory  ┘
```

更精细策略：

```text
hiagent-cross-task       保存通用经验
hiagent-blockworld       保存 blockworld 私有经验
hiagent-gripper          保存 gripper 私有经验
hiagent-tyreworld        保存 tyreworld 私有经验
```

召回时同时查询：

```text
当前任务私有 graph + hiagent-cross-task
```

这样可以避免不同任务之间的 procedural memory 互相污染，同时保留通用语义经验。

## 10. 检查和验证

### 检查服务

```powershell
curl http://localhost:8080/api/v1/health
```

### 检查 graph

```powershell
curl http://localhost:8080/api/v1/graphs/hiagent-cross-task/stats `
  -H "X-API-Key: dev-key-change-me"
```

### 检查记忆节点

```powershell
curl "http://localhost:8080/api/v1/graphs/hiagent-cross-task/nodes?node_type=semantic&limit=10" `
  -H "X-API-Key: dev-key-change-me"
```

### 使用 Inspector

浏览器打开：

```text
http://localhost:8080/inspector/
```

可以查看：

- graph 拓扑
- semantic memory
- procedural memory
- episodic memory
- session timeline
- recall audit

## 11. 最终数据流总结

### 第一次任务

```text
输入:
  当前 task goal + init observation

处理:
  PlugMem graph 为空，recall 结果为空或很少
  HiAgent 正常执行任务
  任务结束后上传 trajectory

输出:
  HiAgent 评估日志
  PlugMem graph 中新增长期记忆
```

### 后续任务

```text
输入:
  新 task goal + init observation
  PlugMem graph 中已有历史任务记忆

处理:
  recall 相关历史经验
  将 recall 结果加入 prompt
  HiAgent 基于当前 observation + working memory + cross-task memory 生成 action
  任务结束后继续上传新 trajectory

输出:
  更丰富的 prompt
  新 action
  更新后的 PlugMem graph
```

## 12. 实施边界

已有代码已经提供：

- HiAgent agent registry
- `ContextEfficientAgentV2`
- 任务运行与日志记录
- PlugMem FastAPI 服务
- PlugMem graph、memory、retrieval API
- OpenClaw 插件中的 remember/recall 设计参考

仍需新增：

- HiAgent 侧 Python PlugMem client
- `PlugMemContextEfficientAgent`
- 任务结束时的 `remember_current_task()` 调用
- 配置项读取与错误降级逻辑

建议错误降级：

- PlugMem 服务不可用时，HiAgent 继续按原始 agent 运行。
- recall 失败时，不注入跨任务记忆。
- upload 失败时，只记录 warning，不中断评估。
