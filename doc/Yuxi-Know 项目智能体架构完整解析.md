# Yuxi-Know 项目智能体架构完整解析

## 目录

1. [项目架构概览](#一项目架构概览)
2. [Agent 实现分析](#二agent-实现分析)
3. [动态配置机制](#三动态配置机制)
4. [两层记忆系统](#四两层记忆系统)
5. [input_context 完全解析](#五input_context-完全解析)
6. [Attachments 附件功能](#六attachments-附件功能)
7. [Graph 执行方法参数详解](#七graph-执行方法参数详解)
8. [完整数据流图](#八完整数据流图)

---

## 一、项目架构概览

### 1.1 目录结构

```
Yuxi-Know/
├── src/
│   ├── agents/                      # 智能体实现
│   │   ├── common/                  # 公共组件
│   │   │   ├── base.py              # BaseAgent 抽象基类
│   │   │   ├── context.py           # BaseContext 配置基类
│   │   │   ├── models.py            # 模型加载工具
│   │   │   ├── middlewares/         # 中间件
│   │   │   │   ├── context_middlewares.py      # 上下文中间件
│   │   │   │   ├── attachment_middleware.py    # 附件中间件
│   │   │   │   └── dynamic_tool_middleware.py  # 动态工具中间件
│   │   │   └── subagents/           # 子智能体
│   │   │       └── calc_agent.py    # 计算子智能体
│   │   ├── chatbot/                 # 通用聊天智能体
│   │   ├── mini_agent/              # 简化演示智能体
│   │   └── reporter/                # SQL报表智能体
│   └── storage/                     # 数据存储层
│       ├── db/models.py             # SQLAlchemy 模型定义
│       └── conversation/manager.py  # 会话管理器
└── server/
    └── routers/chat_router.py       # API 路由层
```

### 1.2 技术栈

| 层次 | 技术 | 说明 |
|------|------|------|
| Agent 框架 | LangChain v1 + LangGraph | 智能体核心 |
| 短期记忆 | AsyncSqliteSaver | LangGraph Checkpointer |
| 长期存储 | SQLAlchemy + SQLite | 业务数据持久化 |
| API 框架 | FastAPI | RESTful API |
| 配置管理 | YAML + Dataclass | 分层配置系统 |

---

## 二、Agent 实现分析

### 2.1 ChatbotAgent - 通用聊天智能体

**源码位置**: `src/agents/chatbot/graph.py`

```python
class ChatbotAgent(BaseAgent):
    name = "智能体助手"
    description = "基础的对话机器人，可以回答问题，默认不使用任何工具，可在配置中启用需要的工具。"
    capabilities = ["file_upload"]  # 支持文件上传功能

    async def get_graph(self, **kwargs):
        # 创建动态工具中间件实例
        dynamic_tool_middleware = DynamicToolMiddleware(
            base_tools=self.get_tools(), 
            mcp_servers=list(MCP_SERVERS.keys())
        )
        await dynamic_tool_middleware.initialize_mcp_tools()

        graph = create_agent(
            model=load_chat_model("siliconflow/Qwen/Qwen3-235B-A22B-Instruct-2507"),
            tools=get_tools(),
            middleware=[
                context_aware_prompt,       # ① 动态系统提示词
                inject_attachment_context,  # ② 附件上下文注入
                context_based_model,        # ③ 动态模型选择
                dynamic_tool_middleware,    # ④ 动态工具选择
            ],
            checkpointer=await self._get_checkpointer(),
        )
        return graph
```

**核心特点**：
- **完整中间件链**：4个中间件协同工作
- **MCP 支持**：预加载所有 MCP 服务器工具
- **文件上传**：通过 `capabilities = ["file_upload"]` 声明能力
- **动态配置**：模型、提示词、工具均可运行时动态选择

---

### 2.2 MiniAgent - 简化演示智能体

**源码位置**: `src/agents/mini_agent/graph.py`

```python
class MiniAgent(BaseAgent):
    name = "智能体 Demo"
    description = "一个基于内置工具的智能体示例"

    async def get_graph(self, **kwargs):
        graph = create_agent(
            model=load_chat_model(config.default_model),
            tools=self.get_tools(),  # 仅使用内置工具
            middleware=[context_aware_prompt, context_based_model],  # 仅2个中间件
            checkpointer=await self._get_checkpointer(),
        )
        return graph
```

**核心特点**：
- **简化配置**：仅使用 2 个核心中间件
- **内置工具**：不支持 MCP 和动态工具选择
- **演示用途**：适合学习和测试

---

### 2.3 SqlReporterAgent - SQL报表智能体

**源码位置**: `src/agents/reporter/graph.py`

```python
_mcp_servers = {
    "mcp-server-chart": {
        "command": "npx", 
        "args": ["-y", "@antv/mcp-server-chart"], 
        "transport": "stdio"
    }
}

class SqlReporterAgent(BaseAgent):
    name = "数据库报表助手"
    description = "一个能够生成 SQL 查询报告的智能体助手。同时调用 Charts MCP 生成图表。"

    async def get_tools(self):
        chart_tools = await get_mcp_tools("mcp-server-chart", additional_servers=_mcp_servers)
        mysql_tools = get_mysql_tools()
        return chart_tools + mysql_tools
```

**核心特点**：
- **专业领域**：专注于 SQL 查询和数据可视化
- **MCP 图表**：集成 @antv/mcp-server-chart 生成图表
- **MySQL 工具**：内置 MySQL 查询工具集

---

### 2.4 calc_agent - 计算子智能体

**源码位置**: `src/agents/common/subagents/calc_agent.py`

```python
# 创建专用计算 Agent
calc_agent = create_agent(
    model=load_chat_model(config.default_model),
    tools=[calculator],
    system_prompt="你可以使用计算器工具，处理各种数学计算任务。",
)

# 封装为工具供其他 Agent 调用
@tool(name_or_callable="calc_agent_tool", description="进行计算任务")
async def calc_agent_tool(description: str) -> str:
    """CalcAgent 工具 - 使用子智能体 CalcAgent 进行计算任务"""
    response = await calc_agent.ainvoke({"messages": [("user", description)]})
    return response["messages"][-1].content
```

**核心特点**：
- **子智能体模式**：Agent 作为 Tool 被其他 Agent 调用
- **职责分离**：专注于数学计算任务
- **无状态**：每次调用独立，不保存历史

---

## 三、动态配置机制

### 3.1 配置优先级

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              配置优先级（高 → 低）                                    │
│                                                                                     │
│   ① 运行时配置 (input_context)                                                      │
│      ↓ 覆盖                                                                         │
│   ② 文件配置 (config.yaml)                                                          │
│      ↓ 覆盖                                                                         │
│   ③ 类默认配置 (BaseContext 默认值)                                                  │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 BaseContext 定义

**源码位置**: `src/agents/common/context.py`

```python
@dataclass(kw_only=True)
class BaseContext:
    """配置优先级: 运行时配置 > 文件配置 > 类默认配置"""

    thread_id: str = field(
        default_factory=lambda: str(uuid.uuid4()),
        metadata={"name": "线程ID", "configurable": False}
    )

    user_id: str = field(
        default_factory=lambda: str(uuid.uuid4()),
        metadata={"name": "用户ID", "configurable": False}
    )

    system_prompt: str = field(
        default="You are a helpful assistant.",
        metadata={"name": "系统提示词", "description": "用来描述智能体的角色和行为"}
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default=sys_config.default_model,
        metadata={"name": "智能体模型", "description": "智能体的驱动模型"}
    )

    @classmethod
    def from_file(cls, module_name: str, input_context: dict = None) -> "BaseContext":
        """从文件加载配置，并用 input_context 覆盖"""
        context = cls()
        
        # 1. 从文件加载
        config_file_path = Path(save_dir) / "agents" / module_name / "config.yaml"
        if os.path.exists(config_file_path):
            file_config = yaml.safe_load(open(config_file_path))
            context.update(file_config)
        
        # 2. 用运行时配置覆盖
        if input_context:
            context.update(input_context)
        
        return context
```

### 3.3 ChatbotAgent 扩展 Context

**源码位置**: `src/agents/chatbot/context.py`

```python
@dataclass(kw_only=True)
class Context(BaseContext):
    # 可选择的工具列表
    tools: Annotated[list[dict], {"__template_metadata__": {"kind": "tools"}}] = field(
        default_factory=list,
        metadata={
            "name": "工具",
            "options": gen_tool_info(get_tools()),
            "description": "工具列表",
        },
    )

    # 可选择的 MCP 服务器列表
    mcps: list[str] = field(
        default_factory=list,
        metadata={
            "name": "MCP服务器", 
            "options": list(MCP_SERVERS.keys()), 
            "description": "MCP服务器列表"
        },
    )
```

### 3.4 动态提示词中间件

**源码位置**: `src/agents/common/middlewares/context_middlewares.py`

```python
@dynamic_prompt
def context_aware_prompt(request: ModelRequest) -> str:
    """从 runtime.context 动态生成系统提示词"""
    return request.runtime.context.system_prompt
```

**工作原理**：
1. 用户在 UI 配置或 API 传入 `system_prompt`
2. `BaseContext.from_file()` 合并配置
3. Context 实例传入 `graph.astream(..., context=context)`
4. 中间件通过 `request.runtime.context.system_prompt` 读取

### 3.5 动态模型选择中间件

```python
@wrap_model_call
async def context_based_model(request: ModelRequest, handler) -> ModelResponse:
    """从 runtime.context 动态选择模型"""
    model_spec = request.runtime.context.model  # 如 "openai/gpt-4o"
    model = load_chat_model(model_spec)
    
    request = request.override(model=model)
    return await handler(request)
```

**工作原理**：
1. 用户传入 `model: "deepseek/deepseek-chat"` 
2. `load_chat_model()` 解析 provider/model 格式
3. 动态创建对应的 ChatModel 实例
4. 覆盖 create_agent 时指定的默认模型

### 3.6 动态工具选择中间件

**源码位置**: `src/agents/common/middlewares/dynamic_tool_middleware.py`

```python
class DynamicToolMiddleware(AgentMiddleware):
    """动态工具选择中间件 - 支持 MCP 工具的动态加载和注册"""

    def __init__(self, base_tools: list, mcp_servers: list[str] | None = None):
        self.tools: list = base_tools
        self._all_mcp_tools: dict[str, list] = {}  # 已加载的 MCP 工具
        self._mcp_servers = mcp_servers or []

    async def initialize_mcp_tools(self) -> None:
        """预加载所有可能用到的 MCP 工具"""
        for mcp_name in self._mcp_servers:
            mcp_tools = await get_mcp_tools(mcp_name)
            self._all_mcp_tools[mcp_name] = mcp_tools
            self.tools.extend(mcp_tools)  # 注册到 middleware.tools

    async def awrap_model_call(self, request, handler) -> ModelResponse:
        """根据配置从已注册的工具中筛选"""
        selected_tools = request.runtime.context.tools
        selected_mcps = request.runtime.context.mcps

        enabled_tools = []
        
        # 筛选基础工具
        if selected_tools:
            enabled_tools = [t for t in self.tools if t.name in selected_tools]
        
        # 筛选 MCP 工具
        if selected_mcps:
            for mcp in selected_mcps:
                if mcp in self._all_mcp_tools:
                    enabled_tools.extend(self._all_mcp_tools[mcp])

        request = request.override(tools=enabled_tools)
        return await handler(request)
```

**关键设计**：
- **预加载**：所有 MCP 工具在 Agent 初始化时加载
- **运行时筛选**：根据 Context 配置筛选可用工具
- **性能优化**：避免每次请求都重新加载 MCP 工具

---

## 四、两层记忆系统

### 4.1 架构图

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              Yuxi-Know 两层记忆系统                                  │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                          第一层：LangGraph Checkpointer                      │   │
│  │                                                                             │   │
│  │   位置：{workdir}/agents/{module_name}/aio_history.db                        │   │
│  │   类型：AsyncSqliteSaver                                                    │   │
│  │   作用：保存 Agent 内部状态（messages 列表）                                   │   │
│  │   标识：thread_id                                                           │   │
│  │   生命周期：跨多轮对话保持                                                     │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                              │
│                                      │ 同步                                         │
│                                      ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                          第二层：ConversationManager                         │   │
│  │                                                                             │   │
│  │   位置：主数据库 (SQLite/PostgreSQL)                                          │   │
│  │   模型：Conversation, Message, ToolCall, ConversationStats                   │   │
│  │   作用：业务级数据持久化（历史查询、统计、导出）                                  │   │
│  │   标识：thread_id + user_id + agent_id                                       │   │
│  │   附加：attachments 存储在 Conversation.extra_metadata                       │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 第一层：LangGraph Checkpointer

**源码位置**: `src/agents/common/base.py`

```python
class BaseAgent:
    async def _get_checkpointer(self):
        try:
            checkpointer = AsyncSqliteSaver(await self.get_async_conn())
        except Exception as e:
            logger.error(f"Checkpointer 初始化失败: {e}, 使用内存存储")
            checkpointer = InMemorySaver()
        return checkpointer

    async def get_async_conn(self) -> aiosqlite.Connection:
        """连接到 Agent 专属的 SQLite 数据库"""
        return await aiosqlite.connect(os.path.join(self.workdir, "aio_history.db"))

    async def get_history(self, user_id, thread_id) -> list[dict]:
        """从 Checkpointer 获取历史消息"""
        config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
        state = await app.aget_state(config)
        if state:
            return [msg.model_dump() for msg in state.values.get("messages", [])]
        return []
```

**存储内容**：
- `messages`: 完整的消息列表（HumanMessage, AIMessage, ToolMessage）
- `jump_to`: 流程跳转标记
- `structured_response`: 结构化响应
- 中间件扩展的自定义字段（如 `attachments`）

### 4.3 第二层：ConversationManager

**源码位置**: `src/storage/conversation/manager.py`

```python
class ConversationManager:
    """业务级会话数据管理"""

    async def create_conversation(self, user_id, agent_id, title, thread_id, metadata):
        """创建新会话"""
        conversation = Conversation(
            thread_id=thread_id or str(uuid.uuid4()),
            user_id=str(user_id),
            agent_id=agent_id,
            title=title or "New Conversation",
            extra_metadata=metadata,  # 包含 attachments
        )
        self.db.add(conversation)
        await self.db.commit()
        return conversation

    async def add_message_by_thread_id(self, thread_id, role, content, ...):
        """添加消息到会话"""
        ...

    async def get_attachments_by_thread_id(self, thread_id) -> list[dict]:
        """获取会话的附件列表"""
        conversation = await self.get_conversation_by_thread_id(thread_id)
        if not conversation:
            return []
        return list(conversation.extra_metadata.get("attachments", []))
```

**数据模型**：

```python
# src/storage/db/models.py

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True)
    thread_id = Column(String(64), unique=True, index=True)  # ★ 关联 Checkpointer
    user_id = Column(String(64), index=True)
    agent_id = Column(String(64), index=True)
    title = Column(String(255))
    status = Column(String(20), default="active")
    extra_metadata = Column(JSON)  # ★ 存储 attachments
    messages = relationship("Message", back_populates="conversation")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    role = Column(String(20))  # user/assistant/system/tool
    content = Column(Text)
    message_type = Column(String(30))  # text/tool_call/tool_result
    extra_metadata = Column(JSON)  # 完整消息 dump
    tool_calls = relationship("ToolCall", back_populates="message")

class ToolCall(Base):
    __tablename__ = "tool_calls"
    id = Column(Integer, primary_key=True)
    message_id = Column(Integer, ForeignKey("messages.id"))
    langgraph_tool_call_id = Column(String(100))  # ★ LangGraph tool_call_id
    tool_name = Column(String(100))
    tool_input = Column(JSON)
    tool_output = Column(Text)
    status = Column(String(20))  # pending/success/error
```

### 4.4 两层记忆同步流程

**源码位置**: `server/routers/chat_router.py`

```python
async def stream_messages():
    conv_manager = ConversationManager(db)
    
    # 1. 保存用户消息到第二层（立即写入）
    await conv_manager.add_message_by_thread_id(
        thread_id=thread_id,
        role="user",
        content=query,
    )
    
    # 2. 从第二层读取 attachments
    attachments = await conv_manager.get_attachments_by_thread_id(thread_id)
    input_context["attachments"] = attachments
    
    # 3. 调用 Agent（第一层自动保存状态）
    async for msg, metadata in agent.stream_messages(messages, input_context=input_context):
        yield msg
    
    # 4. 流式完成后，从第一层同步到第二层
    await save_messages_from_langgraph_state(
        agent_instance=agent,
        thread_id=thread_id,
        conv_mgr=conv_manager,
        config_dict=langgraph_config,
    )
```

**同步函数**：

```python
async def save_messages_from_langgraph_state(agent_instance, thread_id, conv_mgr, config_dict):
    """从 LangGraph state 同步消息到第二层"""
    # 1. 获取 LangGraph 中的所有消息
    messages = await _get_langgraph_messages(agent_instance, config_dict)
    
    # 2. 获取已存在的消息 ID（去重）
    existing_ids = await _get_existing_message_ids(conv_mgr, thread_id)
    
    # 3. 遍历并保存新消息
    for msg in messages:
        msg_dict = msg.model_dump()
        msg_type = msg_dict.get("type")
        
        if msg_type == "human" or msg.id in existing_ids:
            continue  # 跳过用户消息（已保存）和已存在的消息
        
        if msg_type == "ai":
            await _save_ai_message(conv_mgr, thread_id, msg_dict)
        elif msg_type == "tool":
            await _save_tool_message(conv_mgr, msg_dict)
```

### 4.5 交互时序图

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              两层记忆交互时序                                         │
└─────────────────────────────────────────────────────────────────────────────────────┘

    Client           API Router          Layer 2              Layer 1           LLM
      │                  │             (ConvManager)       (Checkpointer)         │
      │                  │                  │                   │                 │
      │ ─── POST /chat ─►│                  │                   │                 │
      │                  │                  │                   │                 │
      │                  │ ─── 写入用户消息 ─►│                   │                 │
      │                  │                  │                   │                 │
      │                  │ ◄─── 读取 attachments ───            │                 │
      │                  │                  │                   │                 │
      │                  │                  │ ◄─── 恢复历史状态 ───│                 │
      │                  │                  │                   │                 │
      │                  │ ───────────────────────── 发送消息 ────────────────────►│
      │                  │                  │                   │                 │
      │                  │ ◄──────────────────────── 流式响应 ◄────────────────────│
      │ ◄── SSE 流式 ────│                  │                   │                 │
      │                  │                  │                   │                 │
      │                  │                  │ ◄─── 自动保存状态 ───│                 │
      │                  │                  │                   │                 │
      │                  │ ─── 同步到 Layer 2 ─►│                │                 │
      │                  │                  │                   │                 │
      │ ◄── 完成响应 ────│                  │                   │                 │
      │                  │                  │                   │                 │
```

---

## 五、input_context 完全解析

### 5.1 input_context 的构成

```python
# server/routers/chat_router.py 中的构造
input_context = {
    "user_id": str(current_user.id),      # 用户标识
    "thread_id": thread_id,                # 对话线程标识
    "attachments": attachments,            # 附件列表（从 Layer 2 读取）
    # 可选：UI 配置项
    "system_prompt": "自定义提示词",
    "model": "openai/gpt-4o",
    "tools": ["search", "calculator"],
    "mcps": ["mcp-server-chart"],
}
```

### 5.2 input_context 的流向

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              input_context 数据流                                    │
└─────────────────────────────────────────────────────────────────────────────────────┘

                              input_context
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            BaseAgent.stream_messages()                              │
│                                                                                     │
│   # 1. 创建 Context 实例（合并配置）                                                 │
│   context = self.context_schema.from_file(                                         │
│       module_name=self.module_name,                                                │
│       input_context=input_context  ◄─── 运行时覆盖文件配置                           │
│   )                                                                                │
│                                                                                     │
│   # 2. 提取 attachments 作为 State 输入                                             │
│   attachments = input_context.get("attachments", [])                               │
│                                                                                     │
│   # 3. 构造 config（包含 configurable）                                             │
│   input_config = {"configurable": input_context, "recursion_limit": 100}           │
│                                                                                     │
│   # 4. 调用 graph.astream                                                          │
│   async for msg, metadata in graph.astream(                                        │
│       {"messages": messages, "attachments": attachments},  ◄─── 传入 State         │
│       stream_mode="messages",                                                      │
│       context=context,                    ◄─── 传入 runtime.context                │
│       config=input_config,                ◄─── 传入 config.configurable            │
│   ):                                                                               │
│       yield msg, metadata                                                          │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
    ┌──────────┐             ┌──────────────┐           ┌─────────────┐
    │  State   │             │   Runtime    │           │   Config    │
    │          │             │              │           │             │
    │ messages │             │ context:     │           │ configurable│
    │ attach-  │             │   system_    │           │   thread_id │
    │  ments   │             │   prompt     │           │   user_id   │
    │          │             │   model      │           │             │
    │          │             │   tools      │           │             │
    │          │             │   mcps       │           │             │
    └──────────┘             └──────────────┘           └─────────────┘
         │                          │                         │
         ▼                          ▼                         ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                        中间件访问方式                            │
    │                                                                 │
    │  request.state["attachments"]     ◄─── AttachmentMiddleware    │
    │  request.runtime.context.model    ◄─── context_based_model     │
    │  request.runtime.context.system_prompt ◄─ context_aware_prompt │
    │  request.runtime.context.tools    ◄─── DynamicToolMiddleware   │
    │  config["configurable"]["thread_id"] ◄─── Checkpointer         │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
```

### 5.3 三个目标的区别

| 目标 | 数据 | 持久化 | 访问方式 |
|------|------|--------|---------|
| **State** | messages, attachments | ✅ Checkpointer 保存 | `request.state` |
| **Runtime.context** | system_prompt, model, tools, mcps | ❌ 每次传入 | `request.runtime.context` |
| **Config.configurable** | thread_id, user_id | ❌ 元数据 | Checkpointer 内部使用 |

---

## 六、Attachments 附件功能

### 6.1 实现原理

Yuxi-Know 的 attachments 功能是基于 LangChain 中间件机制的**自定义扩展**，而非框架原生支持。

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              Attachments 实现架构                                    │
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │ 1. 扩展 State Schema                                                        │   │
│   │                                                                             │   │
│   │    class AttachmentState(AgentState):                                       │   │
│   │        attachments: NotRequired[list[dict]]   ◄─── 新增字段                  │   │
│   │                                                                             │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                          │
│                                          ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │ 2. 中间件声明 state_schema                                                   │   │
│   │                                                                             │   │
│   │    class AttachmentMiddleware(AgentMiddleware[AttachmentState]):            │   │
│   │        state_schema = AttachmentState   ◄─── 告诉框架合并 schema             │   │
│   │                                                                             │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                          │
│                                          ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │ 3. create_agent 自动合并 schema                                              │   │
│   │                                                                             │   │
│   │    # langchain_v1/langchain/agents/factory.py                               │   │
│   │    final_schema = _resolve_schema([                                         │   │
│   │        AgentState,                     # 基础 schema                         │   │
│   │        middleware.state_schema,        # 中间件扩展                          │   │
│   │        user_state_schema,              # 用户自定义                          │   │
│   │    ])                                                                       │   │
│   │                                                                             │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 AttachmentMiddleware 实现

**源码位置**: `src/agents/common/middlewares/attachment_middleware.py`

```python
class AttachmentState(AgentState):
    """扩展 AgentState 以支持附件"""
    attachments: NotRequired[list[dict]]


def _build_attachment_prompt(attachments: Sequence[dict]) -> str | None:
    """将附件渲染为系统提示"""
    if not attachments:
        return None

    chunks = []
    for idx, attachment in enumerate(attachments, 1):
        if attachment.get("status") != "parsed":
            continue
        markdown = attachment.get("markdown")
        if not markdown:
            continue
        
        file_name = attachment.get("file_name") or f"附件 {idx}"
        truncated = "（已截断）" if attachment.get("truncated") else ""
        header = f"### 附件 {idx}: {file_name}{truncated}"
        chunks.append(f"{header}\n\n{markdown}".strip())

    if not chunks:
        return None

    instructions = (
        "以下为用户提供的附件内容，请综合这些文件与用户的新问题进行回答。"
        "如附件与问题无关，可忽略附件内容：\n\n"
    )
    return instructions + "\n\n".join(chunks)


class AttachmentMiddleware(AgentMiddleware[AttachmentState]):
    """从 State 中读取附件并注入到消息中"""

    state_schema = AttachmentState

    async def awrap_model_call(self, request, handler):
        attachments = request.state.get("attachments", [])

        if attachments:
            attachment_prompt = _build_attachment_prompt(attachments)
            if attachment_prompt:
                # 注入为 SystemMessage（不修改 state，仅影响本次调用）
                messages = [
                    {"role": "system", "content": attachment_prompt},
                    *request.messages,
                ]
                request = request.override(messages=messages)

        return await handler(request)


# 导出中间件实例
inject_attachment_context = AttachmentMiddleware()
```

### 6.3 附件数据结构

```python
attachment = {
    "file_id": "abc123",              # 文件唯一标识
    "file_name": "report.pdf",        # 文件名
    "status": "parsed",               # 状态：uploading/parsing/parsed/error
    "markdown": "# 报告内容\n...",    # 解析后的 Markdown 内容
    "truncated": False,               # 是否被截断
    "error": None,                    # 错误信息（如果有）
}
```

### 6.4 附件存储位置

附件元数据存储在 **Conversation.extra_metadata** 中：

```python
# ConversationManager 中的操作
async def add_attachment(self, conversation_id, attachment_info):
    conversation = await self._get_conversation_by_id(conversation_id)
    metadata = dict(conversation.extra_metadata or {})
    attachments = list(metadata.get("attachments", []))
    attachments.append(attachment_info)
    metadata["attachments"] = attachments
    await self._save_metadata(conversation, metadata)
```

---

## 七、Graph 执行方法参数详解

### 7.1 方法签名

`create_agent` 返回的 `CompiledStateGraph` 提供四种执行方法：

| 方法 | 同步/异步 | 返回类型 | 典型场景 |
|------|---------|---------|---------|
| `invoke()` | 同步 | `dict` (最终结果) | 简单脚本、测试 |
| `stream()` | 同步 | `Iterator` | 同步流式处理 |
| `ainvoke()` | 异步 | `dict` (最终结果) | 异步 API 服务 |
| `astream()` | 异步 | `AsyncIterator` | **聊天应用推荐** |

### 7.2 完整参数列表

```python
async def astream(
    self,
    input: InputT | Command | None,           # ① 状态输入
    config: RunnableConfig | None = None,      # ② 运行配置
    *,
    context: ContextT | None = None,           # ③ 运行时上下文
    stream_mode: StreamMode | Sequence[StreamMode] | None = None,
    print_mode: StreamMode | Sequence[StreamMode] = (),
    output_keys: str | Sequence[str] | None = None,
    interrupt_before: All | Sequence[str] | None = None,
    interrupt_after: All | Sequence[str] | None = None,
    durability: Durability | None = None,
    subgraphs: bool = False,
    debug: bool | None = None,
) -> AsyncIterator[dict[str, Any] | Any]:
```

### 7.3 核心参数详解

#### ① input - 状态输入

```python
# 基础用法
input = {
    "messages": [HumanMessage(content="你好")]
}

# 扩展用法（包含 attachments）
input = {
    "messages": [HumanMessage(content="请分析这份文件")],
    "attachments": [
        {"file_id": "001", "file_name": "report.pdf", "status": "parsed", "markdown": "..."}
    ]
}
```

#### ② config - 运行配置

```python
config = {
    "configurable": {
        "thread_id": "conversation-123",  # ★ Checkpointer 必需
        "user_id": "user-456",
    },
    "recursion_limit": 100,  # 防止无限循环
    "tags": ["production"],
    "metadata": {"request_id": "req-789"},
}
```

#### ③ context - 运行时上下文

```python
@dataclass
class MyContext:
    user_id: str
    model: str = "openai/gpt-4o"
    system_prompt: str = "你是一个助手"
    tools: list = field(default_factory=list)
    mcps: list = field(default_factory=list)

context = MyContext(
    user_id="123",
    model="deepseek/deepseek-chat",
    system_prompt="你是专业的数据分析师",
)
```

#### ④ stream_mode - 流式模式

| 模式 | 输出内容 | 典型用途 |
|------|---------|---------|
| `"values"` | 每步后的完整 State | 获取最终结果 |
| `"updates"` | 节点名 + 该节点的更新 | 调试 |
| `"messages"` | LLM token + metadata (2-tuple) | **流式聊天** |
| `"custom"` | StreamWriter 自定义输出 | 自定义数据流 |
| `"debug"` | 详细调试信息 | 开发调试 |

### 7.4 Yuxi-Know 中的实际调用

```python
# src/agents/common/base.py
async def stream_messages(self, messages, input_context=None, **kwargs):
    graph = await self.get_graph()
    context = self.context_schema.from_file(
        module_name=self.module_name, 
        input_context=input_context
    )
    
    attachments = (input_context or {}).get("attachments", [])
    input_config = {"configurable": input_context, "recursion_limit": 100}

    async for msg, metadata in graph.astream(
        {"messages": messages, "attachments": attachments},  # input
        stream_mode="messages",                               # stream_mode
        context=context,                                      # context
        config=input_config,                                  # config
    ):
        yield msg, metadata
```

---

## 八、完整数据流图

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           Yuxi-Know 完整数据流                                        │
└─────────────────────────────────────────────────────────────────────────────────────┘

                                  用户请求
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              API Router (chat_router.py)                             │
│                                                                                     │
│   1. 接收请求参数: query, config, image_content                                       │
│   2. 构建 input_context = {user_id, thread_id, attachments, ...}                    │
│   3. 写入用户消息到 Layer 2 (ConversationManager)                                    │
│   4. 从 Layer 2 读取 attachments                                                     │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              BaseAgent.stream_messages()                            │
│                                                                                     │
│   1. context = Context.from_file(input_context)   ◄─── 合并配置                     │
│   2. attachments = input_context.get("attachments")                                 │
│   3. input_config = {"configurable": input_context}                                 │
│   4. graph.astream({messages, attachments}, context, config)                        │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              LangGraph 执行引擎                                       │
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                         Checkpointer (Layer 1)                              │   │
│   │                                                                             │   │
│   │   • 根据 thread_id 恢复历史 State                                            │   │
│   │   • 合并新的 input 到 State                                                  │   │
│   │   • 执行完成后自动保存 State                                                  │   │
│   │                                                                             │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                              │
│                                      ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                           中间件链处理                                        │   │
│   │                                                                             │   │
│   │   ① context_aware_prompt                                                    │   │
│   │      └─► request.runtime.context.system_prompt ─► 动态系统提示词             │   │
│   │                                                                             │   │
│   │   ② inject_attachment_context                                               │   │
│   │      └─► request.state["attachments"] ─► 注入附件内容到消息                   │   │
│   │                                                                             │   │
│   │   ③ context_based_model                                                     │   │
│   │      └─► request.runtime.context.model ─► 动态选择模型                       │   │
│   │                                                                             │   │
│   │   ④ dynamic_tool_middleware                                                 │   │
│   │      └─► request.runtime.context.tools/mcps ─► 动态选择工具                  │   │
│   │                                                                             │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                              │
│                                      ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                           Agent 执行循环                                      │   │
│   │                                                                             │   │
│   │      ┌──────────┐        ┌──────────┐        ┌──────────┐                   │   │
│   │      │  Model   │───────►│  Tools   │───────►│  Model   │──► ...           │   │
│   │      │  Node    │        │  Node    │        │  Node    │                   │   │
│   │      └──────────┘        └──────────┘        └──────────┘                   │   │
│   │           │                   │                   │                         │   │
│   │           ▼                   ▼                   ▼                         │   │
│   │      ┌─────────────────────────────────────────────────────────────────┐   │   │
│   │      │                      🤖 LLM                                      │   │   │
│   │      │          (由 context.model 动态决定: deepseek/gpt-4o/...)        │   │   │
│   │      └─────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                             │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              流式输出 + 同步到 Layer 2                               │
│                                                                                     │
│   stream_mode="messages":                                                           │
│       (AIMessageChunk("你好"), {"langgraph_node": "model"})                         │
│       (AIMessageChunk("！"), {"langgraph_node": "model"})                           │
│       ...                                                                           │
│                                                                                     │
│   流式完成后:                                                                        │
│       save_messages_from_langgraph_state() ─► 同步到 ConversationManager            │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                                  用户响应
```

---

## 九、总结

### 核心设计亮点

1. **分层记忆架构**：LangGraph Checkpointer（短期）+ ConversationManager（长期），通过 `thread_id` 关联

2. **灵活的配置系统**：三级配置优先级（运行时 > 文件 > 默认），支持动态切换模型、提示词、工具

3. **中间件扩展机制**：通过 `state_schema` 扩展 State，通过 `awrap_model_call` 拦截请求

4. **附件注入**：非侵入式设计，通过中间件将附件内容注入为 SystemMessage

5. **子智能体模式**：Agent 作为 Tool，实现职责分离和复用

### 关键数据结构

| 结构 | 作用 | 持久化位置 |
|------|------|-----------|
| `input_context` | 运行时统一配置入口 | - |
| `Context` | 中间件读取的配置 | config.yaml |
| `AgentState` | Graph 内部状态 | Checkpointer |
| `Conversation` | 业务级会话元数据 | 主数据库 |
| `Message` | 消息记录 | 主数据库 |
| `attachments` | 文件附件元数据 | Conversation.extra_metadata |

