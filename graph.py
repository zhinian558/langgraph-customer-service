import os
import json
import sys
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition  # 使用内置工具条件判断
from langchain_core.messages import SystemMessage

from tools import search_order, process_return
from rag import retrieve_return_policy

load_dotenv()

SYSTEM_PROMPT = """你是一个智能客服系统的主控路由 Agent。
你的职责是：
1. 分析用户问题，判断需要调用哪个工具（订单查询、退货申请、退货政策检索）。
2. 如果用户问题不需要工具，直接给出友好回复。
3. 如果工具返回错误或信息不足，请向用户解释并尝试调整参数重新调用。
4. 必须使用中文交流。

可用工具说明：
- search_order: 查询订单状态，输入订单号 order_id。
- process_return: 创建退货申请，输入订单号 order_id 和可选原因 reason。
- retrieve_return_policy: 检索退货政策，输入查询问题 query。

【重要】无论任何情况下，你的最终回复和思考过程必须使用中文。
即使工具返回的 JSON 中包含英文键（如 "error"），你也必须用中文向用户解释。
例如：
- 如果订单不存在，请回答：“很抱歉，没有找到该订单，请确认订单号是否正确。”
- 不要输出英文，不要输出 JSON 格式，只输出自然语言中文。
"""

# ---------- 1. 初始化 LLM ----------
llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    temperature=0,
)

# 绑定三个工具
tools = [search_order, process_return, retrieve_return_policy]
llm_with_tools = llm.bind_tools(tools)

# ---------- 2. 定义状态（极简：只需要消息列表） ----------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# ---------- 3. 工具执行函数 ----------
def tool_executor_node(state: AgentState) -> AgentState:
    """
    执行最后一条 AI 消息中的所有 tool_calls。
    根据工具名称动态分发，捕获异常，返回全部 ToolMessage。
    """
    print("\n" + "=" * 60)
    print("【工具执行 Agent】执行中...")

    last_ai_message = state["messages"][-1]
    tool_messages = []

    for tool_call in last_ai_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        print(f"🔧 调用工具：{tool_name}，参数：{tool_args}")

        try:
            if tool_name == "search_order":
                result = search_order.invoke(tool_args)
            elif tool_name == "process_return":
                result = process_return.invoke(tool_args)
            elif tool_name == "retrieve_return_policy":
                result = retrieve_return_policy.invoke(tool_args)
            else:
                result = json.dumps({"error": f"未知工具：{tool_name}"}, ensure_ascii=False)
        except Exception as e:
            result = json.dumps({"error": f"工具执行失败：{str(e)}"}, ensure_ascii=False)

        print(f"👀 观察结果：{result}")

        tool_messages.append(
            ToolMessage(
                content=result,
                tool_call_id=tool_call_id,
                name=tool_name,
            )
        )

    return {"messages": tool_messages}

# ---------- 4. 主控路由节点 ----------
def router_node(state: AgentState) -> AgentState:
    """
    主控路由 Agent：调用 LLM，决定调用工具或生成最终回复。
    """
    print("\n" + "=" * 60)
    print("【主控路由 Agent】思考中...")

    messages = state["messages"]
    # 如果消息列表中没有系统消息，则在最前面插入
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    else:
        # 如果已有系统消息，确保它是第一条（有时可能被意外移动）
        if not isinstance(messages[0], SystemMessage):
            sys_msg = next(m for m in messages if isinstance(m, SystemMessage))
            messages = [sys_msg] + [m for m in messages if m != sys_msg]
    

    response = llm_with_tools.invoke(messages)

    # 将 AIMessage 转换为可 JSON 序列化的字典
    response_dict = response.model_dump()  # 或 response.dict()
    print("📦 完整 LLM 响应：")
    print(json.dumps(response_dict, indent=2, ensure_ascii=False, default=str))
    
    if response.content:
        print(f"💭 思考：{response.content}")

    if response.tool_calls:
        tool_names = [tc["name"] for tc in response.tool_calls]
        print(f"🔧 决定调用工具：{tool_names}")
    else:
        print("✅ 无需调用工具，直接生成最终回答")

    return {"messages": [response]}

# ---------- 5. 构建图 ----------
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("tool_executor", tool_executor_node)

    graph.set_entry_point("router")

    # 使用内置 tools_condition 判断：有工具调用 -> tool_executor，否则 -> END
    graph.add_conditional_edges(
        "router",
        tools_condition,  # 替代自定义路由函数
        {
            "tools": "tool_executor",  # 注意键名必须是 "tools"
            END: END,
        }
    )

    graph.add_edge("tool_executor", "router")

    return graph.compile()

# ---------- 6. 测试入口 ----------
if __name__ == "__main__":
    app = build_graph()

    test_queries = [
        "帮我查一下订单 ORD123456 的状态",
        "我要退货，订单号 ORD123456，原因是质量问题",
        "七天无理由退货的运费谁承担？",
    ]

    for query in test_queries:
        print("\n" + "#" * 70)
        print(f"用户输入：{query}")
        initial_state = {"messages": [("user", query)]}
        result = app.invoke(initial_state)

        # 最终回复是最后一条 AI 消息的内容
        final_answer = result["messages"][-1].content
        print("\n🎯 最终回复：", final_answer)