import os
import json
from typing import Annotated, TypedDict, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition

import database
from tools import search_order, process_return, check_return_eligibility  # 导入新工具
from rag import retrieve_return_policy

load_dotenv()
database.init_db()

# ---------- 初始化 LLM ----------
llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    temperature=0,
)

# ---------- 定义路由工具 ----------
@tool
def route_to_agent(agent_name: str) -> str:
    """
    决定下一个执行的 Agent。
    可选值：
    - "order_agent"      用于查询订单信息
    - "policy_agent"     用于检索退货政策
    - "return_agent"     用于执行退货流程（包括资格校验和创建退货单）
    - "final_response"   所有信息已收集完毕，准备生成最终回复
    """
    return f"下一步将调用 {agent_name}"

supervisor_llm = llm.bind_tools([route_to_agent],tool_choice={"type": "function", "function": {"name": "route_to_agent"}})

# ---------- 绑定各自工具的专职 Agent LLM ----------
order_llm = llm.bind_tools([search_order],tool_choice={"type": "function", "function": {"name": "search_order"}})
policy_llm = llm.bind_tools([retrieve_return_policy],tool_choice={"type": "function", "function": {"name": "retrieve_return_policy"}})
# 退货 Agent 绑定两个工具：第一轮强制调用资格检查，后续轮次由模型根据检查结果自由决策
return_first_llm = llm.bind_tools([check_return_eligibility], tool_choice={"type": "function", "function": {"name": "check_return_eligibility"}})
return_llm = llm.bind_tools([check_return_eligibility, process_return])

# ---------- 系统提示词 ----------
SUPERVISOR_PROMPT = """你是智能客服系统的监督者（Supervisor），负责调度专职 Agent 完成用户请求。

**业务调度规则：**
- 用户查询订单状态 → 调用 order_agent
- 用户咨询退货政策 → 调用 policy_agent
- 用户申请退货 → 必须严格按照以下顺序调度，不可跳过或颠倒：
  1. order_agent：获取订单信息（必须完成）
  2. policy_agent：获取退货政策（必须完成）
  3. return_agent：执行退货流程（内部会进行资格校验和创建退货单）
- 当所有必要信息收集完毕，调用 final_response 生成最终回复。
- 申请退货必须通过 route_to_agent 工具调用 return_agent 而不是 create_return_order

**错误处理规则：**
- 如果订单 Agent 返回“订单号不能为空”或“请提供有效的订单号”，立即调用 final_response，不得再次调用订单 Agent。
- 如果政策 Agent 返回“未找到相关政策”，可调用 final_response 告知用户，或根据情况转人工。
- 如果退货 Agent 返回“已超过退货期限”等资格不符信息，直接调用 final_response 向用户说明原因，并停止后续流程。

**重要约束：**
- 你只能调用 route_to_agent 工具，不能调用任何其他工具。
- 你自己不能判断退货资格、不能创建退货单、不能编造任何业务结果。
- 所有业务结果必须由专职 Agent 调用专业工具获得，你只负责调度。
- 如果订单 Agent 返回“订单不存在”或状态不允许，直接调用 final_response 告知用户，停止后续流程。
- 必须使用 route_to_agent 工具输出下一步行动。
- 必须使用中文交流。

【严格限制】你绝对禁止编造、猜测、或发明任何工具（Tool）名称。你只能严格从 `tools` 列表中选择。
如果用户的请求无法通过以上工具解决，请直接回复：“抱歉，我暂时无法处理这个请求。”，严禁凭空捏造函数名。
"""

ORDER_AGENT_PROMPT = """你是订单查询专家，唯一职责是查询订单信息。
你必须调用 search_order 工具获取订单数据，并将工具的返回结果原样返回给 Supervisor，不要添加任何额外解释或判断。
如果工具返回错误，也请直接返回该错误信息，不要自行编造结果。"""

POLICY_AGENT_PROMPT = """你是退货政策专家，唯一职责是检索退货政策。

**必须遵守：**
- 你必须调用 retrieve_return_policy 工具获取政策内容，不得自行编造政策条款。
- 将工具返回的政策文本原样作为你的回复，不要添加任何解释、判断或适用性分析。
- 严禁根据订单信息判断是否符合退货条件，这是后续 Agent 的职责。
- 如果工具返回“未找到相关政策”，也请直接返回该信息。
"""

RETURN_AGENT_PROMPT = """你是退货执行专家，负责处理退货申请。

**必须遵守的流程：**
1. 首先必须调用 check_return_eligibility 工具，检查订单是否符合退货条件。
2. 仔细查看check_return_eligibility工具返回结果中的 `eligible` 字段：
   - 如果 `eligible` 为 true，则可以继续调用 process_return 工具创建退货单。
   - 如果 `eligible` 为 false，则**绝对禁止**调用 process_return，直接根据返回的 `reason` 字段生成拒绝信息。
3. 不得跳过资格检查直接创建退货单，也不得编造“退货成功”等虚假信息。
4. 如果资格检查工具返回错误或异常，请如实告知用户，并停止后续操作。
"""

FINAL_RESPONSE_PROMPT = """你是客服代表，请根据对话历史生成简洁、友好的中文回复。

**要求：**
- 必须使用中文，不要输出 JSON、代码或内部工具名。
- 根据对话中已有的实际结果生成回复，不得添加未发生的信息。
- 如果信息不足或结果不确定，请礼貌地询问用户补充，或者告知已转人工处理。
"""

# ---------- 状态定义 ----------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str

# ---------- Supervisor 节点 ----------
def supervisor_node(state: AgentState) -> AgentState:
    print("\n" + "=" * 60)
    print("【Supervisor 监督者】分析中...")
    messages = list(state["messages"])
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + messages

    response = supervisor_llm.invoke(messages)

    if response.content:
        print(f"💭 Supervisor 思考：{response.content}")

    tool_messages = []
    next_agent = "__end__"

    if response.tool_calls:
        tool_names = [tc["name"] for tc in response.tool_calls]
        print(f"🔧 Supervisor 决定调用：{tool_names}")

        for tool_call in response.tool_calls:
            if tool_call["name"] == "route_to_agent":
                agent_name = tool_call["args"].get("agent_name", "")
                if agent_name in ["order_agent", "policy_agent", "return_agent", "final_response"]:
                    next_agent = agent_name
                else:
                    next_agent = "__end__"
            else:
                next_agent = "__end__"

            tool_messages.append(
                ToolMessage(
                    content=json.dumps({"status": "ok", "next_agent": next_agent}, ensure_ascii=False),
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                )
            )
    else:
        print("⚠️ Supervisor 未调用路由工具，强制结束")

    return {
        "messages": [response] + tool_messages,
        "next_agent": next_agent,
    }

# ---------- 订单 Agent 节点 ----------
def order_agent_node(state: AgentState) -> AgentState:
    print("\n" + "=" * 60)
    print("【订单 Agent】执行中...")
    messages = list(state["messages"])
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=ORDER_AGENT_PROMPT)] + messages

    response = order_llm.invoke(messages)

    if response.content:
        print(f"💭 订单 Agent 思考：{response.content}")

    if response.tool_calls:
        tool_messages = []
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]
            print(f"🔧 调用工具：{tool_name}，参数：{tool_args}")
            try:
                if tool_name == "search_order":
                    result = search_order.invoke(tool_args)
                else:
                    result = json.dumps({"error": "未知工具"}, ensure_ascii=False)
            except Exception as e:
                result = json.dumps({"error": f"工具执行失败：{str(e)}"}, ensure_ascii=False)
            print(f"👀 观察结果：{result}")
            tool_messages.append(ToolMessage(content=result, tool_call_id=tool_call_id, name=tool_name))
        return {"messages": [response] + tool_messages}
    else:
        return {"messages": [response]}

# ---------- 政策 Agent 节点 ----------
def policy_agent_node(state: AgentState) -> AgentState:
    print("\n" + "=" * 60)
    print("【政策 Agent】执行中...")
    messages = list(state["messages"])
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=POLICY_AGENT_PROMPT)] + messages

    response = policy_llm.invoke(messages)

    if response.content:
        print(f"💭 政策 Agent 思考：{response.content}")

    if response.tool_calls:
        tool_messages = []
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]
            print(f"🔧 调用工具：{tool_name}，参数：{tool_args}")
            try:
                if tool_name == "retrieve_return_policy":
                    result = retrieve_return_policy.invoke(tool_args)
                else:
                    result = json.dumps({"error": "未知工具"}, ensure_ascii=False)
            except Exception as e:
                result = json.dumps({"error": f"工具执行失败：{str(e)}"}, ensure_ascii=False)
            print(f"👀 观察结果：{result}")
            tool_messages.append(ToolMessage(content=result, tool_call_id=tool_call_id, name=tool_name))
        return {"messages": [response] + tool_messages}
    else:
        return {"messages": [response]}

# ---------- 退货 Agent 节点 ----------
MAX_RETURN_ITERATIONS = 4

def return_agent_node(state: AgentState) -> AgentState:
    print("\n" + "=" * 60)
    print("【退货 Agent】执行中...")
    messages = list(state["messages"])
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=RETURN_AGENT_PROMPT)] + messages

    # 内部 ReAct 循环：第一轮强制资格检查，之后根据 eligible 结果决定是否创建退货单
    new_messages = []
    for i in range(MAX_RETURN_ITERATIONS):
        current_llm = return_first_llm if i == 0 else return_llm
        response = current_llm.invoke(messages)
        new_messages.append(response)

        if response.content:
            print(f"💭 退货 Agent 思考：{response.content}")

        if not response.tool_calls:
            break

        tool_messages = []
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]
            print(f"🔧 调用工具：{tool_name}，参数：{tool_args}")
            try:
                if tool_name == "check_return_eligibility":
                    result = check_return_eligibility.invoke(tool_args)
                elif tool_name == "process_return":
                    result = process_return.invoke(tool_args)
                else:
                    result = json.dumps({"error": "未知工具"}, ensure_ascii=False)
            except Exception as e:
                result = json.dumps({"error": f"工具执行失败：{str(e)}"}, ensure_ascii=False)
            print(f"👀 观察结果：{result}")
            tool_messages.append(ToolMessage(content=result, tool_call_id=tool_call_id, name=tool_name))

        new_messages.extend(tool_messages)
        messages = messages + [response] + tool_messages

    return {"messages": new_messages}

# ---------- 最终回复节点 ----------
def final_response_node(state: AgentState) -> AgentState:
    print("\n" + "=" * 60)
    print("【最终回复】生成中...")
    messages = list(state["messages"])
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=FINAL_RESPONSE_PROMPT)] + messages

    response = llm.invoke(messages)
    print(f"💭 最终回复：{response.content}")
    return {"messages": [response]}

# ---------- 路由函数 ----------
def route_supervisor(state: AgentState) -> Literal["order_agent", "policy_agent", "return_agent", "final_response", "__end__"]:
    next_agent = state.get("next_agent", "__end__")
    if next_agent in ["order_agent", "policy_agent", "return_agent", "final_response"]:
        return next_agent
    else:
        return "__end__"

# ---------- 构建图 ----------
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("order_agent", order_agent_node)
    graph.add_node("policy_agent", policy_agent_node)
    graph.add_node("return_agent", return_agent_node)
    graph.add_node("final_response", final_response_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "order_agent": "order_agent",
            "policy_agent": "policy_agent",
            "return_agent": "return_agent",
            "final_response": "final_response",
            "__end__": END,
        }
    )

    graph.add_edge("order_agent", "supervisor")
    graph.add_edge("policy_agent", "supervisor")
    graph.add_edge("return_agent", "supervisor")
    graph.add_edge("final_response", END)

    return graph.compile()

# ---------- 测试入口 ----------
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
        final_msg = result["messages"][-1]
        print("\n🎯 最终回复：", final_msg.content)