import streamlit as st
import sys
import io
import json
from contextlib import redirect_stdout

from graph import build_graph

# ---------- 页面配置 ----------
st.set_page_config(page_title="智能客服 MVP", page_icon="🤖")
st.title("🤖 多 Agent 智能客服系统（简历 MVP）")
st.caption("基于 LangGraph + DeepSeek + Chroma 向量检索")

# ---------- 初始化会话状态 ----------
if "messages" not in st.session_state:
    st.session_state.messages = []  # 聊天历史：[{"role": "user"/"assistant", "content": str}]

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()  # 只构建一次，避免重复加载模型

# ---------- 辅助函数：格式化日志中的 JSON ----------
def _format_json_in_text(text: str) -> str:
    """
    扫描文本，找到所有 JSON 对象（以 { 开头），尝试解析并用缩进格式替换。
    这样日志中的 JSON 数据会以多行结构显示，更易读。
    """
    decoder = json.JSONDecoder()
    result_parts = []
    pos = 0
    while pos < len(text):
        # 查找下一个 '{' 的位置
        start = text.find('{', pos)
        if start == -1:
            result_parts.append(text[pos:])
            break

        # 尝试从 start 开始解析 JSON
        try:
            obj, end = decoder.raw_decode(text[start:])
            pretty = json.dumps(obj, indent=2, ensure_ascii=False)
            result_parts.append(text[pos:start])
            result_parts.append('\n' + pretty + '\n')
            pos = start + end
        except json.JSONDecodeError:
            # 不是有效 JSON，保留原样并移动一个字符
            result_parts.append(text[pos:start+1])
            pos = start + 1

    return ''.join(result_parts)

# ---------- 辅助函数：执行图并捕获日志 ----------
def run_graph_with_logs(langchain_messages):
    """
    运行 LangGraph，传入完整的消息列表（包含历史），捕获 ReAct 日志。
    返回最终回复和原始日志字符串。
    """
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    try:
        initial_state = {"messages": langchain_messages}
        result = st.session_state.graph.invoke(initial_state)
        final_answer = result["messages"][-1].content
    except Exception as e:
        final_answer = f"系统错误：{str(e)}"
    finally:
        sys.stdout = old_stdout

    logs = buffer.getvalue()
    return final_answer, logs

# ---------- 显示历史消息 ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- 用户输入 ----------
if prompt := st.chat_input("请输入您的问题，例如：帮我查一下订单 ORD123456"):
    # 添加用户消息到历史
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 构建 LangGraph 消息列表（包含历史所有消息）
    langchain_messages = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            langchain_messages.append(("user", msg["content"]))
        elif msg["role"] == "assistant":
            langchain_messages.append(("assistant", msg["content"]))
    # 注意：系统提示词不需要在此添加，graph.py 中的 router_node 会自动注入

    # 执行图并获取回复和原始日志
    with st.spinner("Agent 思考中..."):
        final_answer, raw_logs = run_graph_with_logs(langchain_messages)

    # 添加 AI 消息到历史
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
    with st.chat_message("assistant"):
        st.markdown(final_answer)

    # 格式化日志中的 JSON，提升可读性
    formatted_logs = _format_json_in_text(raw_logs)

    # 显示内部日志（折叠面板）
    with st.expander("🔍 查看内部 ReAct 日志", expanded=False):
        st.code(formatted_logs, language="text")