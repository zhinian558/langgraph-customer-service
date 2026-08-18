import json
from datetime import datetime, timedelta
from langchain_core.tools import tool
import database

# ---------- 工具 1：查询订单 ----------
@tool
def search_order(order_id: str) -> str:
    """
    查询订单状态工具。
    输入：order_id，例如 ORD123456。
    输出：订单状态 JSON 字符串。
    """

    if not order_id or order_id.strip() == "":
        return json.dumps({"error": "订单号不能为空，请提供有效的订单号"}, ensure_ascii=False)
    
    conn = database.get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM orders WHERE order_id = ?", (order_id.upper(),))
    order = cur.fetchone()
    conn.close()

    if order:
        return json.dumps(dict(order), ensure_ascii=False)

    return json.dumps({"error": "订单不存在", "order_id": order_id}, ensure_ascii=False)

# ---------- 工具 2：创建退货申请（含业务校验） ----------
@tool
def process_return(order_id: str, reason: str = "七天无理由") -> str:
    """
    创建退货申请工具。
    输入：order_id，可选 reason。
    输出：退货申请单 JSON 字符串。
    业务规则：
    - 订单必须存在
    - 订单状态必须为“已签收”
    - 签收时间不超过 15 天（质量问题）或 7 天（无理由）
    """
    conn = database.get_connection()
    cur = conn.cursor()

    # 查询订单
    cur.execute("SELECT * FROM orders WHERE order_id = ?", (order_id.upper(),))
    order = cur.fetchone()
    if not order:
        conn.close()
        return json.dumps({"error": "订单不存在，无法申请退货"}, ensure_ascii=False)

    order_dict = dict(order)
    if order_dict["status"] != "已签收":
        conn.close()
        return json.dumps({"error": "订单尚未签收或状态不允许退货"}, ensure_ascii=False)

    # 检查签收时间
    updated_at = datetime.strptime(order_dict["updated_at"], "%Y-%m-%d %H:%M:%S")
    current_time = datetime.now()
    days_diff = (current_time - updated_at).days

    max_days = 7 if "无理由" in reason else 15
    if days_diff > max_days:
        conn.close()
        return json.dumps({"error": f"已超过{max_days}天退货期限"}, ensure_ascii=False)

    # 生成退货单
    return_id = f"RTN-{order_id.upper()}-{current_time.strftime('%Y%m%d%H%M%S')}"
    cur.execute("""
        INSERT INTO return_requests (return_id, order_id, user_id, reason, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        return_id,
        order_id.upper(),
        order_dict["user_id"],
        reason,
        "待审核",
        current_time.strftime("%Y-%m-%d %H:%M:%S"),
        current_time.strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()

    return json.dumps({
        "return_id": return_id,
        "order_id": order_id.upper(),
        "reason": reason,
        "status": "退货申请已创建，等待审核"
    }, ensure_ascii=False)

# ---------- 工具 3：检查退货资格（真实业务校验） ----------
@tool
def check_return_eligibility(order_id: str, reason: str = "七天无理由") -> str:
    """
    检查订单是否符合退货条件。
    输入：order_id，可选 reason。
    输出：JSON 字符串，包含 eligible 字段（true/false）和原因。
    规则：
    - 订单必须存在且状态为已签收
    - 无理由退货：签收后7天内
    - 质量问题退货：签收后15天内
    """
    conn = database.get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM orders WHERE order_id = ?", (order_id.upper(),))
    order = cur.fetchone()
    conn.close()

    if not order:
        return json.dumps({"eligible": False, "reason": "订单不存在"}, ensure_ascii=False)

    order_dict = dict(order)
    if order_dict["status"] != "已签收":
        return json.dumps({"eligible": False, "reason": "订单尚未签收或状态不允许退货"}, ensure_ascii=False)

    updated_at = datetime.strptime(order_dict["updated_at"], "%Y-%m-%d %H:%M:%S")
    current_time = datetime.now()
    days_diff = (current_time - updated_at).days

    max_days = 7 if "无理由" in reason else 15
    elapsed = current_time - updated_at
    if elapsed > timedelta(days=max_days):
        return json.dumps(
            {"eligible": False, "reason": f"已超过{max_days}天退货期限"},
            ensure_ascii=False
        )

    return json.dumps(
        {"eligible": True, "reason": "符合退货条件", "max_days": max_days, "days_diff": elapsed.days},
        ensure_ascii=False
    )