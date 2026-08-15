import json
from datetime import datetime
from langchain_core.tools import tool


@tool
def search_order(order_id: str) -> str:
    """
    查询订单状态工具。
    输入：order_id，例如 ORD123456。
    输出：订单状态 JSON 字符串。
    """
    mock_orders = {
        "ORD123456": {
            "order_id": "ORD123456",
            "status": "已签收",
            "item": "机械键盘",
            "amount": 399.0,
            "created_at": "2026-08-01"
        },
        "ORD789012": {
            "order_id": "ORD789012",
            "status": "运输中",
            "item": "降噪耳机",
            "amount": 899.0,
            "created_at": "2026-08-10"
        }
    }

    order = mock_orders.get(order_id.upper())
    if order:
        return json.dumps(order, ensure_ascii=False)

    return json.dumps(
        {"error": "订单不存在", "order_id": order_id},
        ensure_ascii=False
    )


@tool
def process_return(order_id: str, reason: str = "七天无理由") -> str:
    """
    创建退货申请工具。
    输入：order_id，可选 reason。
    输出：退货申请单 JSON 字符串。
    """
    return_id = f"RTN-{order_id.upper()}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    return json.dumps(
        {
            "return_id": return_id,
            "order_id": order_id,
            "reason": reason,
            "status": "退货申请已创建，等待审核"
        },
        ensure_ascii=False
    )