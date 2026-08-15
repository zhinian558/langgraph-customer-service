from tools import search_order, process_return

if __name__ == "__main__":
    print("=== 测试 search_order ===")
    print(search_order.invoke({"order_id": "ORD123456"}))

    print("\n=== 测试 process_return ===")
    print(process_return.invoke({"order_id": "ORD123456", "reason": "尺寸不合适"}))