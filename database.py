import sqlite3
import os
import random
from datetime import datetime, timedelta

DB_PATH = "customer_service.db"

def get_connection():
    """获取数据库连接，开启行工厂以便字典访问"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """初始化表结构并插入示例数据"""
    conn = get_connection()
    cur = conn.cursor()

    # 创建表
    cur.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            item TEXT NOT NULL,
            amount REAL NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS return_requests (
            return_id TEXT PRIMARY KEY,
            order_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            reason TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders (order_id)
        )
    """)

    # 插入示例订单（如果不存在）
    sample_orders = [
        ("ORD123456", "user_001", "机械键盘", 399.0, "已签收", "2026-08-01 10:00:00", "2026-08-05 15:30:00"),
        ("ORD789012", "user_002", "降噪耳机", 899.0, "运输中", "2026-08-10 09:20:00", "2026-08-12 18:00:00"),
        ("ORD555555", "user_001", "智能音箱", 299.0, "待发货", "2026-08-13 11:00:00", "2026-08-13 11:00:00"),
    ]

    # ---------- 批量测试订单：ORD100001~ORD100100（10 个用户轮转） ----------
    # 固定随机种子保证每次生成一致，配合 INSERT OR IGNORE 实现幂等（重启不重复插入）
    random.seed(42)
    products = [
        ("机械键盘", 399.0), ("降噪耳机", 899.0), ("智能音箱", 299.0),
        ("无线鼠标", 129.0), ("4K显示器", 1499.0), ("USB-C扩展坞", 259.0),
        ("人体工学椅", 1999.0), ("护眼台灯", 189.0), ("移动硬盘", 459.0),
        ("蓝牙手柄", 349.0),
    ]
    statuses = ["已签收", "运输中", "待发货", "已完成", "已取消"]
    status_weights = [40, 25, 20, 10, 5]

    now = datetime.now()
    for i in range(1, 101):
        order_id = f"ORD{100000 + i}"
        user_id = f"user_{(i - 1) % 10 + 1:03d}"   # user_001 ~ user_010 轮转
        item, amount = products[(i - 1) % len(products)]
        status = random.choices(statuses, weights=status_weights)[0]
        created_at = now - timedelta(days=random.randint(1, 60))
        if status == "已签收":
            # 签收时间 1~30 天前：覆盖 7 天内(可无理由退)、7~15天(仅质量问题)、超15天(不可退)三类样本
            updated_at = now - timedelta(days=random.randint(1, 30))
        else:
            updated_at = created_at + timedelta(days=random.randint(0, 3))
        sample_orders.append((
            order_id, user_id, item, amount, status,
            created_at.strftime("%Y-%m-%d %H:%M:%S"),
            updated_at.strftime("%Y-%m-%d %H:%M:%S"),
        ))

    for order in sample_orders:
        cur.execute("""
            INSERT OR IGNORE INTO orders (order_id, user_id, item, amount, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, order)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("数据库初始化完成。")