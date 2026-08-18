import sqlite3
import os
from datetime import datetime

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