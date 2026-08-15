from rag import retrieve_return_policy

if __name__ == "__main__":
    queries = [
        "七天无理由退货的运费谁承担？",
        "商品有质量问题怎么退货？",
        "定制商品可以退吗？",
    ]
    for q in queries:
        print(f"查询：{q}")
        print(f"结果：{retrieve_return_policy.invoke({'query': q})}\n")