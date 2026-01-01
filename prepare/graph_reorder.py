import subprocess

# 定义一个列表，包含所有要处理的数据集
# 'cora', 'flickr', 'ogbn-arxiv', 'reddit', 'yelp', 'amazon-products', 'ogbn-products', 'wiki-RfA', 'higgs-twitter'
datasets = ['olafu','mip1','dawson5','mycielskian15','mycielskian17','nd12k','human_gene1','pattern1',   
]

# 循环遍历数据集列表
for dataset_name in datasets:
    print(f"开始处理 {dataset_name} ...")
    # 执行 graph_reorder.sh，并将 dataset_name 作为参数传入
    subprocess.run(["bash", "graph_reorder.sh", dataset_name], check=True)
    print(f"处理完成 {dataset_name}\n")

print("全部数据集处理完毕！")