import scipy.io
import scipy.sparse as sp
import numpy as np
import os

# 定义加载 .mtx 文件并保存为边列表的函数
def load_mtx_and_save_coo(dataset_name, mtx_file_path):
    # 读取 .mtx 文件
    mtx = scipy.io.mmread(mtx_file_path)
    
    # 确保是稀疏矩阵
    if not sp.issparse(mtx):
        raise ValueError(f"The file {mtx_file_path} is not a sparse matrix!")
    
    # 获取行列索引和非零值
    row_indices = mtx.row  # 行索引
    col_indices = mtx.col  # 列索引
    
    # 生成边列表，每一行是两个索引，表示一个边
    edge_list = np.vstack([row_indices, col_indices]).T  # 转换为边列表的形式
    
    # 生成保存目录
    save_dir = f"./data_coo/{dataset_name}"  # 替换为你的保存路径
    os.makedirs(save_dir, exist_ok=True)
    
    # 动态设置保存的边列表文件路径
    edge_list_file = f'{save_dir}/{dataset_name}_coo.txt'
    
    # 保存为 .txt 文件，每行是 "源节点 目标节点"
    np.savetxt(edge_list_file, edge_list, fmt='%d', delimiter=' ', comments='')
    
    print(f"Edge list saved to {edge_list_file}")

def main():
    # 处理多个数据集
    # 'olafu','mip1','dawson5','mycielskian15','mycielskian17','nd12k','human_gene1','pattern1',
    # 'cora', 'flickr', 'ogbn-arxiv', 'reddit', 'yelp', 'amazon-products', 'ogbn-products', 'wiki-RfA', 'higgs-twitter'
    for dataset_name in ['reddit']:
        # 指定 .mtx 文件路径
        mtx_file_path = f'./data_mtx/{dataset_name}.mtx'
        # 加载 .mtx 文件并保存为边列表
        load_mtx_and_save_coo(dataset_name, mtx_file_path)

# 执行主函数
if __name__ == "__main__":
    main()