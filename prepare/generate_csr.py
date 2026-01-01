import torch
import scipy.io
import scipy.sparse as sp
import numpy as np
import os

# 定义加载和转换 .mtx 文件为 CSR 格式的函数
def load_mtx_and_save_csr(dataset_name, mtx_file_path):
    # 读取 .mtx 文件
    mtx = scipy.io.mmread(mtx_file_path)
    
    # 确保是稀疏矩阵
    if not sp.issparse(mtx):
        raise ValueError(f"The file {mtx_file_path} is not a sparse matrix!")

    # 将 row 和 col 合并为一个 numpy 数组
    indices = np.vstack([mtx.row, mtx.col])  # 使用 numpy.vstack 合并行列
    indices_tensor = torch.LongTensor(indices)  # 转换为 PyTorch 张量
    
    # 转换为 CSR 格式
    adj_csr = torch.sparse_coo_tensor(
        indices=indices_tensor,  # 行列索引
        values=torch.FloatTensor(mtx.data),  # 非零值
        size=torch.Size(mtx.shape)  # 矩阵的大小
    ).to_sparse_csr()  # 转换为 CSR 格式

    crow_indices = adj_csr.crow_indices().cpu().numpy()
    col_indices = adj_csr.col_indices().cpu().numpy()
    values = adj_csr.values().cpu().numpy()
    
    save_dir = f"./data_csr/{dataset_name}"  # 替换为你的保存路径
    os.makedirs(save_dir, exist_ok=True)

    # 根据数据集名称动态设置二进制文件的路径
    crow_file = f'{save_dir}/{dataset_name}_csr_crow_indices.bin'
    col_file = f'{save_dir}/{dataset_name}_csr_col_indices.bin'
    values_file = f'{save_dir}/{dataset_name}_csr_values.bin'

    # 保存为二进制文件
    crow_indices.tofile(crow_file)  # 保存 crow_indices
    col_indices.tofile(col_file)    # 保存 col_indices
    values.tofile(values_file)      # 保存 values

    print(f"CSR crow_indices saved to {crow_file}")
    print(f"CSR col_indices saved to {col_file}")
    print(f"CSR values saved to {values_file}")

# 定义加载和转换 .mtx 文件为 CSR 格式的函数，并进行节点重排序
def load_mtx_and_save_csr_with_reorder(dataset_name, mtx_file_path, reorder_file_path=None):
    try:
        # 读取 .mtx 文件
        mtx = scipy.io.mmread(mtx_file_path)

        # 确保是稀疏矩阵
        if not sp.issparse(mtx):
            raise ValueError(f"The file {mtx_file_path} is not a sparse matrix!")

        # 读取节点重排序索引文件
        if reorder_file_path:
            with open(reorder_file_path, 'r') as f:
                reorder_indices = np.array([int(line.strip()) for line in f.readlines()])
            print(f"Reorder indices for {dataset_name} loaded from {reorder_file_path}")

            # 使用重排序索引调整行列索引
            mtx.row = reorder_indices[mtx.row]  # 重排序 row 索引
            mtx.col = reorder_indices[mtx.col]  # 重排序 col 索引

        # 将 row 和 col 合并为一个 numpy 数组
        indices = np.vstack([mtx.row, mtx.col])  # 使用 numpy.vstack 合并行列
        indices_tensor = torch.LongTensor(indices)  # 转换为 PyTorch 张量

        # 转换为 CSR 格式
        adj_csr = torch.sparse_coo_tensor(
            indices=indices_tensor,  # 行列索引
            values=torch.FloatTensor(mtx.data),  # 非零值
            size=torch.Size(mtx.shape)  # 矩阵的大小
        ).to_sparse_csr()  # 转换为 CSR 格式

        crow_indices = adj_csr.crow_indices().cpu().numpy()
        col_indices = adj_csr.col_indices().cpu().numpy()
        values = adj_csr.values().cpu().numpy()

        # 创建保存目录
        save_dir = f"./data_reorder_csr/{dataset_name}"  
        os.makedirs(save_dir, exist_ok=True)

        # 动态设置文件路径
        crow_file = f'{save_dir}/{dataset_name}_csr_crow_indices.bin'
        col_file = f'{save_dir}/{dataset_name}_csr_col_indices.bin'
        values_file = f'{save_dir}/{dataset_name}_csr_values.bin'

        # 保存为二进制文件
        crow_indices.tofile(crow_file)
        col_indices.tofile(col_file)
        values.tofile(values_file)

        print(f"CSR data for {dataset_name} saved successfully!")
        print(f"Crow indices: {crow_file}")
        print(f"Col indices: {col_file}")
        print(f"Values: {values_file}")
    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")

# 主函数
def main():
    # 'cora', 'flickr', 'ogbn-arxiv', 'reddit', 'yelp', 'amazon-products', 'ogbn-products', 'wiki-RfA', 'higgs-twitter'
    # 'olafu','mip1','dawson5','mycielskian15','mycielskian17','nd12k','human_gene1','pattern1',
    datasets = ['reddit']
    for dataset_name in datasets:
        # 指定 .mtx 文件路径
        mtx_file_path = f'./data_mtx/{dataset_name}.mtx'
        reorder_file_path = f'./data_reorder/{dataset_name}/{dataset_name}_coo_reorder.txt'

        # 检查重排序文件是否存在
        if os.path.exists(reorder_file_path):
            load_mtx_and_save_csr_with_reorder(dataset_name, mtx_file_path, reorder_file_path)
        else:
            print(f"Reorder file {reorder_file_path} does not exist, skipping reorder.")
            load_mtx_and_save_csr(dataset_name, mtx_file_path)

if __name__ == "__main__":
    main()
