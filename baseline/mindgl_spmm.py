import mindspore as ms
# from mindspore_gl import Graph, GraphField
# from mindspore_gl.nn import GNNCell
from mindspore import Tensor
import mindspore.ops as ops

import time
import numpy as np

from mindspore import context
context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", device_id=0)

# class GraphAggregation(GNNCell):
#     def __init__(self):
#         """
#         初始化图聚合层。
#         """
#         super().__init__()

#     def construct(self, x, g: Graph):
#         """
#         构造图聚合的前向计算流程。
        
#         Args:
#             x (Tensor): 输入节点特征，形状为 (num_nodes, embedding_dim)。
#             g (Graph): 图对象，用于聚合。
        
#         Returns:
#             Tensor: 聚合后的节点特征，形状为 (num_nodes, embedding_dim)。
#         """
#         x = ms.ops.Squeeze()(x)
#         g.set_vertex_attr({"x": x})
#         # Aggregation using g.sum
#         for v in g.dst_vertex:
#             v.x = g.sum([u.x for u in v.innbs])
#         return [v.x for v in g.dst_vertex]

def GraphAggregation_gl(
    x,
    src_idx,
    dst_idx,
    n_nodes,
):    # 定义聚合操作
    SCATTER_ADD = ms.ops.TensorScatterAdd()
    GATHER = ms.ops.Gather()
    ZEROS = ms.ops.Zeros()
    SHAPE = ms.ops.Shape()
    RESHAPE = ms.ops.Reshape()

    # 调整索引形状，确保它们是二维的
    scatter_dst_idx = RESHAPE(dst_idx, (SHAPE(dst_idx)[0], 1))

    # 收集源节点特征
    SCATTER_INPUT_SNAPSHOT1 = GATHER(x, src_idx, 0)

    # 创建一个零张量作为目标节点的初始值
    x = SCATTER_ADD(
        ZEROS(
            (n_nodes,) + SHAPE(SCATTER_INPUT_SNAPSHOT1)[1:],
            SCATTER_INPUT_SNAPSHOT1.dtype
        ),
        scatter_dst_idx,
        SCATTER_INPUT_SNAPSHOT1
    )

    return x
    

def csr_to_coo_repeat(nodePointer, edgeList):
    """
    使用向量化操作将 CSR 格式转换为 COO 格式。

    Args:
        nodePointer (numpy.ndarray): CSR 格式的行指针，形状为 (num_nodes + 1,)
        edgeList (numpy.ndarray): CSR 格式的列索引（边列表），形状为 (num_edges,)

    Returns:
        src_idx (numpy.ndarray): COO 格式的源节点索引
        dst_idx (numpy.ndarray): COO 格式的目标节点索引
    """
    # 使用 np.repeat 生成源节点索引
    print("repeat start")
    dst_idx = np.repeat(np.arange(len(nodePointer) - 1, dtype=np.int32), np.diff(nodePointer).astype(np.int32))
    print("repeat over")
    src_idx = edgeList
    
    src_idx = np.array(src_idx, dtype=np.int32)
    dst_idx = np.array(dst_idx, dtype=np.int32)
    
    return src_idx, dst_idx

def aiv_spmm_minibatch(x, src_idx, dst_idx, n_nodes, batch_size):
    """
    批次处理的 Sparse-Dense 矩阵乘法 (SPMM)。
    
    参数：
        x (Tensor): 节点特征矩阵，形状为 (n_nodes, feature_dim)。
        src_idx (Tensor): 边的源节点索引，形状为 (n_edges,)。
        dst_idx (Tensor): 边的目标节点索引，形状为 (n_edges,)。
        n_nodes (int): 节点总数。
        batch_size (int): 每次处理的边数。
    
    返回：
        Tensor: 聚合后的节点特征矩阵，形状为 (n_nodes, feature_dim)。
    """
    # 初始化所需算子
    SCATTER_ADD = ops.TensorScatterAdd()
    RESHAPE = ops.Reshape()
    ZEROS = ops.Zeros()
    
    # 初始化输出张量
    output_shape = (n_nodes,) + x.shape[1:]  # 输出形状
    output = ZEROS(output_shape, x.dtype)  # 零初始化

    # 获取总边数
    total_edges = src_idx.shape[0]
    
    # 分批次处理边的索引
    for start_idx in range(0, total_edges, batch_size):
        end_idx = min(start_idx + batch_size, total_edges)
        # 当前批次的索引
        batch_src_idx = src_idx[start_idx:end_idx]
        batch_dst_idx = dst_idx[start_idx:end_idx]
        # 将目标索引调整为二维
        batch_dst_idx_reshaped = RESHAPE(batch_dst_idx, (-1, 1))
        # 收集源节点特征
        batch_features = x[batch_src_idx] 
        # 执行 TensorScatterAdd 累加
        output = SCATTER_ADD(output, batch_dst_idx_reshaped, batch_features)
    return output

def main():
    # dataset_name = 'amazon-products'
    # 'cora', 'flickr', 'ogbn-arxiv', 'reddit', 'yelp', 'amazon-products', 'ogbn-products'
    # 'wiki-RfA', 'higgs-twitter'
    # 'olafu','mip1','dawson5','mycielskian15','mycielskian17','nd12k','human_gene1','pattern1',
    for dataset_name in ['reddit']:

        nodePointer_np = np.fromfile(f'../prepare/data_csr/{dataset_name}/{dataset_name}_csr_crow_indices.bin', dtype=np.int64)
        edgeList_np = np.fromfile(f'../prepare/data_csr/{dataset_name}/{dataset_name}_csr_col_indices.bin', dtype=np.int64)
        
        nodePointer = Tensor(nodePointer_np, ms.int32)
        edgeList = Tensor(edgeList_np, ms.int32)  # 边列表，表示连接的节点ID
        embedding_dim = 256
        
        n_nodes = nodePointer.shape[0] - 1
        n_edges = edgeList.shape[0]
        
        src_idx, dst_idx = csr_to_coo_repeat(nodePointer_np, edgeList_np)
        src_idx = Tensor(src_idx, ms.int32)
        dst_idx = Tensor(dst_idx, ms.int32)
        
        print("src_idx shape : ", src_idx.shape)
        print("dst_idx shape : ", dst_idx.shape)
        np.random.seed(42)
        node_feat = Tensor(np.random.rand((len(nodePointer) - 1), embedding_dim), ms.float16)
        print("input feature:\n", node_feat)

        print("Data load over!!")
        total_time = 0
        num_runs = 10
        # ms_profiler = ms.Profiler(profiler_level=0, aicore_metrics=1, profile_memory=True, l2_cache=True, hbm_ddr=True, pcie=True, output_path="./prof_gl")
        for i in range(num_runs):
            start_time = time.time()
            batch_size = 8*1024*1024
            ret = aiv_spmm_minibatch(node_feat, src_idx, dst_idx, n_nodes, batch_size)
            print(ret[0][0])
            end_time = time.time()
            if(i > 0):
                total_time += end_time - start_time
            print(f"Execution time : {(end_time - start_time):.4f} seconds")
        # ms_profiler.analyse()
        print(f"Execution time for gl: {total_time/(num_runs - 1):.4f} seconds")

    
# 执行主函数
if __name__ == "__main__":
    main()