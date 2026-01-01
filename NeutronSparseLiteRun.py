import time
import os
import numpy as np
import mindspore as ms
from mindspore import Tensor, COOTensor, ops, context

# ==========================================
# 1. 全局环境设置
# ==========================================
# 注意：device_id 根据你的实际环境修改，单卡通常是 0
context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", device_id=0)

# ==========================================
# 2. 配置类
# ==========================================
class SpMMConfig:
    """配置类：管理超参数"""
    def __init__(self, dataset_name, blk_h=8192, blk_w=32, embed_dim=256):
        self.dataset = dataset_name
        # 数据根目录，根据你的实际路径调整
        self.data_root = f'./data_save/{dataset_name}'
        self.blk_h = blk_h
        self.blk_w = blk_w
        self.embed_dim = embed_dim
        
        if self.embed_dim % self.blk_w != 0:
            print(f"Warning: embed_dim ({embed_dim}) usually works best as multiple of blk_w ({blk_w})")

# ==========================================
# 3. 核心引擎类 (OOP 封装)
# ==========================================
class HybridSpMMEngine:
    """
    混合 SpMM 计算引擎
    包含基于 Cube (Dense A11) 和 Vector (Sparse A12/A2) 的混合计算逻辑
    """
    def __init__(self, config: SpMMConfig):
        self.cfg = config
        self.ops_scatter = ops.TensorScatterAdd()
        
        # 1. 加载数据
        print(f"--> Loading dataset: {config.dataset}...")
        self._load_data()
        
        # 2. 预处理 A11 (Dense Part) - 用于 Cube 核心计算
        print("--> Preprocessing A11 (Dense Part)...")
        self.a11_mats, self.a11_edges = self._prepare_a11()
        
        # 3. 预处理 A12/A2 (Sparse Part) - 转换为 COO 格式以便 Vector 核心计算
        print("--> Preprocessing A12/A2 (Sparse Part)...")
        self.a12_coo = self._csr_to_coo(self.raw_a12['ptr'], self.raw_a12['idx'])
        self.a2_coo = self._csr_to_coo(self.raw_a2['ptr'], self.raw_a2['idx'])
        
        # 4. 将 shape 调整为 (-1, 1) 以适配 TensorScatterAdd 的索引要求
        self.map_a11_row = self.raw_a11['orig_row'].reshape(-1, 1)
        self.map_a12_row = self.raw_a12['orig_row'].reshape(-1, 1)
        
        # 5. 计算动态 batch size (用于 AIVector 分块处理)
        self.batch_size = self._calc_dynamic_batch_size(self.a11_num_nodes, config.embed_dim)
        
        # 6. 清理原始不再需要的 numpy/tensor 数据以释放内存
        self._cleanup_raw_data()
        print("--> Initialization Complete.")

    def _load_data(self):
        """加载二进制文件到内存 (Private Helper)"""
        def load_bin(folder, name, dtype=np.int32):
            # 优先尝试新命名规则，不存在则回退到旧命名规则
            path = os.path.join(self.cfg.data_root, folder, f'{name}.bin')
            if not os.path.exists(path): 
                path = os.path.join(self.cfg.data_root, folder, f'{name}_tensor.bin')
            return np.fromfile(path, dtype=dtype)

        # 加载 A11 数据
        self.raw_a11 = {
            'ptr': Tensor(load_bin('A11', 'A11_nodePointer'), ms.int32),
            'idx': Tensor(load_bin('A11', 'A11_edgeList'), ms.int32),
            'part': Tensor(load_bin('A11', 'A11_block_partition'), ms.int32),
            'col_map': Tensor(load_bin('A11', 'A11_edge_to_column'), ms.int32),
            'row_map': Tensor(load_bin('A11', 'A11_edge_to_row'), ms.int32),
            'orig_row': Tensor(load_bin('A11', 'A11_origin_row_index'), ms.int32)
        }
        self.a11_num_nodes = self.raw_a11['ptr'].shape[0] - 1

        # 加载 A12 数据
        self.raw_a12 = {
            'ptr': load_bin('A12', 'A12_nodePointer'),
            'idx': load_bin('A12', 'A12_edgeList'),
            'orig_row': Tensor(load_bin('A12', 'A12_origin_row_index'), ms.int32)
        }
        self.a12_nodes = len(self.raw_a12['ptr']) - 1

        # 加载 A2 数据
        self.raw_a2 = {
            'ptr': load_bin('A2', 'A2_nodePointer'),
            'idx': load_bin('A2', 'A2_edgeList'),
        }
        self.a2_nodes = len(self.raw_a2['ptr']) - 1
        
        # 全局节点数 (用于生成随机输入)
        global_ptr = load_bin('origin', 'A_nodePointer', dtype=np.int32)
        self.num_global_nodes = len(global_ptr) - 1

    def _prepare_a11(self):
        """构建 A11 的稠密块列表和边索引"""
        sparse_mats = []
        edge_arrays = []
        
        ptr = self.raw_a11['ptr'].asnumpy()
        part = self.raw_a11['part']
        row_map = self.raw_a11['row_map']
        col_map = self.raw_a11['col_map']
        idx = self.raw_a11['idx']
        
        blk_h = self.cfg.blk_h
        blk_w = self.cfg.blk_w
        num_wins = part.shape[0]

        for win_id in range(num_wins):
            # 1. 构建 Dense Matrix Block
            num_tc = int(part[win_id].asnumpy().item())
            start = int(ptr[win_id * blk_h])
            end = int(ptr[min((win_id + 1) * blk_h, self.a11_num_nodes)])
            
            # 转换为局部坐标
            r_idx = row_map[start:end] - (win_id * blk_h)
            c_idx = col_map[start:end]
            
            indices = ops.stack((r_idx, c_idx), axis=1)
            values = ops.ones((end - start,), ms.float16)
            shape = (blk_h, blk_w * num_tc)
            
            # 转为 Dense 用于 Matmul (Cube Unit)
            dense_block = COOTensor(indices, values, shape).to_dense()
            sparse_mats.append(dense_block)
            
            # 2. 构建 Edge Array (Sorted & Unique)
            raw_edges = idx[start:end]
            if raw_edges.shape[0] > 0:
                unique_edges = ops.unique(raw_edges)[0]
                sorted_edges = ops.sort(unique_edges)[0]
                edge_arrays.append(sorted_edges)
            else:
                edge_arrays.append(Tensor([], ms.int32))
                
        self.max_tc_blocks = int(part.max().asnumpy().item())
        return sparse_mats, edge_arrays

    @staticmethod
    def _csr_to_coo(ptr, idx):
        """CSR 转 COO (Vectorized numpy ops)"""
        repeat_counts = np.diff(ptr).astype(np.int32)
        dst_idx = np.repeat(np.arange(len(ptr) - 1, dtype=np.int32), repeat_counts)
        src_idx = np.array(idx, dtype=np.int32)
        return Tensor(src_idx, ms.int32), Tensor(dst_idx, ms.int32)

    def _cleanup_raw_data(self):
        """释放不再需要的原始数据"""
        del self.raw_a12
        del self.raw_a2

    def _calc_dynamic_batch_size(self, num_nodes, dim):
        """根据矩阵规模和维度动态计算 Batch Size"""
        tc_blk_size = 2 * num_nodes / (1024**3)
        base = 1024 * 1024
        if tc_blk_size < 4:
            factors = (32, 16, 8)
        elif tc_blk_size < 8:
            factors = (16, 8, 4)
        elif tc_blk_size < 12:
            factors = (8, 4, 2)
        else:
            factors = (4, 2, 1)
            
        if dim <= 128: return factors[0] * base
        elif dim <= 256: return factors[1] * base
        return factors[2] * base

    def compute_dense_part(self, x):
        """AICore (Cube) 计算部分 (A11)"""
        res = ops.zeros((self.a11_num_nodes, self.cfg.embed_dim), ms.float16)
        
        # 预分配最大的 Tile 缓存
        max_tile_w = self.cfg.blk_w * self.max_tc_blocks
        tile_buffer = ops.zeros((max_tile_w, self.cfg.embed_dim), ms.float16)
        
        blk_h = self.cfg.blk_h
        
        for win_id, mat in enumerate(self.a11_mats):
            edge_arr = self.a11_edges[win_id]
            if edge_arr.shape[0] == 0: continue

            # Gather input features
            feat_subset = x[edge_arr]
            tile_buffer[:feat_subset.shape[0]] = feat_subset
            
            # Matmul (Sparse Block x Dense Tile)
            curr_width = mat.shape[1]
            valid_tile = tile_buffer[:curr_width]
            temp_res = ops.matmul(mat, valid_tile)
            
            # Assign back to result
            r_start = win_id * blk_h
            r_end = min((win_id + 1) * blk_h, self.a11_num_nodes)
            h_len = r_end - r_start
            res[r_start:r_end] = temp_res[:h_len]
            
        return res

    def compute_sparse_part(self, x, src_idx, dst_idx, n_nodes):
        """AIVector (Vector) 计算部分 (Generic Scatter Add) - 支持 Batch"""
        out = ops.zeros((n_nodes, x.shape[1]), x.dtype)
        total_edges = src_idx.shape[0]
        
        for start in range(0, total_edges, self.batch_size):
            end = min(start + self.batch_size, total_edges)
            
            b_src = src_idx[start:end]
            b_dst = dst_idx[start:end].reshape(-1, 1)
            b_feat = x[b_src]
            
            out = self.ops_scatter(out, b_dst, b_feat)
            
        return out

    def reindex_merge(self, c11, c12, c2):
        """合并 A11, A12, A2 的结果"""
        # 使用预先提取并 reshape 好的索引
        c2 = self.ops_scatter(c2, self.map_a11_row, c11)
        c2 = self.ops_scatter(c2, self.map_a12_row, c12)
        return c2

    def forward(self, x):
        """前向计算主流程"""
        # 1. 计算 A11 (AIC Dense)
        c11 = self.compute_dense_part(x)
        # 2. 计算 A12 (AIV Sparse)
        c12 = self.compute_sparse_part(x, self.a12_coo[0], self.a12_coo[1], self.a12_nodes)
        # 3. 计算 A2 (AIV Sparse)
        c2 = self.compute_sparse_part(x, self.a2_coo[0], self.a2_coo[1], self.a2_nodes)   
        # 4. 合并结果
        result = self.reindex_merge(c11, c12, c2)
        return result

# ==========================================
# 4. 主程序入口
# ==========================================
def main():
    # 1. 设置配置
    # 请确保 './data_save/reddit/...' 目录下有对应数据
    dataset = 'reddit' 
    cfg = SpMMConfig(dataset, blk_h=2048, blk_w=32, embed_dim=256)
    
    # 2. 初始化引擎 (包含数据加载和预处理)
    engine = HybridSpMMEngine(cfg)
    
    # 3. 构造随机输入特征
    print(f"--> Generating random features ({engine.num_global_nodes} x {cfg.embed_dim})...")
    np.random.seed(42)
    # 输入特征通常不需要梯度，放在 NPU 上
    x = Tensor(np.random.rand(engine.num_global_nodes, cfg.embed_dim), ms.float16)
    
    # 4. 预热 & 性能测试循环
    warmup = 5
    num_runs = 20
    print(f"--> Starting benchmark ({num_runs} epochs)...")
    
    total_time = 0.0
    for i in range(num_runs):
        ms.runtime.synchronize() 
        t0 = time.time()
        # 执行核心计算
        output = engine.forward(x)
        ms.runtime.synchronize() 
        t1 = time.time()
        epoch_time = t1 - t0
        print(f"Epoch {i}: {epoch_time:.6f} s")
        
        # 仅统计 Warmup 之后的耗时
        if i >= warmup:
            total_time += epoch_time
            
    avg_time = total_time / (num_runs - warmup)
    print(f"\n==> Average Time (exclude warmup): {avg_time:.6f} seconds")

if __name__ == "__main__":
    main()