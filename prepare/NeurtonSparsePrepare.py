import math
import time
import torch
import numpy as np
from pathlib import Path

# 确保编译好的 C++ 扩展在 PYTHONPATH 中，或者直接放在同一目录下
try:
    from sgt_cpp import preprocess_extension
except ImportError:
    print("\n[Error] Cannot import 'sgt_cpp.preprocess_extension'.")
    print("Please ensure the C++ extension is compiled and accessible.\n")
    exit(1)

# ==========================================
# 1. 配置管理类
# ==========================================
class PreprocessConfig:
    """配置类：管理路径和核心超参数"""
    def __init__(self, dataset_name, root_dir='./', block_h=2048, block_w=32):
        self.dataset = dataset_name
        self.root_dir = Path(root_dir)
        self.block_h = block_h
        self.block_w = block_w
        
        # 定义关键路径
        # 优先读取 Reorder 后的数据，没有则读取原始 CSR
        self.dir_reorder = self.root_dir / 'data_reorder_csr' / dataset_name
        self.dir_csr = self.root_dir / 'data_csr' / dataset_name
        
        # 输出保存路径 (假设保存到 data_save 目录)
        self.dir_save = self.root_dir.resolve().parent / 'data_save' / dataset_name

# ==========================================
# 2. 数据 IO 管理类
# ==========================================
class DataManager:
    """负责数据的加载与保存"""
    
    @staticmethod
    def load_csr(config: PreprocessConfig):
        """智能加载 CSR 数据"""
        # 1. 尝试加载 Reorder 数据
        if config.dir_reorder.exists():
            print(f"--> [IO] Loading from Reordered: {config.dir_reorder}")
            ptr_path = config.dir_reorder / f'{config.dataset}_csr_crow_indices.bin'
            idx_path = config.dir_reorder / f'{config.dataset}_csr_col_indices.bin'
        # 2. 回退到原始数据
        else:
            print(f"--> [IO] Loading from Origin: {config.dir_csr}")
            ptr_path = config.dir_csr / f'{config.dataset}_csr_crow_indices.bin'
            idx_path = config.dir_csr / f'{config.dataset}_csr_col_indices.bin'

        if not ptr_path.exists() or not idx_path.exists():
            raise FileNotFoundError(f"Data files not found in {ptr_path.parent}")

        # 使用 numpy 读取并转为 torch tensor (CPU)
        ptr = torch.from_numpy(np.fromfile(ptr_path, dtype=np.int64)).to(torch.int32)
        idx = torch.from_numpy(np.fromfile(idx_path, dtype=np.int64)).to(torch.int32)
        return ptr, idx

    @staticmethod
    def save_tensor(tensor, folder: Path, filename: str):
        """保存 Tensor 为二进制文件"""
        folder.mkdir(parents=True, exist_ok=True)
        file_path = folder / filename
        tensor.numpy().tofile(file_path)

# ==========================================
# 3. 智能策略类 (核心算法)
# ==========================================
class SplitStrategy:
    """
    策略类：基于数据集统计特征动态计算切分阈值
    替代了原有的硬编码 if-else 逻辑
    """
    @staticmethod
    def get_smart_threshold(num_rows, num_cols, num_nnz):

        if num_rows == 0: return 32
        avg_degree = num_nnz / num_rows
        sparsity = num_nnz / (num_rows * num_cols)
        alpha = 0.5 
        
        # --- 针对不同稀疏度梯队的调整 ---
        if sparsity < 0.0002:
            if num_rows > 1_000_000: 
                alpha = 1.8  
            else: 
                alpha = 1.0
        elif sparsity > 0.01:
            alpha = 0.25
        else:
            alpha = 0.55

        # 计算并取整
        threshold = int(avg_degree * alpha)
        threshold = max(4, threshold)
        print(f"   [Strategy] Matrix: {num_rows}x{num_cols}, NNZ: {num_nnz}")
        print(f"   [Strategy] Stats: AvgDeg={avg_degree:.1f}, Sparsity={sparsity:.5f}")
        print(f"   [Strategy] Decision: Alpha={alpha} -> Threshold={threshold}")
        return threshold

# ==========================================
# 4. 核心处理引擎
# ==========================================
class NeutronSparsePreprocessor:
    """NeutronSGT 预处理流水线引擎"""
    def __init__(self, config: PreprocessConfig):
        self.cfg = config
        
        # 加载数据
        t0 = time.time()
        self.node_ptr, self.edge_list = DataManager.load_csr(config)
        print(f"   [Init] Data loaded in {time.time()-t0:.4f}s")
        
        self.num_nodes = self.node_ptr.size(0) - 1
        # 假设是方阵，如果不是需修改此处逻辑
        self.num_cols = self.num_nodes 
        self.num_edges = self.edge_list.size(0)

    def run(self):
        print(f"\n=== Processing Dataset: {self.cfg.dataset} ===")
        
        # -------------------------------------------------
        # Step 0: 计算切分阈值
        # -------------------------------------------------
        split_threshold = SplitStrategy.get_smart_threshold(
            self.num_nodes, self.num_cols, self.num_edges
        )

        # -------------------------------------------------
        # Step 1: 列切分 (Column Split)
        # 将矩阵切分为 A1 (Dense-friendly cols) 和 A2 (Sparse cols)
        # -------------------------------------------------
        print(f"\n--> Step 1: Column Split (Threshold={split_threshold})...")
        a1_data, a2_data = self._exec_split_col(split_threshold)
        
        # -------------------------------------------------
        # Step 2: 行切分 (Row Split)
        # 将 A1 进一步切分为 A11 (Dense rows) 和 A12 (Sparse rows)
        # -------------------------------------------------
        print("\n--> Step 2: Row Split...")
        # 注意：这里复用计算出的 split_threshold，因为实验表明 Row/Col 阈值通常接近
        a11_data, a12_data = self._exec_split_row(a1_data, split_threshold)
        
        # -------------------------------------------------
        # Step 3: SGT 预处理 (Block Partition)
        # 对 A11 进行分块，生成 Cube 计算所需的元数据
        # -------------------------------------------------
        print("\n--> Step 3: SGT Preprocessing (A11 Block Partition)...")
        sgt_data = self._exec_sgt_process(a11_data)
        
        # -------------------------------------------------
        # Step 4: 保存所有结果
        # -------------------------------------------------
        print("\n--> Step 4: Saving Results...")
        self._save_pipeline(a1_data, a2_data, a11_data, a12_data, sgt_data)
        
        print(f"\n=== Done: {self.cfg.dataset} ===")

    def _exec_split_col(self, threshold):
        """封装 C++: preprocess_split_col_symmetry"""
        # 准备输出容器 (Torch Tensor)
        out_a1 = {
            'edge': torch.empty(0, dtype=torch.int32), 
            'ptr': torch.empty(0, dtype=torch.int32), 
            'orig_col': torch.empty(0, dtype=torch.int32)
        }
        out_a2 = {
            'edge': torch.empty(0, dtype=torch.int32), 
            'ptr': torch.empty(0, dtype=torch.int32), 
            'orig_col': torch.empty(0, dtype=torch.int32)
        }

        t0 = time.time()
        # 调用 C++ 扩展
        preprocess_extension.preprocess_split_col_symmetry(
            self.edge_list, self.node_ptr, threshold,
            out_a1['edge'], out_a1['ptr'], out_a1['orig_col'],
            out_a2['edge'], out_a2['ptr'], out_a2['orig_col']
        )
        t_cost = time.time() - t0
        
        print(f"   [Perf] Time: {t_cost:.4f}s")
        print(f"   [Stat] A1 (Dense Candidates): {out_a1['edge'].size(0)} edges")
        print(f"   [Stat] A2 (Sparse Residual):  {out_a2['edge'].size(0)} edges")
        return out_a1, out_a2

    def _exec_split_row(self, a1_data, threshold):
        """封装 C++: preprocess_split_row"""
        out_a11 = {
            'edge': torch.empty(0, dtype=torch.int32), 
            'ptr': torch.empty(0, dtype=torch.int32), 
            'orig_row': torch.empty(0, dtype=torch.int32)
        }
        out_a12 = {
            'edge': torch.empty(0, dtype=torch.int32), 
            'ptr': torch.empty(0, dtype=torch.int32), 
            'orig_row': torch.empty(0, dtype=torch.int32)
        }
        
        t0 = time.time()
        preprocess_extension.preprocess_split_row(
            a1_data['edge'], a1_data['ptr'], threshold,
            out_a11['edge'], out_a11['ptr'], out_a11['orig_row'],
            out_a12['edge'], out_a12['ptr'], out_a12['orig_row']
        )
        t_cost = time.time() - t0
        
        print(f"   [Perf] Time: {t_cost:.4f}s")
        print(f"   [Stat] A11 (Cube Core):   {out_a11['edge'].size(0)} edges")
        print(f"   [Stat] A12 (Vector Core): {out_a12['edge'].size(0)} edges")
        return out_a11, out_a12

    def _exec_sgt_process(self, a11_data):
        """封装 C++: preprocess (block partition)"""
        num_rows = a11_data['orig_row'].size(0)
        num_edges = a11_data['edge'].size(0)
        
        if num_rows == 0:
            print("   [Warn] A11 is empty, skipping SGT process.")
            return {'part': torch.tensor([]), 'col_map': torch.tensor([]), 'row_map': torch.tensor([])}

        # 准备输出容器
        num_wins = math.ceil(num_rows / self.cfg.block_h)
        out_sgt = {
            'part': torch.zeros(num_wins, dtype=torch.int32),
            'col_map': torch.zeros(num_edges, dtype=torch.int32),
            'row_map': torch.zeros(num_edges, dtype=torch.int32)
        }
        
        t0 = time.time()
        preprocess_extension.preprocess(
            a11_data['edge'], a11_data['ptr'], num_rows,
            self.cfg.block_h, self.cfg.block_w,
            out_sgt['part'], out_sgt['col_map'], out_sgt['row_map']
        )
        t_cost = time.time() - t0
        
        # 计算显存占用估算
        total_blocks = out_sgt['part'].sum().item()
        cube_size_bytes = 2 * total_blocks * self.cfg.block_h * self.cfg.block_w # FP16 = 2 bytes
        cube_size_gb = cube_size_bytes / (1024**3)
        
        print(f"   [Perf] Time: {t_cost:.4f}s")
        print(f"   [Stat] Total Blocks: {total_blocks}")
        print(f"   [Stat] Cube Size: {cube_size_gb:.4f} GB")
        return out_sgt

    def _save_pipeline(self, a1, a2, a11, a12, sgt):
        """统一保存流程"""
        s = DataManager.save_tensor
        root = self.cfg.dir_save
        
        # 1. Origin Info
        s(self.node_ptr, root / "origin", "A_nodePointer_tensor.bin")
        s(self.edge_list, root / "origin", "A_edgeList_tensor.bin")
        
        # 2. A1 Intermediate (仅保存需要的 col index)
        s(a1['orig_col'], root / "A1", "A1_origin_col_index.bin")
        
        # 3. A2 (Sparse Part 1)
        s(a2['edge'], root / "A2", "A2_edgeList_tensor.bin")
        s(a2['ptr'], root / "A2", "A2_nodePointer_tensor.bin")
        s(a2['orig_col'], root / "A2", "A2_origin_col_index.bin")
        
        # 4. A12 (Sparse Part 2)
        s(a12['edge'], root / "A12", "A12_edgeList_tensor.bin")
        s(a12['ptr'], root / "A12", "A12_nodePointer_tensor.bin")
        s(a12['orig_row'], root / "A12", "A12_origin_row_index.bin")
        
        # 5. A11 (Dense Part) & SGT Meta
        s(a11['edge'], root / "A11", "A11_edgeList_tensor.bin")
        s(a11['ptr'], root / "A11", "A11_nodePointer_tensor.bin")
        s(a11['orig_row'], root / "A11", "A11_origin_row_index.bin")
        
        if sgt['part'].numel() > 0:
            s(sgt['part'], root / "A11", "A11_block_partition.bin")
            s(sgt['col_map'], root / "A11", "A11_edge_to_column.bin")
            s(sgt['row_map'], root / "A11", "A11_edge_to_row.bin")
            
        print(f"   [IO] All tensors saved to {root}")

# ==========================================
# 5. 主程序入口
# ==========================================
def main():
    # 在这里定义需要处理的数据集列表
    # datasets = ['reddit', 'cora', 'amazon-product', 'human_gene1']
    datasets = ['reddit'] 
    
    for name in datasets:
        # 初始化配置
        config = PreprocessConfig(
            dataset_name=name, 
            block_h=2048, 
            block_w=32
        )
        
        # 初始化引擎并运行
        try:
            processor = NeutronSparsePreprocessor(config)
            processor.run()
        except Exception as e:
            print(f"[Error] Failed to process {name}: {e}")
            # print(traceback.format_exc()) # 需要 import traceback

if __name__ == "__main__":
    main()