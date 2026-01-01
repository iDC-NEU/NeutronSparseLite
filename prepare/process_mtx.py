#!/usr/bin/env python3

import os

# 这里列出你需要处理的所有数据集（不含文件后缀），或者直接写全名也可以
datasets = ['olafu','mip1','dawson5','mycielskian15','mycielskian17','nd12k','human_gene1','pattern1',]

for dataset_name in datasets:
    # 若你的文件名为 dataset_name.mtx，可以用下面两行
    input_file = f"./data_mtx/{dataset_name}.mtx"
    output_file = f"./tmp/{dataset_name}.mtx"

    # 如果文件名和 dataset_name 不一样，请根据需要自行调整

    with open(input_file, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()

        # 1) 提取前三行（或更多，根据实际情况），其中第三行通常包含行数、列数和非零元数
        line0 = lines[0]  # %%MatrixMarket ...
        line1 = lines[1]  # 可能是注释行或留空行
        line2 = lines[2]  # e.g. "num_rows num_cols num_nonzeros"
        data_lines = lines[3:]  # 之后的行才是具体数据

        # 2) 解析第三行，获取行数、列数和旧的非零元数
        #    假设格式固定为三列：row_count col_count nonzero_count
        row_count, col_count, old_nonzeros = line2.strip().split()
        row_count = int(row_count)
        col_count = int(col_count)
        old_nonzeros = int(old_nonzeros)

        # 3) 处理数据行：将第三列改为1；如果只有两列，也补一个1
        #    并且如果 row != col，则再额外添加一条反向记录
        processed_entries = []  # 用于存储最终要写出的 (row, col) 对
        added_count = 0         # 记录新增的反向记录数

        for line in data_lines:
            parts = line.strip().split()
            if not parts:
                # 如果是空行，直接跳过或视情况处理
                continue
            
            if len(parts) == 2:
                r, c = parts
                val = '1'
            elif len(parts) == 3:
                r, c, _ = parts
                val = '1'
            else:
                # 如果格式异常，保留或跳过，这里选择保留
                processed_entries.append(line.rstrip('\n'))
                continue

            # 将 row, col, val 都转换为字符串，以便最后写出
            row_str = str(r)
            col_str = str(c)
            val_str = '1'

            processed_entries.append(f"{row_str} {col_str} {val_str}")

            # 如果第一列和第二列不相等，则增加一个反向记录
            # 但如果 row == col，就不需要
            if row_str != col_str:
                reversed_entry = f"{col_str} {row_str} {val_str}"
                processed_entries.append(reversed_entry)
                added_count += 1  # 统计反向记录的新增数

        # 4) 计算新的非零元数： 原始非零元数 + 新增反向记录数
        new_nonzeros = old_nonzeros + added_count

        # 5) 写出到新文件
        with open(output_file, 'w', encoding='utf-8') as fout:
            # 保留第一、二行原样
            fout.write(line0)
            fout.write(line1)
            # 第三行修改非零元数
            fout.write(f"{row_count} {col_count} {new_nonzeros}\n")
            # 输出修改后的数据行
            for entry in processed_entries:
                fout.write(entry + "\n")

        print(f"处理完成：{input_file} -> {output_file}")
        print(f"旧的非零元数: {old_nonzeros}, 新的非零元数: {new_nonzeros}")
        print("-------------------------------------------------")