#!/bin/bash

# 检查是否提供了参数
if [ $# -eq 0 ]; then
  echo "Usage: $0 <graph_name>"
  exit 1
fi

# 获取输入的 graph_name 参数
graph_name=$1

# 构建文件路径
coo_file="./data_coo/${graph_name}/${graph_name}_coo.txt"
output_file="./data_reorder/${graph_name}/${graph_name}_coo_reorder.txt"

# 检查输入文件是否存在
if [ ! -f "$coo_file" ]; then
  echo "Error: File $coo_file does not exist!"
  exit 1
fi

# 创建输出目录（如果不存在的话）
mkdir -p "$(dirname "$output_file")"

# 执行 reorder 命令，并将输出重定向到指定的文件
./rabbit_order/demo/reorder "$coo_file" > "$output_file"

echo "Reordered graph saved to $output_file"