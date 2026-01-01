#include <torch/extension.h>
#include <vector>
#include <string.h>
#include <cstdlib>
#include <map>
#include <iostream>

#define min(x, y) (((x) < (y))? (x) : (y))

// condense a sorted array with duplication: [1,2,2,3,4,5,5]
// after condense, it becomes: [1,2,3,4,5].
// Also, mapping the origin value to the corresponding new location in the new array.
// 1->[0], 2->[1], 3->[2], 4->[3], 5->[4]. 
std::map<unsigned, unsigned> inplace_deduplication(unsigned* array, unsigned length){
    int loc=0, cur=1;
    std::map<unsigned, unsigned> nb2col;
    nb2col[array[0]] = 0;
    while (cur < length){
        if(array[cur] != array[cur - 1]){
            loc++;
            array[loc] = array[cur];
            nb2col[array[cur]] = loc;       // mapping from eid to TC_block column index.[]
        }
        cur++;
    }
    return nb2col;
}

void preprocess(torch::Tensor edgeList_tensor, 
                torch::Tensor nodePointer_tensor, 
                int num_rows, 
                int blockSize_h,
                int blockSize_w,
                torch::Tensor blockPartition_tensor, 
                torch::Tensor edgeToColumn_tensor,
                torch::Tensor edgeToRow_tensor
                ){
    // printf("mask 1\n");
    // input tensors.
    auto edgeList = edgeList_tensor.accessor<int, 1>();
    auto nodePointer = nodePointer_tensor.accessor<int, 1>();
    // printf("mask 2\n");
    // output tensors.
    auto blockPartition = blockPartition_tensor.accessor<int, 1>();
    auto edgeToColumn = edgeToColumn_tensor.accessor<int, 1>();
    auto edgeToRow = edgeToRow_tensor.accessor<int, 1>();
    // printf("mask 3\n");
    unsigned block_counter = 0;

    #pragma omp parallel for 
    for (unsigned nid = 0; nid < num_rows; nid++){
        for (unsigned eid = nodePointer[nid]; eid < nodePointer[nid+1]; eid++)
            edgeToRow[eid] = nid;
    }
    // printf("mask 4\n");
    #pragma omp parallel for reduction(+:block_counter)
    for (unsigned iter = 0; iter < num_rows + 1; iter +=  blockSize_h){
        unsigned windowId = iter / blockSize_h;
        unsigned block_start = nodePointer[iter];
        unsigned block_end = nodePointer[min(iter + blockSize_h, num_rows)];
        unsigned num_window_edges = block_end - block_start;
        unsigned *neighbor_window = (unsigned *) malloc (num_window_edges * sizeof(unsigned));
        memcpy(neighbor_window, &edgeList[block_start], num_window_edges * sizeof(unsigned));

        // Step-1: Sort the neighbor id array of a row window.
        std::sort(neighbor_window, neighbor_window + num_window_edges);  // 替换为 std::sort

        // Step-2: Deduplication of the edge id array.
        std::map<unsigned, unsigned> clean_edges2col = inplace_deduplication(neighbor_window, num_window_edges);

        // generate blockPartition --> number of TC_blcok in each row window.
        blockPartition[windowId] = (clean_edges2col.size() + blockSize_w - 1) /blockSize_w;
        block_counter += blockPartition[windowId];

        // scan the array and generate edge to column mapping. --> edge_id to compressed_column_id of TC_block.
        for (unsigned e_index = block_start; e_index < block_end; e_index++){
            unsigned eid = edgeList[e_index];
            edgeToColumn[e_index] = clean_edges2col[eid];
        }
    }
    printf("TC_Blocks:\t%d\nExp_Edges:\t%d\n", block_counter, block_counter * blockSize_h * blockSize_w);
}

// unsigned compute_similarity(const torch::Tensor& nodePointer_tensor, unsigned row1, unsigned row2) {
//     auto nodePointer = nodePointer_tensor.accessor<int, 1>();
//     unsigned start1 = nodePointer[row1];
//     unsigned end1 = nodePointer[row1 + 1];
//     unsigned start2 = nodePointer[row2];
//     unsigned end2 = nodePointer[row2 + 1];

//     // 使用集合交集来计算相似度
//     std::vector<unsigned> neighbors1(end1 - start1);
//     std::vector<unsigned> neighbors2(end2 - start2);

//     for (unsigned i = start1; i < end1; ++i) {
//         neighbors1[i - start1] = i;
//     }
//     for (unsigned i = start2; i < end2; ++i) {
//         neighbors2[i - start2] = i;
//     }

//     // 计算交集的大小
//     unsigned intersection_size = 0;
//     for (unsigned i = 0; i < neighbors1.size(); ++i) {
//         for (unsigned j = 0; j < neighbors2.size(); ++j) {
//             if (neighbors1[i] == neighbors2[j]) {
//                 intersection_size++;
//             }
//         }
//     }
//     return intersection_size;
// }

// // 主函数
// void sort_rows_based_on_similarity(
//     torch::Tensor edgeList_tensor,
//     torch::Tensor nodePointer_tensor,
//     torch::Tensor nodeArray_tensor,
//     torch::Tensor Sort_edgeList_tensor,
//     torch::Tensor Sort_nodePointer_tensor,
//     torch::Tensor Sort_nodeArray_tensor,
//     torch::Tensor Row_Part
// ) {
//     auto edgeList = edgeList_tensor.accessor<int, 1>();
//     auto nodePointer = nodePointer_tensor.accessor<int, 1>();
//     auto nodeArray = nodeArray_tensor.accessor<int, 1>();

//     auto Sort_edgeList = Sort_edgeList_tensor.accessor<int, 1>();
//     auto Sort_nodePointer = Sort_nodePointer_tensor.accessor<int, 1>();
//     auto Sort_nodeArray = Sort_nodeArray_tensor.accessor<int, 1>();

//     int num_rows = nodeArray.size(0);

//     // Step 1: 计算每对行之间的相似度并排序
//     std::vector<std::tuple<unsigned, unsigned, unsigned>> row_similarities;  // (行号1, 行号2, 相似度)
    
//     #pragma omp parallel for
//     for (unsigned i = 0; i < num_rows; ++i) {
//         for (unsigned j = i + 1; j < num_rows; ++j) {
//             unsigned similarity = compute_similarity(nodePointer_tensor, i, j);
//             row_similarities.push_back(std::make_tuple(i, j, similarity));
//         }
//     }

//     // Step 2: 根据相似度排序行对
//     std::sort(row_similarities.begin(), row_similarities.end(), [](const auto& a, const auto& b) {
//         return a.second > b.second;  // 按照相似度降序排列
//     });

//     // Step 3: 根据相似度将行划分到不同的部分
//     unsigned part_id = 0;
//     std::vector<unsigned> row_part(num_rows, -1);  // 初始化为 -1，表示尚未分配到任何部分
//     Row_Part.fill_(0);  // 清空 Row_Part

//     for (const auto& [row1, row2, similarity] : row_similarities) {
//         if (row_part[row1] == -1 && row_part[row2] == -1) {
//             // 如果这两行都没有分配到部分，则将它们放入同一部分
//             row_part[row1] = part_id;
//             row_part[row2] = part_id;
//             Row_Part[part_id] += 2;
//             part_id++;
//         } else if (row_part[row1] != -1 && row_part[row2] == -1) {
//             row_part[row2] = row_part[row1];  // 将相似的行放入同一部分
//             Row_Part[row_part[row1]]++;
//         } else if (row_part[row2] != -1 && row_part[row1] == -1) {
//             row_part[row1] = row_part[row2];  // 将相似的行放入同一部分
//             Row_Part[row_part[row2]]++;
//         }
//     }

//     // Step 4: 重排节点（行）
//     unsigned idx = 0;
//     for (unsigned i = 0; i < num_rows; ++i) {
//         Sort_nodeArray[idx] = nodeArray[i];
//         Sort_nodePointer[idx] = nodePointer[i];
//         idx++;
//     }

//     // Step 5: 重排边列表
//     for (unsigned i = 0; i < edgeList_tensor.size(0); ++i) {
//         Sort_edgeList[i] = edgeList[i];
//     }

//     // 最后打印出行排序后的结果（可选）
//     std::cout << "Sorted Edge List: ";
//     for (unsigned i = 0; i < Sort_edgeList.size(0); ++i) {
//         std::cout << Sort_edgeList[i] << " ";
//     }
//     std::cout << std::endl;

//     std::cout << "Sorted Node Pointer: ";
//     for (unsigned i = 0; i < Sort_nodePointer.size(0); ++i) {
//         std::cout << Sort_nodePointer[i] << " ";
//     }
//     std::cout << std::endl;

//     std::cout << "Row Part: ";
//     for (unsigned i = 0; i < Row_Part.size(0); ++i) {
//         std::cout << Row_Part[i] << " ";
//     }
//     std::cout << std::endl;
// }

void preprocess_split_row(
    torch::Tensor A_edgeList_tensor,
    torch::Tensor A_nodePointer_tensor,
    int row_split_threshold,
    torch::Tensor& A1_edgeList_tensor,
    torch::Tensor& A1_nodePointer_tensor,
    torch::Tensor& A1_origin_row_index,
    torch::Tensor& A2_edgeList_tensor,
    torch::Tensor& A2_nodePointer_tensor,
    torch::Tensor& A2_origin_row_index) {
    
    // Input tensors.
    auto A_edgeList = A_edgeList_tensor.accessor<int, 1>();
    auto A_nodePointer = A_nodePointer_tensor.accessor<int, 1>();
    int num_rows = A_nodePointer.size(0) - 1;

    // Resize output tensors with an initial guess (will be resized again if needed).
    A1_edgeList_tensor.resize_({A_edgeList_tensor.size(0)});
    A1_nodePointer_tensor.resize_({num_rows + 1});
    A1_origin_row_index.resize_({num_rows});
    A2_edgeList_tensor.resize_({A_edgeList_tensor.size(0)});
    A2_nodePointer_tensor.resize_({num_rows + 1});
    A2_origin_row_index.resize_({num_rows});

    // Accessors for output tensors.
    auto A1_edgeList = A1_edgeList_tensor.accessor<int, 1>();
    auto A1_nodePointer = A1_nodePointer_tensor.accessor<int, 1>();
    auto A1_origin_index = A1_origin_row_index.accessor<int, 1>();
    auto A2_edgeList = A2_edgeList_tensor.accessor<int, 1>();
    auto A2_nodePointer = A2_nodePointer_tensor.accessor<int, 1>();
    auto A2_origin_index = A2_origin_row_index.accessor<int, 1>();

    // Initialize node pointers.
    A1_nodePointer[0] = 0;
    A2_nodePointer[0] = 0;

    int A1_edge_count = 0;
    int A2_edge_count = 0;
    int A1_row_count = 0;
    int A2_row_count = 0;

    // // #pragma omp parallel for
    // for (int row = 0; row < num_rows; ++row) {
    //     int row_start = A_nodePointer[row];
    //     int row_end = A_nodePointer[row + 1];
    //     int num_nonzeros = row_end - row_start;

    //     if (num_nonzeros > row_split_threshold) {
    //         // #pragma omp critical
    //         {
    //             A1_origin_index[A1_row_count] = row;
    //             A1_nodePointer[A1_row_count + 1] = A1_nodePointer[A1_row_count] + num_nonzeros;
    //             for (int idx = row_start; idx < row_end; ++idx) {
    //                 A1_edgeList[A1_edge_count++] = A_edgeList[idx];
    //             }
    //             A1_row_count++;
    //         }
    //     } else {
    //         // #pragma omp critical
    //         {
    //             A2_origin_index[A2_row_count] = row;
    //             A2_nodePointer[A2_row_count + 1] = A2_nodePointer[A2_row_count] + num_nonzeros;
    //             for (int idx = row_start; idx < row_end; ++idx) {
    //                 A2_edgeList[A2_edge_count++] = A_edgeList[idx];
    //             }
    //             A2_row_count++;
    //         }
    //     }
    // }
    for (int row = 0; row < num_rows; ++row) {
        int row_start = A_nodePointer[row];
        int row_end = A_nodePointer[row + 1];
        int num_nonzeros = row_end - row_start;
        if (num_nonzeros > row_split_threshold) {
            A1_origin_index[A1_row_count] = row;
            A1_nodePointer[A1_row_count + 1] = A1_nodePointer[A1_row_count] + num_nonzeros;
            A1_row_count++;
        } else {
            A2_origin_index[A2_row_count] = row;
            A2_nodePointer[A2_row_count + 1] = A2_nodePointer[A2_row_count] + num_nonzeros;
            A2_row_count++;
        }
    }

    // 根据 A1_origin_index 和 A_edgeList 生成 A1_edgeList
    A1_edge_count = 0;
    for (int i = 0; i < A1_row_count; ++i) {
        int row = A1_origin_index[i];
        int row_start = A_nodePointer[row];
        int row_end = A_nodePointer[row + 1];
        for (int idx = row_start; idx < row_end; ++idx) {
            A1_edgeList[A1_edge_count++] = A_edgeList[idx];
        }
    }

    // 根据 A2_origin_index 和 A_edgeList 生成 A2_edgeList
    A2_edge_count = 0;
    for (int i = 0; i < A2_row_count; ++i) {
        int row = A2_origin_index[i];
        int row_start = A_nodePointer[row];
        int row_end = A_nodePointer[row + 1];
        for (int idx = row_start; idx < row_end; ++idx) {
            A2_edgeList[A2_edge_count++] = A_edgeList[idx];
        }
    }

    // Resize output tensors to actual sizes.
    A1_edgeList_tensor.resize_({A1_edge_count});
    A1_nodePointer_tensor.resize_({A1_row_count + 1});
    A1_origin_row_index.resize_({A1_row_count});
    A2_edgeList_tensor.resize_({A2_edge_count});
    A2_nodePointer_tensor.resize_({A2_row_count + 1});
    A2_origin_row_index.resize_({A2_row_count});

    // Print final results for verification.
    // std::cout << "A1_edgeList size: " << A1_edgeList_tensor.size(0) << std::endl;
    // std::cout << "A1_nodePointer size: " << A1_nodePointer_tensor.size(0) << std::endl;
    // std::cout << "A2_edgeList size: " << A2_edgeList_tensor.size(0) << std::endl;
    // std::cout << "A2_nodePointer size: " << A2_nodePointer_tensor.size(0) << std::endl;
}

void preprocess_split_col(
    torch::Tensor A1_edgeList_tensor,
    torch::Tensor A1_nodePointer_tensor,
    int col_split_threshold,
    torch::Tensor& A11_edgeList_tensor,
    torch::Tensor& A11_nodePointer_tensor,
    torch::Tensor& A11_origin_col_index,
    torch::Tensor& A12_edgeList_tensor,
    torch::Tensor& A12_nodePointer_tensor,
    torch::Tensor& A12_origin_col_index) {
    
    // Input tensors.
    auto A1_edgeList = A1_edgeList_tensor.accessor<int, 1>();
    auto A1_nodePointer = A1_nodePointer_tensor.accessor<int, 1>();
    int num_rows = A1_nodePointer.size(0) - 1;

    // Calculate the number of columns.
    int num_cols = 0;
    for (int i = 0; i < A1_edgeList_tensor.size(0); ++i) {
        if (A1_edgeList[i] > num_cols) {
            num_cols = A1_edgeList[i];
        }
    }
    num_cols += 1; // Since columns are zero-indexed.

    // Create column-wise count of non-zeros.
    std::vector<int> col_nonzeros(num_cols, 0);
    for (int row = 0; row < num_rows; ++row) {
        for (int idx = A1_nodePointer[row]; idx < A1_nodePointer[row + 1]; ++idx) {
            int col = A1_edgeList[idx];
            #pragma omp atomic
            col_nonzeros[col]++;
        }
    }

    // Resize output tensors with an initial guess (will be resized again if needed).
    A11_edgeList_tensor.resize_({A1_edgeList_tensor.size(0)});
    A11_nodePointer_tensor.resize_({num_rows + 1});
    A11_origin_col_index.resize_({num_cols});
    A12_edgeList_tensor.resize_({A1_edgeList_tensor.size(0)});
    A12_nodePointer_tensor.resize_({num_rows + 1});
    A12_origin_col_index.resize_({num_cols});

    // Accessors for output tensors.
    auto A11_edgeList = A11_edgeList_tensor.accessor<int, 1>();
    auto A11_nodePointer = A11_nodePointer_tensor.accessor<int, 1>();
    auto A11_origin_index = A11_origin_col_index.accessor<int, 1>();
    auto A12_edgeList = A12_edgeList_tensor.accessor<int, 1>();
    auto A12_nodePointer = A12_nodePointer_tensor.accessor<int, 1>();
    auto A12_origin_index = A12_origin_col_index.accessor<int, 1>();

    // Initialize node pointers.
    A11_nodePointer[0] = 0;
    A12_nodePointer[0] = 0;

    int A11_edge_count = 0;
    int A12_edge_count = 0;
    int A11_col_count = 0;
    int A12_col_count = 0;

    // Split columns based on non-zero count threshold.
    for (int col = 0; col < num_cols; ++col) {
        if (col_nonzeros[col] <= col_split_threshold) {
            A12_origin_index[A12_col_count++] = col;
        } else {
            A11_origin_index[A11_col_count++] = col;
        }
    }

    // Create new CSR representation for A11 and A12.
    A11_nodePointer[0] = 0;
    A12_nodePointer[0] = 0;
    int current_A11_edge_count = 0;
    int current_A12_edge_count = 0;

    for (int row = 0; row < num_rows; ++row) {
        int row_start = A1_nodePointer[row];
        int row_end = A1_nodePointer[row + 1];

        for (int idx = row_start; idx < row_end; ++idx) {
            int col = A1_edgeList[idx];
            if (col_nonzeros[col] <= col_split_threshold) {
                A12_edgeList[current_A12_edge_count++] = col;
            } else {
                A11_edgeList[current_A11_edge_count++] = col;
            }
        }

        A11_nodePointer[row + 1] = current_A11_edge_count;
        A12_nodePointer[row + 1] = current_A12_edge_count;
    }

    // Resize output tensors to actual sizes.
    A11_edgeList_tensor.resize_({current_A11_edge_count});
    A11_nodePointer_tensor.resize_({num_rows + 1});
    A11_origin_col_index.resize_({A11_col_count});
    A12_edgeList_tensor.resize_({current_A12_edge_count});
    A12_nodePointer_tensor.resize_({num_rows + 1});
    A12_origin_col_index.resize_({A12_col_count});

    // Print final results for verification.
    // std::cout << "A11_edgeList size: " << A11_edgeList_tensor.size(0) << std::endl;
    // std::cout << "A11_nodePointer size: " << A11_nodePointer_tensor.size(0) << std::endl;
    // std::cout << "A12_edgeList size: " << A12_edgeList_tensor.size(0) << std::endl;
    // std::cout << "A12_nodePointer size: " << A12_nodePointer_tensor.size(0) << std::endl;
}

// 利用矩阵对称性
void preprocess_split_col_symmetry(
    torch::Tensor A1_edgeList_tensor,
    torch::Tensor A1_nodePointer_tensor,
    int col_split_threshold,
    torch::Tensor& A11_edgeList_tensor,
    torch::Tensor& A11_nodePointer_tensor,
    torch::Tensor& A11_origin_col_index,
    torch::Tensor& A12_edgeList_tensor,
    torch::Tensor& A12_nodePointer_tensor,
    torch::Tensor& A12_origin_col_index) {
    
// Access the input CSR format tensors.
    auto A1_edgeList = A1_edgeList_tensor.accessor<int, 1>();
    auto A1_nodePointer = A1_nodePointer_tensor.accessor<int, 1>();
    int num_rows = A1_nodePointer.size(0) - 1;

    // Estimate the number of columns indirectly since the matrix is symmetric
    int num_cols = num_rows;

    // Create column-wise count of non-zeros using A1_nodePointer_tensor directly.
    std::vector<int> col_nonzeros(num_cols, 0);

    // Step 1: Use row pointers to count non-zero elements for each column.
    #pragma omp parallel for
    for (int row = 0; row < num_rows; ++row) {
        col_nonzeros[row] = A1_nodePointer[row + 1] - A1_nodePointer[row];
    }

    // Resize output tensors with an initial guess (will be resized again if needed).
    A11_edgeList_tensor.resize_({A1_edgeList_tensor.size(0)});
    A11_nodePointer_tensor.resize_({num_rows + 1});
    A11_origin_col_index.resize_({num_cols});
    A12_edgeList_tensor.resize_({A1_edgeList_tensor.size(0)});
    A12_nodePointer_tensor.resize_({num_rows + 1});
    A12_origin_col_index.resize_({num_cols});

    // Accessors for output tensors.
    auto A11_edgeList = A11_edgeList_tensor.accessor<int, 1>();
    auto A11_nodePointer = A11_nodePointer_tensor.accessor<int, 1>();
    auto A11_origin_index = A11_origin_col_index.accessor<int, 1>();
    auto A12_edgeList = A12_edgeList_tensor.accessor<int, 1>();
    auto A12_nodePointer = A12_nodePointer_tensor.accessor<int, 1>();
    auto A12_origin_index = A12_origin_col_index.accessor<int, 1>();

    // Initialize node pointers.
    A11_nodePointer[0] = 0;
    A12_nodePointer[0] = 0;

    int A11_edge_count = 0;
    int A12_edge_count = 0;
    int A11_col_count = 0;
    int A12_col_count = 0;

    // Split columns based on non-zero count threshold.
    for (int col = 0; col < num_cols; ++col) {
        if (col_nonzeros[col] <= col_split_threshold) {
            A12_origin_index[A12_col_count++] = col;
        } else {
            A11_origin_index[A11_col_count++] = col;
        }
    }

    // Create new CSR representation for A11 and A12.
    A11_nodePointer[0] = 0;
    A12_nodePointer[0] = 0;
    int current_A11_edge_count = 0;
    int current_A12_edge_count = 0;

    for (int row = 0; row < num_rows; ++row) {
        int row_start = A1_nodePointer[row];
        int row_end = A1_nodePointer[row + 1];

        for (int idx = row_start; idx < row_end; ++idx) {
            int col = A1_edgeList[idx];
            if (col_nonzeros[col] <= col_split_threshold) {
                A12_edgeList[current_A12_edge_count++] = col;
            } else {
                A11_edgeList[current_A11_edge_count++] = col;
            }
        }

        A11_nodePointer[row + 1] = current_A11_edge_count;
        A12_nodePointer[row + 1] = current_A12_edge_count;
    }

    // Resize output tensors to actual sizes.
    A11_edgeList_tensor.resize_({current_A11_edge_count});
    A11_nodePointer_tensor.resize_({num_rows + 1});
    A11_origin_col_index.resize_({A11_col_count});
    A12_edgeList_tensor.resize_({current_A12_edge_count});
    A12_nodePointer_tensor.resize_({num_rows + 1});
    A12_origin_col_index.resize_({A12_col_count});

    // Print final results for verification.
    // std::cout << "A11_edgeList size: " << A11_edgeList_tensor.size(0) << std::endl;
    // std::cout << "A11_nodePointer size: " << A11_nodePointer_tensor.size(0) << std::endl;
    // std::cout << "A12_edgeList size: " << A12_edgeList_tensor.size(0) << std::endl;
    // std::cout << "A12_nodePointer size: " << A12_nodePointer_tensor.size(0) << std::endl;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("sort_rows_based_on_similarity", &sort_rows_based_on_similarity, "sort_rows_based_on_similarity");
    m.def("preprocess", &preprocess, "Preprocess Step (CPU)");
    m.def("preprocess_split_row", &preprocess_split_row, "Split Rows of Sparse Matrix (CPU)",
          py::arg("A_edgeList_tensor"), 
          py::arg("A_nodePointer_tensor"), 
          py::arg("row_split_threshold"),
          py::arg("A1_edgeList_tensor"), 
          py::arg("A1_nodePointer_tensor"), 
          py::arg("A1_origin_row_index"),
          py::arg("A2_edgeList_tensor"), 
          py::arg("A2_nodePointer_tensor"), 
          py::arg("A2_origin_row_index"));
    m.def("preprocess_split_col", &preprocess_split_col, "Preprocess Split Columns (CSR)",
          py::arg("A1_edgeList_tensor"),
          py::arg("A1_nodePointer_tensor"),
          py::arg("col_split_threshold"),
          py::arg("A11_edgeList_tensor"),
          py::arg("A11_nodePointer_tensor"),
          py::arg("A11_origin_col_index"),
          py::arg("A12_edgeList_tensor"),
          py::arg("A12_nodePointer_tensor"),
          py::arg("A12_origin_col_index"));
    m.def("preprocess_split_col_symmetry", &preprocess_split_col_symmetry, "Preprocess Split Columns (CSR)",
          py::arg("A1_edgeList_tensor"),
          py::arg("A1_nodePointer_tensor"),
          py::arg("col_split_threshold"),
          py::arg("A11_edgeList_tensor"),
          py::arg("A11_nodePointer_tensor"),
          py::arg("A11_origin_col_index"),
          py::arg("A12_edgeList_tensor"),
          py::arg("A12_nodePointer_tensor"),
          py::arg("A12_origin_col_index"));
}


