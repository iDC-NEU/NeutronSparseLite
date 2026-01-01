# NeutronSparse

## 0. Environment Setup
We use **openEuler 2.10.11** as the operating system and **CANN Toolkit 8.2.rc1** for the Ascend backend.

The Python packages and corresponding versions required for NeutronSparse are as follows:

| Package | Version |
| :--- | :--- |
| **Python** | 3.11 |
| **MindSpore** | 2.7.0.rc1 |

## 1. Compiling
Before running the project, you must compile the custom C++ operators.

Enter the compilation directory

```cd prepare/sgt_cpp```

Clean previous build artifacts

```rm -rf build```

Compile the extension in-place

```python setup.py build_ext --inplace```

## 2. Get Dataset
Navigate to the data directory and extract the dataset archive.

Go to the data directory (assuming you are currently in prepare/sgt_cpp)

```cd ../data_mtx```

Extract the data

```tar -zxvf data.tar.gz```

## 3. Preprocess Data
After extracting the data, navigate to the prepare folder to run the preprocessing scripts.

Go back to the ```prepare``` folder

1. Convert MTX format to COO format

   ```python mtx2coo.py```

2. Reorder the graph (Replace 'reddit' with your dataset name)

   ```bash graph_reorder.sh reddit```

3. Generate CSR format

   ```python generate_csr.py```

4. Prepare data for NeutronSparse

   ```python NeutronSparsePrepare.py```


## 4. Run NeutronSparse
Once compilation and data preparation are complete, you can run the main program.

Go back to the project root, then enter the run folder

```cd ../NeutronSparseLite```

Quick start the training/inference

```python NeutronSparseLiteRun.py```