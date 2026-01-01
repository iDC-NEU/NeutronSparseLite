
-------------------------------------COMPLIE CPP--------------------------------------------------
Execute:
    cd sgt_cpp 
And read the README in sgt_cpp

---------------------------------------READ MTX---------------------------------------------------
Must prepare your data.mtx in ./data_mtx and execute:

    python mtx2csr.py
    python NeutronSparsePrepare.py

-------------------------------------REORDER GRAPH------------------------------------------------
Reorder graph require complie rabbit_order，so cd rabbit_order and complie it meet the requirement
if you complie rabbit_order complete：

    python mtx2coo.py
    bash graph_reoder.sh graph_name
    
The graph_name is parameter，such as reddit

