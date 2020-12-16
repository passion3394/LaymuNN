# LaymuNN
All NN models implemented by Laymu

# train
distributed training on multi GPUs with cifar10 datasets

`CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch --nproc_per_node=4 train.py`


The train.py will download the cifar10 dataset *automatically*.
