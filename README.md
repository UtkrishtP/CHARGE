# CHARGE: Leveraging competitive <ins>C</ins>PU sampling in a CPU-GPU <ins>H</ins>eterogenous environment to <ins>A</ins>ccele<ins>r</ins>ate end-to-end <ins>G</ins>NN training using an <ins>E</ins>lastic pipeline

## ABSTRACT

Graph Neural Networks (GNNs) have demonstrated exceptional performance across a wide range of applications, driving their widespread adoption. Current frameworks employ CPU and GPU resources—either in isolation or heterogeneously—to train GNNs, incorporating mini-batching and sampling techniques to mitigate scalability challenges posed by limited GPU memory. Sample-based GNN training is divided into three phases: Sampling, Extraction, and Training. Existing systems orchestrate these tasks across CPU and GPU in various ways, but exhaustive experiments reveal that not every stage is equally suited to both processors; notably, CPU sampling can outperform GPU sampling for certain samplers. Moreover, most frameworks lack adaptability to different samplers, datasets, and hardware configurations. 
In this thesis, we propose CHARGE, a system that leverages competitive CPU sampling to accelerate end-to-end GNN training. An intelligent controller assigns each stage—Sampling, Extraction, and Training—to the most appropriate processor (CPU or GPU), agnostic to sampler, dataset, batch size, model, or underlying hardware. Built atop the DGL framework, CHARGE retains ease of programmability while delivering substantial improvements over state-of-the-art systems across multiple samplers, datasets, and models.

## System Requirements

- CPU DRAM >= 256G
- CUDA v11.7
- gcc-10
- g++-10 
- Python 3.11.4
- Cmake 3.5

## INSTALLATION GUIDE

We have modified both DGL and PyTorch and have added them as submodules in this repo

Follow the general guidelines for installing DGL and Pytorch from source, but before we will create a conda environment to isolate our workspace for a clean build and development

(Install miniconda if not [available](https://www.anaconda.com/docs/getting-started/miniconda/install))
1. Create conda environment with python 3.11.4

```
$ conda create -n charge -m python==3.11.4 -y
```

2. Install these dependencies explicitly via pip:
```
$ pip install ogb pynvml tabulate colorama
$ pip install "numpy<2"
```

3. Clone charge
```
$ git clone git@github.com:UtkrishtP/CHARGE.git
$ cd CHARGE
```

### DGL source install

4. Below we have added guidelines to install DGL, though the method is similar to the one mentioned on official DGL repo.
```
$ cd dgl
$ mkdir build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DUSE_LIBXSMM=ON -DUSE_CUDA=ON -DBUILD_TORCH=ON -DBUILD_SPARSE=OFF -DUSE_AVX=ON ..
$ make -j
$ cd ../python
$ pip install -e .

```

 <!-- 
 -DCMAKE_C_COMPILER=/usr/bin/gcc-10 -DCMAKE_CXX_COMPILER=/usr/bin/g++-10 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc -DCMAKE_POLICY_VERSION_MINIMUM=3.5
-->

### PyTorch source install

Follow the official guidelines for source installing torch in the above attached [submodule](https://github.com/UtkrishtP/torch_repo/tree/c88d6f58051f6fe93a6e870b137f3c265d31f417)

### Increase ulimits

We need to increase the ulimits to the max configurable value in our system, follow this [link]() to configure limits as per you system
```
$ sudo vi /etc/security/limits.conf
```

we need to edit the above file with the following lines: (replace the value `1048576` with the max as per your system)
```
    * soft nofile 1048576
    * hard nofile 1048576
```
apply changes to take effect:
```
$ sudo sysctl fs.file-max=1048576
```

We use shared memory (`\dev\shm`) extensively to load graph, nfeats and cpu_queue, gpu_queue. Hence, mount complete DRAM as shared memory. Below is a command for temp remounting, check the [link](https://stackoverflow.com/questions/58804022/how-to-resize-dev-shm) for more details
```
$ sudo mount -o remount,size=<total-dram-size>G /dev/shm
```
## Datasets download

We use the following datasets in our experiments: (We have attached their repective git repos and download links)
- [ogbn-papers100M](https://github.com/snap-stanford/ogb)
- [IGB](https://github.com/IllinoisGraphBenchmark/IGB-Datasets)
- [Twitter](https://github.com/vigna/webgraph/)
- [Friendster](https://github.com/SJTU-IPADS/fgnn-artifacts)

The below file contains our scripts to load the files as part of `dgl_repo`, it can be customized as per your requirement:
```
examples/pytorch/graphsage/hybrid_101/custom_dl.py
```

## Running CHARGE

1. Load dataset in shared memory
```
$ cd /dgl_repo/examples/pytorch/graphsage/hybrid_101/
$ python load_dataset.py --dataset ogbn-papers100M #change the name accordingly
```

2. Run CHARGE based on the type of:
    - sampler
    - batch size
    - fan-out
    - model_type
    - hidden size
    - epochs
    - nfeat_dim

An example run
```
$ python charge.py --epoch 20 --batch_size 1024 --mfg_size 0 --dataset igb-large --cache_size 60449000 --sampler shadow --workers 64 --num_threads 64 --variant hybrid --hid_size 256 --diff 0 --fan_out 15,10 --slack_test 0 --nfeat_dim 128 --hybrid 1 --model_type gcn --mfg_buffer_size 0 --opt 0 --ablation 0
```