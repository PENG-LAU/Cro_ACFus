# A Mamba-Based Cross-Adaptive CollaborativeFusion Algorithm On 3D Human Pose Stream
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

## Environment
The project is developed under the following environment:
- Python 3.10.16
- PyTorch 2.3.1
- CUDA 11.8
- Causal_conv1d 1.2.2
- Mamba_ssm 2.0.4

For installation of the project dependencies, please run:
```
pip install -r requirements.txt
``` 

## Human3.6M
### Preprocessing
1. Please download the dataset from [Human3.6M](http://vision.imar.ro/human3.6m/) website and [MPI-INF-3DHP](https://vcai.mpi-inf.mpg.de/3dhp-dataset/) website, and refer to [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) to set up the Human3.6M dataset ('./dataset' directory). Or you can download the processed data from [here](https://drive.google.com/drive/folders/18mvXIZ98LKGAqDFpRsNVvCRonVBAlgoX?usp=share_link). 

2. Slice the motion clips by running the following python code in `data/preprocess` directory.
```text
python h36m.py --n-frames 81
```

### Train
to train a model in Human 3.6M dataset:
``` bash
python train.py --config configs/h36m/yourmodel.yaml
```
This should create a new file named `best_epoch.pth.tr` within `checkpoint` directory.

### Test
``` bash
python train.py --eval-only --checkpoint checkpoint --checkpint-file  xxx.pth.tr  --config configs/h36m/xxx.yaml
```

## MPI-INF-3DHP
### Preprocessing
Please refer to [P-STMO](https://github.com/paTRICK-swk/P-STMO#mpi-inf-3dhp) for dataset setup. And the generated ".npz" files (`data_train_3dhp.npz` and `data_test_3dhp.npz`) should be located at `data/motion3d` directory.

### Train
to train a model in MPI-INF-3DHP dataset, file `xxx.yaml` should be located at `data/mpi` directory:
``` bash
python train_3dhp.py --config configs/mpi/xxx.yaml
```
### Test
``` bash
python train_3dhp.py --eval-only --checkpoint mpi-checkpoint --checkpoint-file xxx.pth.tr --config configs/mpi/xxx.yaml
```
