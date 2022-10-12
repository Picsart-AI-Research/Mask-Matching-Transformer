# Mask Matching Transformer for Few-Shot Segmentation



#### Pre-requests

1. Setup a conda environment(e.g., miniconda as follows).

  ```shell
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  chmod +x Miniconda3-latest-Linux-x86_64.sh
  ./Miniconda3-latest-Linux-x86_64.sh
  source ~/.bashrc
  ```
  
2. Create a virtual env. with python 3.8.5:
  ```shell
  conda create -n mmformerEnv python=3.8.5
  ```

  ```shell
  conda activate mmformerEnv
  conda install pytorch=1.11.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch
  # For nvcc
  conda install -c conda-forge cudatoolkit-dev=11.3
  # Downgrade the setuptools
  pip install setuptools==58.2.0
  ```

  ```shell

  git clone https://github.com/facebookresearch/detectron2.git
  cd detectron2
  git reset --hard 932f25ad38768d
  # HEAD is now at 932f25a Add build tracker based on registry
  cd ..
  python -m pip install -e detectron2
  ```

  ```shell
  conda install -c conda-forge opencv timm
  conda install scipy 
  ```
 
   **Note**: Using the latest version of detectron2 may cause weight loading failure. Please use the following command to return the version:
  ```
  git reset --hard 932f25ad38768d
  ```

#### Build Dependencies

```
cd mask2former/modeling/pixel_decoder/ops/
bash make.sh
```



#### List Preparation

+ Please add [file](https://drive.google.com/file/d/1kkBOtL_Ujd-bAkGXADYFaOivTl1WD4b_/view?usp=sharing) to prepare `./list/`

#### Data Preparation

+ Please refer to [CyCTR](https://github.com/YanFangCS/CyCTR-Pytorch) to prepare the datasets 
```
${YOUR_PROJ_PATH}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- instances_train2017.json
        |   `-- instances_val2017.json
        |-- train2017
        |   |-- 000000000009.jpg
        |   |-- 000000000025.jpg
        |   |-- 000000000030.jpg
        |   |-- ... 
        `-- val2017
            |-- 000000000139.jpg
            |-- 000000000285.jpg
            |-- 000000000632.jpg
            |-- ... 
```

Then, run  
```
python prepare_coco_data.py
```
to prepare COCO-20^i data.

#### Train
Run this command for training:
```
    python TRAIN.py --config-file configs/DATASET/STEP.yaml
```
For example
1. Modify `DATASETS.SPLIT` in `configs/coco/step1.yaml` and run this command for training **step1** of COCO: 
```
    python train_step1.py --config-file configs/coco/step1.yaml --num-gpus 1
```

2. Modify `DATASETS.SPLIT` and `MODEL.WEIGHTS` in `configs/coco/step2.yaml` and run this command for training **step2** of COCO: 
```
    python train_step2.py --config-file configs/coco/step2.yaml --num-gpus 1
```



#### Test Only
Modify `eval.yaml` file (`DATASETS.SPLIT`, `MODEL.META_ARCHITECTURE` and `MODEL.WEIGHTS`)
Run the following command: 
```
    python test.py --config-file configs/DATASET/eval.yaml --num-gpus 1 --eval-only
```

For example, 
```
    python test.py --config-file configs/pascal/eval.yaml --num-gpus 1 --eval-only
```

#### Pretrained models
[models](https://drive.google.com/drive/folders/1D4EiAqyeejQnxGydDapflTABXeYBplqK?usp=sharing)
