# ScoreSeg

Official implementation of ScoreSeg: Leveraging Score-based Generative model for Self-Supervised Semantic Segmentation of Remote Sensing

## Environment
- This repository is built and tested in RTX 3090Ti GPUs, CUDA==11.4.

- We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n ScoreSeg python=3.9
    conda activate ScoreSeg
    ```
  
- Then, install the corresponding Pytorch version
    ```bash
    conda install pytorch=1.12.1 torchvision=0.13.1 cudatoolkit=11.3 -c pytorch
    ```

- Other requirements
    ```bash
    pip install -r requirements.txt
    ```

- CUDA operators for [Deformable Transformer](https://github.com/fundamentalvision/Deformable-DETR)
    ```bash
    cd ./model/deformalbe_ops
    sh ./make.sh
    ```


## Data Preprocessing
We follow the folder structure and data preprocessing in the [GeoSeg](https://github.com/WangLibo1995/GeoSeg). 
As a result, for example, 13824 tiles of 256 x 256 crops can be obtained in the Potsdam dataset for training while 8064 tiles for testing.

Note that offline data augmentation should be avoided since the following sampling procedure may collect more images.

You can randomly select portions of the dataset, e.g., 1%, 5% and 10% in our experiments. For example,

```bash
python dataset-split.py --dataset_name potsdam --phase train
```

## Training score-based generative models
We recommend you to download the pretrained DDPM released in [ddpm-cd](https://github.com/wgcban/ddpm-cd), 
which can save your time to pretrain a score-based generative model (or so-called diffusion model) on remote sensing
images yourself.

Or, alternatively, you can also run the following command to start the training procedure on 4 GPUs in the DistributedDataParallel (DDP) style.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun  --nnodes 1 --master_port 29500 \
--nproc_per_node=4 \ 
ScoreTrain.py --config config/score_pretraining.json --phase train
```

It will take around 2 days to train a score-based generative model on mixture of Potsdam and Deepglobe datasets.

## Training and testing segmentation modules
After the pretraining of score-based generative models, you can specify the load path of its pth file in the 
segmentation config files, e.g., [potsdam_segmentation.json](./config/potsdam_segmentation.json)

Then run the following command to train the segmentation modules on 2 GPUs and enable wandb (optional) for logging.
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun  --nnodes 1 --master_port 29500 \
--nproc_per_node=2 \
ScoreSeg.py --config config/potsdam_segmentation.json -enable_wandb
```
This will record checkpoints and visualization of segmentation maps in the specified paths in config files automatically.

For testing, remember to specify the 'resume_state' parameter in config files first.
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun  --nnodes 1 --master_port 29500 \
--nproc_per_node=2 \
ScoreSeg.py --config config/potsdam_segmentation.json -enable_wandb  --phase test
```

## Pretrained weights
For convenience, we provide several pretrained segmentation modules weights here.

|    Dataset    |   Ratio  |   OA  |  mF1  |  mIoU |   Url  |
|:-------------:|:--------:|:-----:|:-----:|------:|-------:|
|    Potsdam    |    1%    | 74.62 | 69.84 | 55.60 | [link](https://drive.google.com/drive/folders/10XiIbafE4Hx0IfCiPw0mi0JwrwRjRr-3?usp=sharing) |
|    Vaihingen  |    10%   | 83.75 | 67.39 | 56.33 | [link](https://drive.google.com/drive/folders/1beAKwcV8GnLpCmZFaoQJVDxRU3VHnix3?usp=sharing) |
|    Deepglobe  |    5%    | 78.66 | 66.68 | 53.68 | [link](https://drive.google.com/drive/folders/1i6rQc-NHeSY2E3NnqpUtHlsN8UunDmA1?usp=sharing) |

*Note:*
1. These models are all trained with the score-based models weights in [ddpm-cd](https://github.com/wgcban/ddpm-cd).
2. These models use the same hyparameters as provided [config examples](./config).
3. "Ratio" means the proportion of dataset labels used.

## References
The code of score-based models is from [ddpm-cd](https://github.com/wgcban/ddpm-cd). 
The implementation of deformable transformer and segmentation losses are from [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR) and [GeoSeg](https://github.com/WangLibo1995/GeoSeg) respectively.

## Citation
