# FIBA: Frequency-Injection based Backdoor Attack in Medical Image Analysis

## Introduction

We introduce a novel backdoor attack method named FIBA in the Medical Image Analysis (MIA) domain. FIBA injects the trigger in the amplitude spectrum in the frequency domain. It preserves the semantics of the poisoned image pixels by maintaining the phase information, making it capable of delivering attacks to both classification and dense prediction models.

This is an official implementation of the CVPR 2022 Paper **[FIBA: Frequency-Injection based Backdoor Attack in Medical Image Analysis](https://arxiv.org/abs/2112.01148)** in Pytorch. 

## News

5/5/2022 - Our code of ISIC2019 has been relesed.

## Requirements

* Install required python packages:

  >$ pip install -r requirements.txt

* Download the **[ISIC2019 dataset](https://challenge.isic-archive.com/data/)** and prepare the TXT files  for dataset partitioning (shown in `.\txt`)
* Download the **[trigger image](https://drive.google.com/file/d/1-0j1b_WhCoclkCfk0yICJQ4o06QG5q6r/view?usp=sharing)** and the **[noise images](https://drive.google.com/file/d/1--Uelbs-GrYUCa3YTgjSK6aywdZb2fRx/view?usp=sharing)**

## Training

1. Modify the contents of `./utils/dataloader.py` depending on the way you save the dataset.
2. Create `./checkpoints` to save the model.
3. Start traning:
    > python train.py  --target_label 3 –pc 0.1 --alpha 0.15 –beta 0.1  --target_img './coco_val75/000000002157.jpg' --cross_dir './coco_test1000' --split_idx 0 --experiment_idx 'demo' 

    * target_label: The label of target class

    * pc: The ratio of poisoned images

    * alpha: The blend ratio $\alpha$ 

    * beta: $\beta$​ which determines the location and range of the low-frequency patch inside the amplitude spectrum to be blended

    * target_img: The path of trigger image

    * cross_dir: The path of the folder which saves noise images

    * split_idx: Used for multi-fold cross validation experiment

    * experiment_idx: The name of the experiment

  The model will be save at `checkpoints/ISIC2019/all2onedemo/best_acc_bd_ckpt.pth.tar`

## Test

> python eval.py --target_label 3 --pc 0.1 --alpha 0.15 –beta 0.1  --target_img './coco_val75/000000002157.jpg' --cross_dir './coco_test1000' --split_idx 0  --test_model './checkpoints/ISIC2019/all2onedemo/best_acc_bd_ckpt.pth.tar'

* test_model: The path of test model

## Citation

If you find this repo useful for your research, please consider citing our paper

>@article{feng2021fiba,
>title={FIBA: Frequency-Injection based Backdoor Attack in Medical Image Analysis},
>author={Feng, Yu and Ma, Benteng and Zhang, Jing and Zhao, Shanshan and Xia, Yong and Tao, Dacheng},
>journal={arXiv preprint arXiv:2112.01148},
>year={2021}
>}

### Acknowledgement

Some of the code is adapted from **[WaNet](https://github.com/VinAIResearch/Warping-based_Backdoor_Attack-release)** and **[FedDG](https://github.com/liuquande/FedDG-ELCFS)**, which have been cited in our paper.
