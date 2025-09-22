# DS-DKSVD
The code of **"DS-DKSVD: Leveraging Dynamic-Static Dictionary  Learning for Deep K-SVD"** from _The Visual Computer_.

![image](./Images/Figure_2.png)  
* The overall architecture of DS-DKSVD.*

![image](./Images/Figure_3.png)  
* The subnetwork details of the proposed DS-DKSVD.*

## Recommended Environment  
We recommend the following environment for running the code:  
- **Python** 3.8.17
- **PyTorch** 1.12.0 
- **torchvision** 0.13.2  

---

## Installation  
Follow the steps below to set up the environment and run the code:

1. **Clone the Repository**  
   ```bash  
   git clone https://github.com/yaojingzeo/DS-DKSVD.git
   cd DS-DKSVD  
2. **Install Dependencies**
   ```bash  
   pip install -r requirements.txt  
   
## Datasets

In this study, we utilize the following publicly available datasets for non-blind and blind image denoising, as well as image classification:

- **BSD (Berkeley Segmentation Dataset)**  
  [Official Website](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) | [Kaggle Mirror (BSDS500)](https://www.kaggle.com/datasets/balraj98/berkeley-segmentation-dataset-500-bsds500)

- **Set12**  
  [Download Link (HyperAI)](https://hyper.ai/en/datasets/17513)

- **Urban100**  
  [Kaggle Dataset](https://www.kaggle.com/datasets/harshraone/urban100)

- **FMD (Flickr Material Database)**  
  [Download Link](https://sourl.cn/Wyqrui)

- **Br35H (Brain Tumor Detection Dataset)**  
  [Kaggle Dataset](https://www.kaggle.com/ahmedhamada0/brain-tumor-detection)

👉 *BSD, Set12, Urban100, and FMD are used for image denoising tasks, while  Br35H are used for image classification.*

## Dataset Preparation
After downloading, organize the datasets as follows:
Dataset structure:  
```
├── Classification/  
   ├── dataset/  
       ├── test/    
       ├── train/    
       ├── val/
├── Denoising/  
   ├── blind/  # Both synthetic blind and non-blind denoising are performed on the same dataset. 
       ├── real_dateset/
         ├── Confocal_FISH/        
         ├── Confocal_MICE/
         ├── TwoPhoton_MICE/        
   ├── nonblind/  
       ├── gray/    
       ├── test_set12/   
       ├── Urban100/
```
Make sure the data is correctly organized before proceeding to training or evaluation.

## Training and Testing

We provide scripts for both image denoising (non-blind and blind) and classification tasks.

### Image Denoising
- **Non-blind denoising (training):**
  ```bash
  python Denoising/non_blind/train_non.py
- **blind denoising (training):**
   ```bash  
   python Denoising/blind/train_synthetic or train_real.py
   ```
- **Testing (PSNR/SSIM evaluation):**
   ```bash  
   python Denoising/load_model_psnr_ssim_ours.py
   ```
### Classification
   ```bash  
   python Classification/train_class.py
   ```



## Citation
If you find this work useful in your research, please consider citing our paper:
```
@article{wu2025dual,  
  title={DS-DKSVD: Leveraging Dynamic-Static Dictionary  Learning for Deep K-SVD},  
  author={Sun Zhonggui,Yao Jing and Zhang Can},  
  journal={The Visual Computer},  
  year={2025},  
  publisher={Springer},  
}  
```

## Acknowledgment  
We acknowledge and appreciate the contributions of the open-source community, whose valuable work has greatly inspired and supported our research. In particular, some parts of our code are inspired by and adapted from the Deep K-SVD (https://github.com/meyerscetbon/Deep-K-SVD)

## Contact
If you have any questions about our work or code, please email `jingelle@hotmail.com` .

