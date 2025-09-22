# DPRF
The code of "DS-DKSVD: Leveraging Dynamic-Static Dictionary  Learning for Deep K-SVD" from _The Visual Computer_.

## The Overall Architecture of DS-DKSVD
![image](./Images/Figure_2.png)
The workflow of the proposed **DS-DKSVD: Leveraging Dynamic-Static Dictionary Learning for Deep K-SVD**. Left: the overall architecture of DS-DKSVD. Right: the architecture of one stage in DS-DKSVD.

## The subnetwork details of the proposed DS-DKSVD
![image](./Images/Figure_3.png)
The subnetwork details of the proposed DS-DKSVD


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
   
## denoising Datasets
In this study, we utilize the following publicly available datasets for non-blind and blind image denoising, as well as image classification:

- **BSD (Berkeley Segmentation Dataset)**  
  [Official Website](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) | [Kaggle Mirror (BSDS500)](https://www.kaggle.com/datasets/balraj98/berkeley-segmentation-dataset-500-bsds500)

- **Set12**  
  [Download Link (HyperAI)](https://hyper.ai/en/datasets/17513)

- **Urban100**  
  [Kaggle Dataset](https://www.kaggle.com/datasets/harshraone/urban100)

- **FMD (Flickr Material Database)**  
  [Download Link](https://sourl.cn/Wyqrui)
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

## Training
### Non-blind denoising
To train the non-blind denoising model, run: 
   ```bash  
   python Denoising/non_blind/train_non.py
   ```

## Evaluation
You can evaluate the performance of the trained models using the following scripts:
* Evaluate Registration Network:[metrics_reg.py](Evaluator/metrics_reg.py)
* Evaluate Fusion Network:[metrics_fus.py](Evaluator/metrics_fus.py) 

## Registration Results
![image](./Imgs/visual-reg-00202-noGrid.jpg)
![image](./Imgs/visual-reg-00718N-noGrid.jpg)
![image](./Imgs/visual-reg-FLIR_08954-noGrid.jpg)

## Fusion Results
![image](./Imgs/visual-fus-FMB_00089.jpg)
![image](./Imgs/visual-fus-M3FD_00738.jpg)

## Citation
If you find this work useful in your research, please consider citing our paper:
```
@article{wu2025dual,  
  title={Dual-Pyramid Framework for Robust Infrared-Visible Image Registration and Fusion via Convolutional Dictionary Learning},  
  author={Wu, Chaojie and Li, Zheng},  
  journal={The Visual Computer},  
  year={2025},  
  publisher={Springer},  
}  
```

## Acknowledgment  
We acknowledge and appreciate the contributions of the open-source community, whose valuable work has greatly inspired and supported our research. In particular, some parts of our code are inspired by and adapted from the DCDicL_denoising (https://github.com/natezhenghy/DCDicL_denoising)

## Contact
If you have any questions about our work or code, please email `2023223040094@stu.scu.edu.cn` .

