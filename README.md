# DPRF
The code of "Dual-Pyramid Framework for Robust Infrared-Visible Image Registration and Fusion via Convolutional Dictionary Learning" from _The Visual Computer_.

## Overall framework
![image](./Imgs/workflow.jpg)
The workflow of the proposed **Dual-Pyramid Infrared-Visible Image Registration and Fusion Framework (DPRF)**. This framework is designed for robust cross-modal image registration and multi-scale image fusion.

## Registration Network
![image](./Imgs/reg.jpg)
The architecture of the proposed **Gaussian-Dictionary Dual Encoding Residual Flow Registration (GDFR)** Network. For clarity, only a 3-level pyramid structure is shown here. The network utilizes Gaussian-Dictionary dual domain learning-based feature extraction and residual flow estimation for accurate infrared-visible image registration.

## Fusion Network
![image](./Imgs/fus.jpg)
The architecture of the proposed **Laplacian Pyramid and Dictionary Learning-based Multi-scale Feature Fusion Network (LPDF-Net)**. This fusion network leverages multi-scale features to generate high-quality fused images.

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
   cd DPRF  
2. **Install Dependencies**
   ```bash  
   pip install -r requirements.txt  
   
## Datasets
In this work, we used the following publicly available datasets for infrared-visible image registration and fusion:
* RoadScene (https://github.com/hanna-xu/RoadScene)
* MSRS (https://github.com/Linfeng-Tang/MSRS)
* M3FD (https://github.com/JinyuanLiu-CV/TarDAL)

## Dataset Preparation
After downloading, organize the datasets as follows:
Dataset structure:  
```
├── data/  
   ├── RoadScene/  
       ├── ir/    # Infrared images  
       ├── vi/    # Visible images  
   ├── MSRS/  
       ├── ir/    # Infrared images  
       ├── vi/    # Visible images  
   ├── M3FD/  
       ├── ir/    # Infrared images  
       ├── vi/    # Visible images  
```
Make sure the data is correctly organized before proceeding to training or evaluation.

## Training
To train the DPRF model with the default parameters already configured, simply run the following command:  
   ```bash  
   python Trainer/Trainer.py
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

