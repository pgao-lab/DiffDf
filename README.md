# DiffDf
## Towards Effective and Efficient Adversarial Defense with Diffusion Models for Robust Visual Tracking

## Environment

-  **Ubuntu**: 20.04
-  **Python**: 3.8
-  **CUDA**: 11.3
-  **PyTorch**: 1.12.0
## Installation


### Steps
#### Clone the repository
```
git clone https://github.com/6uolnxps/DiffDf.git
cd <Project_name>
```
#### Install dependencies
```
pip install -r requirements.txt
```

#### Download pretrained models
1. SiamRPN++([Model_Zoo](https://github.com/STVIR/pysot/blob/master/MODEL_ZOO.md))   
Download **siamrpn_r50_l234_dwxcorr** and **siamrpn_r50_l234_dwxcorr_otb**  
Put them under pysot/experiments/<MODEL_NAME>
2. Perturbation Generators  
Download ([CSA checkpoints](https://github.com/MasterBin-IIAU/CSA/blob/master/README.md)) you need, then put them under checkpoints/<MODEL_NAME>/
#### Download Dataset
[OTB2015](http://faculty.ucmerced.edu/mhyang/papers/pami15_tracking_benchmark.pdf) 
[VOT2018](http://votchallenge.net) 
#### Training

```cd pysot```  

train your own model
```
python train_diff.py
```
#### Testing
open ```common_path.py```, choose the dataset and siamese model to use.  
open ```GAN_utils_xx.py```, choose the generator model to use.  

```cd pysot/tools```  

run experiments about defensing **search regions**  
```
python run_search_diff.py
```
run experiments about defensing **the template**  
```
python run_template_diff.py
```
run experiments about defensing **both search regions and the template**
```
python run_template_search_diff.py
```
