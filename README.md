<div align="center">
    <h2>Improving S-PLUS Image Quality with SRGANs</h2>
</div>

### Abstract

<p align="justify">
Several factors contribute to noise in astronomical images. Random noise from the sky background, the telescope detector and optical system play a part to build noise in images that can make it difficult to identify and study structures. Schawinski et al. 2017 showed the potential of GANs for noise reduction and recovery of galaxy features for images artificially degraded [add solar denoise paper here]. In this project[work] we will use SRGANs to increase image quality (pixel density and signal to noise ratio) of S-PLUS survey images using deeper images of the same objects from Legacy survey as a baseline for training. Preliminary results were qualitatively evaluated and show good concordance with legacy survey images.
</p>

### 1. Introduction

### 2. Methodology

#### 2.1 Generative Adversarial Networks

#### 2.2 Network Architecture

#### 2.3 Train Data

### 3. Results

### 3.X Limitations

### 4. Discussion and Conclusions

### 5. Future Work
- changes in architecture to get a network more suitable to reduce noise at the same time that it increases resolution
- cite paper solar denoising and its references, maybe use a hybrid architecture with noise reduction capabilities
- Evaluate generator on more diverse objects, not only galaxies
- check for consistency using other metrics other than PSNR.
- Different loss function
- Features from a different model (not VGG)
- Check which vgg features get activated with galaxies images
- Low frequency artifacts might be caused by the generator trying to mimic legacy images' noise (it is a good idea to check if sone feature in VGG is activated by noise in images)
- Train with images with more consistent amount of noise
- Adjust preprocessing to decrease input noise

### 6. References

### 7. Appendix
#### A.1 Validation Images

The figure X bellow show the validation images along with the respective generated images and images from legacy survey. All objects selected to the validation set in were chosen randomly from the original collected objects.

<p align="center">
  <img  width="1000" src="./images/validation.png"/>
  Figure X. Left - SPLUS original image, Center - image enhanced using the trained generatod, Right - image from the same object from Legacy Survey
</p>

#### A.2 Sample of Train Images

The figure X bellow show a sample of the train images along with the respective generated images and images from legacy survey.

<p align="center">
  <img  width="1000" src="./images/train.png"/>
  Figure X. Left - SPLUS original image, Center - image enhanced using the trained generatod, Right - image from the same object from Legacy Survey
</p>

#### A.3 Validation Images

<p align="center">
  <img  height="400" src="./images/limitations_01.png"/>   
  Figure X. 
</p>

<p align="center">
  <img  height="300" src="./images/limitations_02.png"/>    
  Figure X. 
</p>