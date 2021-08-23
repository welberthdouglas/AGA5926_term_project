<div align="center">
    <h2>Improving S-PLUS Image Quality with SRGANs</h2>
</div>

<div align="center">
<h3> Abstract </h3>
</div>

<p align="justify">
Several factors contribute to noise in astronomical images. Random noise from the sky background, the telescope detector and optical system play a part to build noise in images that can make it difficult to identify and study structures. Schawinski et al. 2017 showed the potential of GANs for noise reduction and recovery of galaxy features for images artificially degraded [add solar denoise paper here]. In this project[work] we will use SRGANs to increase image quality (pixel density and signal to noise ratio) of S-PLUS survey images using deeper images of the same objects from Legacy survey as a baseline for training. Preliminary results were qualitatively evaluated and show good concordance with legacy survey images.
</p>

### 1. Introduction

<p align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Several factors contribute to noise in astronomical images: Random noise from the sky background, the telescope's detector and its optical system, and data modifications during preprocessing handling are examples of elements that play a part to build noise in images. Although there are known ways to decrease images' noise, such as bigger exposure times, totally removing the noise is virtually impossible. Furthermore, in some applications, increasing exposure times could mean reducing the covered field significantly. The noise present in images can have a big impact on its overall characteristics. Images with a low signal-to-noise ratio (S/N) are notably difficult to study as structures such as galaxies' spiral arms, and other features may get lost in the noise, making its detection and analysis harder. 
</p>
<p align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The contributions of deep learning in image astronomy have grown in the last years with uses in object classification (Bom et. al., 2021), image reconstruction (Bouman et. al., 2015), and others. In this context, machine learning and deep learning have been successfully used in astronomy for noise reduction with the use of principal component analysis (PCA), convolutional neural networks as shown by (Baso et. al., 2019), and more recently generative adversarial networks (GANs) (Schawinski et al., 2017).
</p>

<p align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Since the introduction of Generative Adversarial Networks by Goodfelow et al. (2014), we have observed the development of several uses in fields ranging from finances to arts; In particular, there were expressive developments in computer vision that lead to use in astronomy in the last years. GANs have been used to retrieve galaxy features and exoplanetary atmospheres  (Schawinski et al., 2017; Zingales and Waldmann., 2018), and to enhance cosmological simulations (Ullmo et al. 2020; Li et al., 2020) to cite a few use cases. Schawinski et al. 2017 showed the potential of GANs for noise reduction and recovery of galaxy features for images artificially degraded. In this work, we will use super-resolution generative adversarial networks (SRGANs), a class of GAN, to increase image quality (pixel density and signal-to-noise ratio) of S-PLUS survey images using deeper images of the same objects from LEGACY survey as a baseline for training.
</p>

### 2. Methodology

#### 2.1 Generative Adversarial Networks

<p align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Generative Adversarial Networks are a class of machine learning frameworks in which two neural networks are trained simultaneously against each other: A generator that tries to build increasingly realistic examples and a discriminator that increasingly gets better at identifying generated samples from real ones. The general goal of training a GAN is to create a generator that would output realistic examples of the objects of interest at the end of training. In the specific case of images, the generator could have different uses depending on its definition: it could create realistic face images using n-dimensional vectors as in style-GAN (Karras et. al., 2019), or it could transform images from one style to another (Isola et. al., 2018). In the case of SRGANs, the architecture used in this work, the purpose of the generator is to increase the resolution of the input image as introduced by Ledig et. al (2017). Figure 2.1 below shows the overall structure of the networks used in this work.
</p>

<p align="center">
  <img  src="./images/schematics.png"/>   
  Figure 2.1. SRGAN general structure used in this work.
</p>

<p align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Initially, a low-resolution image (128x128 S-PLUS original image) is input to the generator. The generator, through a series of convolutions and residual operations, transforms this image into a super-resolution image of size 256 x 256 pixels. The generated image is then given as an input to the discriminator training along with the real high-resolution image (256 x 256 LEGACY survey image of the same object). A Loss function is calculated using the output tensor of the discriminator as well as a feature representation of the image created using a pre-trained VGG19 network. After this phase, the weights and biases of the discriminator are updated, and it gets better at differentiating between generated and real images.
</p>

<p align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In a second phase, the loss function is used to update the weights and biases of the generator via backpropagation. After this step, the generator gets better at creating 256x256 images from a 128x128 input, it effectively gets better at fooling the discriminator. This training process repeats for each iteration and consists in solving the min-max adversarial problem defined by the equation shown bellow.
</p>
    

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;min_{\theta&space;_{G}}max_{\theta&space;_{D}}\mathbb{E}_{I^{HR}\sim&space;p_{train}\left&space;(&space;I^{HR}&space;\right&space;)}\left&space;[&space;log&space;D_{\theta&space;_{D}}\left&space;(&space;I^{HR}&space;\right&space;)&space;\right&space;]&plus;\mathbb{E}_{I^{LR}\sim&space;p_{G}\left&space;(&space;I^{LR}&space;\right&space;)}\left&space;[&space;log\left&space;(&space;1-&space;D_{\theta&space;_{D}}\left&space;(G_{\theta_{G}}&space;I^{LR}&space;\right&space;)\right&space;)&space;\right&space;]" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\large&space;min_{\theta&space;_{G}}max_{\theta&space;_{D}}\mathbb{E}_{I^{HR}\sim&space;p_{train}\left&space;(&space;I^{HR}&space;\right&space;)}\left&space;[&space;log&space;D_{\theta&space;_{D}}\left&space;(&space;I^{HR}&space;\right&space;)&space;\right&space;]&plus;\mathbb{E}_{I^{LR}\sim&space;p_{G}\left&space;(&space;I^{LR}&space;\right&space;)}\left&space;[&space;log\left&space;(&space;1-&space;D_{\theta&space;_{D}}\left&space;(G_{\theta_{G}}&space;I^{LR}&space;\right&space;)\right&space;)&space;\right&space;]" title="\large min_{\theta _{G}}max_{\theta _{D}}\mathbb{E}_{I^{HR}\sim p_{train}\left ( I^{HR} \right )}\left [ log D_{\theta _{D}}\left ( I^{HR} \right ) \right ]+\mathbb{E}_{I^{LR}\sim p_{G}\left ( I^{LR} \right )}\left [ log\left ( 1- D_{\theta _{D}}\left (G_{\theta_{G}} I^{LR} \right )\right ) \right ]" />
</a>
</p>
    
    
<p align="justify">
Where <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;I^{HR}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\large&space;I^{HR}" title="\large I^{HR}" /></a> is an high-resolution image, <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;I^{LR}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\large&space;I^{LR}" title="\large I^{LR}" /></a> is a low-resolution image and <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;D_{\theta&space;_{D}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\large&space;D_{\theta&space;_{D}}" title="\large D_{\theta _{D}}" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;D_{\theta&space;_{D}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\large&space;D_{\theta&space;_{D}}" title="\large D_{\theta _{D}}" /></a> are the discriminator and the generator networks respectively parametrized by a set of weights and biases defined by <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\theta&space;_{D}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\large&space;\theta&space;_{D}" title="\large \theta _{D}" /></a> and  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\theta&space;_{G}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\large&space;\theta&space;_{G}" title="\large \theta _{G}" /></a> as shown in Ledig et. al (2017). The generative adversarial training process is known to be inherently unstable, and to avoid divergence and possible collapse during the training, hyperparameters, the network's architecture, and the training process should be finely tuned.
</p>

#### 2.2 Network Architecture

- Say that upsampling with bilinear interpolation eliminated almost completly the high frequency artifacts (checkerboard pattern).


#### 2.3 Train Data

### 3. Results

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=MSE&space;=&space;\frac{1}{m&space;\cdot&space;n}\sum_{i=0}^{m-1}\sum_{j=0}^{n-1}\left&space;[&space;I(i,j)&space;-&space;K(i,j)&space;\right&space;]^{2}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?MSE&space;=&space;\frac{1}{m&space;\cdot&space;n}\sum_{i=0}^{m-1}\sum_{j=0}^{n-1}\left&space;[&space;I(i,j)&space;-&space;K(i,j)&space;\right&space;]^{2}" title="MSE = \frac{1}{m \cdot n}\sum_{i=0}^{m-1}\sum_{j=0}^{n-1}\left [ I(i,j) - K(i,j) \right ]^{2}" />
</a>
</p>

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=PSNR&space;=&space;20\log_{10}\left&space;(&space;\frac{MAX_{I}}{\sqrt{MSE}}\right&space;)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?PSNR&space;=&space;20\log_{10}\left&space;(&space;\frac{MAX_{I}}{\sqrt{MSE}}\right&space;)" title="PSNR = 20\log_{10}\left ( \frac{MAX_{I}}{\sqrt{MSE}}\right )" />
 </a>
</p>


<p align="center">
  <img src="./images/validation_images.png"/>   
  Figure 3.X. 
</p>

<p align="center">
  Table 3.1. 
</p>
<p align="center">
  <img height = "180" src="./images/PSNR.png"/>   
</p>

<p align="center">
  <img src="./images/histogram.png"/>   
  Figure 3.X. 
</p>

### 4. Discussion and Conclusions

### 5. Future Work
- changes in architecture to get a network more suitable to reduce noise at the same time that it increases resolution
- Evaluate generator on more diverse objects, not only galaxies
- check for consistency using other metrics other than PSNR.
- Different loss function
- Features from a different model (not VGG)
- Check which vgg features get activated with galaxies images
- Low frequency artifacts might be caused by the generator trying to mimic legacy images' noise (it is a good idea to check if sone feature in VGG is activated by noise in images)
- Train with images with more consistent amount of noise
- Adjust preprocessing to decrease input noise

### 6. References

[Baso, C. J. D. ; Rodríguez, J.C.; Danilovic, S. 2019.](https://arxiv.org/abs/1908.02815) Solar image denoising with convolutional neural networks.    
[Bom, C. R.; Cortesi, A.; Lucatelli, G.; Dias, L. O.; Schubert, P.; Schwarz, G.B.O.; Cardoso, N. M.; Lima, E. V. R.; Oliveira, C.M.; Sodre Jr., L.; Castelli, A.V.S.;Ferrari, F.; Damke, G.; Overzier, R.; Kanaan, A.; Ribeiro, T.; Schoenell, W.. 2021.](https://arxiv.org/abs/2104.00018)Deep Learning Assessment of galaxy morphology in S-PLUS DataRelease 1.    
[Bouman, K. L.; Johnson, M.D.; Zoran, D.; Fish, V.L.; Doeleman, S.S.; Freeman, W.T.. 2015](https://arxiv.org/abs/1512.01413) Computational Imaging for VLBI Image Reconstruction.   
[Goodfellow, I. J. ; Abadie, J.P.; Mirza,M.; Xu, B.; Farley, D. W.; Ozair, S.; Courville, A.; Bengio, A.. 2014.](https://arxiv.org/abs/1406.2661) Generative Adversarial Networks.
[Isola, P.; Zhu, J.Y.; Zhou, T.; Efros, A.A.. 2018](https://arxiv.org/abs/1611.07004) Image-to-Image Translation with Conditional Adversarial Networks.   
[Ledig, C.; Theis, L.; Huszar, F.; Caballero, J; Cunningham, A.; Acosta, A.; Aitken, A.; Tejani, A.; Totz, J.; Wang, A.; Shi, W.. 2017.](https://arxiv.org/abs/1609.04802) Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network. 
[Karras, T.; Laine, S.; Aila,T.. 2019](https://arxiv.org/abs/1611.07004) A Style-Based Generator Architecture for Generative Adversarial Networks.    
[Fussell, L.; Moews, B. 2018.](https://arxiv.org/abs/1811.03081) Forging new worlds: high-resolution synthetic galaxies with chained generative adversarial networks.       
[Li, Y.; Ni, Y.; Ruppert, A. C. C.; Matteo, T. D.; Bird, S.; Feng, Y.. 2020.](https://arxiv.org/abs/2010.06608) AI-assisted super-resolution cosmological simulations.   
[Schawinski, K.; Zhang, C.; Zhang, H.; Fowler, L.; Santhanam, G. K.. 2017.](https://academic.oup.com/mnrasl/article/467/1/L110/2931732) Generative adversarial networks recover features in astrophysical images of galaxies beyond the deconvolution limit.   
[Ullmo, M.; Decelle, A.; Aghanim, N.. 2020.](https://arxiv.org/abs/2011.05244) Encoding large-scale cosmological structure with generative adversarial networks.   
[Zingales, T and Waldmann, I. P.. 2018.](https://arxiv.org/abs/1806.02906) ExoGAN: Retrieving Exoplanetary Atmospheres Using Deep Convolutional Generative Adversarial Networks.   

### 7. Appendix
#### A.1 Validation Images

The figure A.1 bellow show the validation images along with the respective generated images and images from legacy survey. All objects selected to the validation set in were chosen randomly from the original collected objects.

<p align="center">
  <img src="./images/validation.png"/>
  Figure A.1. Left - SPLUS original image, Center - image enhanced using the trained generatod, Right - image from the same object from Legacy Survey
</p>

#### A.2 Sample of Train Images

The figure A.2 bellow show a sample of the train images along with the respective generated images and images from legacy survey.

<p align="center">
  <img src="./images/train.png"/>
  Figure A.2. Left - SPLUS original image, Center - image enhanced using the trained generatod, Right - image from the same object from Legacy Survey
</p>

#### A.3 Limitations of Generated images

The images bellow show detected flaws present in some generated images in the validation set. The possible origin of these flaws could be due to several factors such as network's architecture and loss function, and input images quality. 

<p align="center">
  <img src="./images/limitations_01.png"/>   
  Figure A.3.1. Detected flaws in generated images. Left and Center: Unnatural structures and colors detected in light regions. Right: Very noisy input image resulting in a cloudy output image. 
</p>
    
<p align="center">
  <img src="./images/limitations_02.png"/>       
  Figure A.3.2. Low frequency feature present in all generated images.
</p>