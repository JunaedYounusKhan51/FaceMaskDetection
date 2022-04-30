# FaceMaskDetection: A Comparative Analysis of Machine Learning Approaches for Automated Face Mask Detection During COVID-19


# Links:
**Report:** https://arxiv.org/abs/2112.07913 (Pre-print)

**Presentation:** [619_data_analytics_project_presentation.pptx](https://github.com/al-alamin/FaceMaskDetection/Project_presentation.pptx)

<br>

# Dataset:

* Simulated Facemask dataset (SFMD) (https://github.com/cabani/MaskedFace-Net):<br>
The dataset is published by Cabani et al. It contains high resolution (1024×1024) 67K simulated samples of people wearing mask correctly and incorrectly. The original images are collected from Flicker dataset. For our study, we randomly selected 6,442 masked images from SFMD and  6,442 unmasked images from the Flicker dataset.



* Real Facemask dataset (RFMD) (https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset):<br>
This is the biggest dataset of masked face for this problem. It consists of 90K images of 525 different people without masks and 6,442 images of real masked people. Among the masked images, many are from the same 525 people and some are different. To keep a balanced dataset, we used 6,442 masked and 6,442 unmasked images.

<br><br>

# Machine Learning Model Experiment:
* VGG19
* VGG19 with Transfer Learning
* ResNet50
* ResNet50 with Transfer Learning
* Cross Platform Domain Adaptation Analysis

Please read our pre-print paper for more details.

**Cluster:** All the experiments are conducted in the Univeristy of Calgary ARC and TALC Cluster with NVIDIA Tesla T4 GPU.