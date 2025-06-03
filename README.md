# MS-Detection
Multiple Sclerosis Early Prediction and Lesion Segmentation using Deep Learning
(Graduation Project – Computer and Artificial Intelligence – Helwan University (2024)

Project Overview

•	MS classification using pre-trained CNN models.

•	Lesion segmentation using semantic segmentation models.

•	A web-based platform for image upload, visualization, and MS information


Motivation & Target Paper
We were inspired by and sought to build upon the findings of the following paper:

 Exploring Deep Learning Techniques for Multiple Sclerosis Segmentation
 
 (PDF):https://drive.google.com/file/d/1n3LNgnWgeyfW3eV70gllHsbEStCpU1lH/view
 
Our project improved the model performance detailed in the paper, particularly in:

•	Classification test accuracy

•	Segmentation Dice coefficient and Jaccard Index

•	Simpler deployment using 2D slices instead of complex 3D volumes


Core Components

1.	MS Classification (2D MRI):
   
We trained and evaluated 16+ pre-trained models on a curated dataset of MRI images to predict MS presence.

 Classification Dataset:
 
Kaggle – Multiple Sclerosis MRI Classification Dataset:https://www.kaggle.com/datasets/buraktaci/multiple-sclerosis

 Comparison of Pre-trained Models on MS Classification 
 
•	Higher test accuracy vs. referenced paper results

•	Better generalization due to custom preprocessing, augmentation, and dataset balancing


![gp classificaton model](https://github.com/user-attachments/assets/29131733-3992-4b54-bb84-e5d28ffd2b77)

2.	Lesion Segmentation (2D MRI)
   
We developed a custom DeepLabV3+ model for segmenting lesions from MRI slices,achieving:

![segmentation gp](https://github.com/user-attachments/assets/40d283f4-1cb6-4fbd-8566-3ecf7d5c57a9)

  Segmentation Dataset:
  
•	MSLSC Dataset (Original):https://www.kaggle.com/datasets/washingtongold/mslsc-multiplesclerosis-lesion-segmentation

•	Converted 2D Version on Kaggle:https://www.kaggle.com/datasets/ayamahmoudzaki/multiple/data


We chose DeepLabV3+ over 3D Attention U-Net due to:

•	Faster training on 2D slices

•	Simpler deployment

•	Improved performance on limited data

Web Platform

A modern Angular-based educational website was built to:

•	Allow users to upload MRI images and view predictions.

•	Display lesion segmentation results in real-time.

•	Provide educational content and FAQs about MS.


 Frontend Tools:
 
•	Angular + Angular Material + NG Bootstrap

•	Hosting-ready structure (can be integrated with backend APIs for full deployment)


Tools & Technologies

•	Programming: Python 3, Jupyter, Visual Studio Code

•	Libraries: Keres, TensorFlow, PyTorch, NumPy, Pandas, Scikit-learn, OpenCV, Matplotlib, Kinter.

•	Web Development: Angular, HTML, SCSS, TypeScript

•	Preprocessing: https://www.kaggle.com/code/drvirushussein/preprocess-nii-files : used for preprocessing and converting 3D .nii MRI files into 2D image slices suitable for training.

•	Other Tools: Unsplash (image assets), remove.bg (background removal), Kaggle (datasets, notebooks)

Resources & Links

•	 Target Paper (PDF):https://drive.google.com/file/d/1n3LNgnWgeyfW3eV70gllHsbEStCpU1lH/view

•	 Classification Dataset:https:https://www.kaggle.com/datasets/buraktaci/multiple-sclerosis

•	 Segmentation Dataset (Original):https://www.kaggle.com/datasets/washingtongold/mslsc-multiplesclerosis-lesion-segmentation

•	 Segmentation Dataset (Converted):https://www.kaggle.com/datasets/ayamahmoudzaki/multiple/data

•	 Preprocessing Notebook:https:https://www.kaggle.com/code/drvirushussein/preprocess-nii-files

 Key Achievements
 
•	Achieved 99.57% test accuracy for segmentation and up to 97% classification accuracy using EfficientNetB0.

•	Improved Dice coefficient (0.79) and Jaccard Index (0.65) over reference models.

•	Created a fully functional web-based interface.

•	Designed and conducted preprocessing pipelines for 3D → 2D conversion.

•	Enhanced performance using data augmentation, model fine-tuning, and early stopping techniques.









