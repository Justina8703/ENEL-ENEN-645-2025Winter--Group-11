Note:The weights were too large for GitHub and so we uploaded them on Google Drive and the following is the link to that file:
https://drive.google.com/file/d/1SmOPYnAPlshN_ul3aCPWlk7uR6BPJNfD/view?usp=sharing

## Overview
This notebook presents a garbage classification task utilizing a hybrid model that integrates DistilBERT for text processing and MobileNetV2 for image feature extraction. The model is trained to categorize images into four classes—black bin, green bin, blue bin, and other—by using both textual and visual information to enhance the accuracy of prediction.
### Steps
Data Preprocessing  
Deep Learning Model Definition  
Graph of the Training Process  
Evaluation and Prediction    
### Requirements
PyTorch for model training  
Transformers for text encoding using DistilBERT  
Torchvision for MobileNetV2  
## Model Architecture  
### MobileNetV2 Feature Extraction  
We use MobileNetV2 as the image feature extractor. Pre-trained weights from ImageNet are used, and the classifier is removed to obtain a 1280-dimensional feature vector, which is then passed through a fully connected layer and reduced to 750 dimensions before fusion with text features.  
### DistilBERT Text Encoder
We use DistilBERT, a lightweight version of BERT, to process the textual descriptions extracted from image filenames. The text is tokenized, and the [CLS] token embedding (768 dimensions) is extracted. This embedding is then passed through a fully connected layer, reducing it to 750 dimensions, followed by batch normalization.  


