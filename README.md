_Note:The weights were too large for GitHub and so we uploaded them on Google Drive and the following is the link to that file:_
_https://drive.google.com/file/d/1SmOPYnAPlshN_ul3aCPWlk7uR6BPJNfD/view?usp=sharing_

# Garbage Classification with DistilBERT and MobileNetV2
Justice Nsafoah (30076935) Junqi Li(30270860) Nurkeldi Iznat (30261427) Osama Kashif (30037753)
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
### Classifier
The 750-dimensional image and text features are concatenated (1500 dimensions total).  
The fused representation passes through a fully connected classifier (1500 → 1000 → num_classes), with ReLU activation and dropout to improve generalization.
## Hyperparameters  
Learning rate: 2e-5
Batch size: 64 
Number of epochs: 10
## Results and Discussions  
### Model Performance  
| **Class**   |**Precision**|**Recall**|**F1-Score**|**Support**|
|-------------|-----------|--------|----------|---------|
| **Black**   | 0.76      | 0.66   | 0.71     | 695     |
| **Blue**    | 0.81      | 0.88   | 0.84     | 1086    |
| **Green**   | 0.90      | 0.94   | 0.92     | 799     |
| **TTR**     | 0.84      | 0.81   | 0.83     | 852     |
### **Overall Performance**  
| Metric        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| **Accuracy**  | -        | -      | 0.83     | 3432    |
| **Macro Avg** | 0.83     | 0.82   | 0.82     | 3432    |
| **Weighted Avg** | 0.83   | 0.83   | 0.83     | 3432    |  

 
Among the classes, the "Green" category stands out with a high recall (0.94) and F1-score (0.92), suggesting that the model is particularly effective in identifying this class. The "Blue" category also performs well, achieving a recall of 0.88 and an F1-score of 0.84. 
On the other hand, the "Black" category has a lower recall (0.66) and F1-score (0.71), indicating that some samples may be misclassified, suggesting potential areas for increasing the performance. Despite this, its precision (0.76) remains reasonable. 
Overall, the classification results demonstrate consistent performance across all classes, with good accuracy, precision, recall, and F1-scores. The model achieves an overall accuracy of 83%, indicating its effectiveness in distinguishing different categories. 



