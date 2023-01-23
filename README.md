# Fruit And Vegetables Image Classification
Team: Anne Marshall, Robert Park, Shirley Jiang, Cynthia Zhu

## Introduction
Our project aims to classify fruits and vegetables using handcrafted features–including color and texture quantified by Fast Fourier Transforms (FFT), Histogram of Gradients (HOG), and Local Binary Pattern (LBP) – and learned features from ResNet pretrained models. We classify the images using Kmeans, KNN, SVM, and CNN models. We compare performance between the models using combinations of our handcrafted and learned features. Our final results show that the best performing model was SVM using only the ResNet features.

## Dataset
The dataset comes from Fruits and Vegetables Image Recognition Dataset on Kaggle. The dataset contains 36 total labels. The original dataset was created by scraping Bing image search results. Due to the nature of the data collection, images were incredibly varied—some having outdoor backgrounds and some on white backgrounds, some with single or multiple pieces of the fruit or vegetable, and others being cartoonized, stylized, or a memeified version of the fruit or vegetable.

## Result
The ResNet feature set has a strong representation of our images, and we achieved 91.667% final test accuracy with ResNet features alone with SVM. With the addition of hand-crafted features, we were able to achieve 89.286% validation accuracy.

#### 31 Category KNN Validation Set Accuracy and F1 Scores by Best Performing Feature Combination

|          | KNN Color Features Only | KNN FFT Only | KNN ResNet only | KNN Color + FFT | KNN Color + FFT + ResNet |
|----------|-------------------------|--------------|-----------------|-----------------|--------------------------|
| Accuracy |                  11.508 |        17.46 |          79.762 |                 |                   81.746 |
| F1       |                  13.359 |       19.935 |          80.215 |                 |                   82.251 |

#### 31 Category SVM Validation Set Accuracy and F1 Scores by Best Performing Feature Combination

|          | SVM Color Features Only | SVM FFT Only | SVM ResNet only | SVM Color + FFT | SVM Color + FFT + ResNet |
|----------|-------------------------|--------------|-----------------|-----------------|--------------------------|
| Accuracy |                   17.46 |       20.635 |          90.079 |          29.365 |                   89.286 |
| F1       |                  19.122 |       23.448 |          90.065 |          30.961 |                   89.189 |

#### 31 Category CNN Validation Set Accuracy Scores by Best Performing Feature Combination

|          | CNN Color Features Only | CNN FFT Only | CNN ResNet only | CNN Color + FFT | CNN Color + FFT + ResNet |
|----------|-------------------------|--------------|-----------------|-----------------|--------------------------|
| Accuracy |                   19.05 |              |          83.333 |                 |                    83.33 |
| F1       |                         |              |          83.562 |                 |                          |

## Report
Please see [report.ipynb](https://github.com/CynYZhu/food_image_classification/blob/main/report.ipynb) for complete analysis. 
