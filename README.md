# Fruit And Vegetables Image Classification
Team: Anne Marshall, Robert Park, Shirley Jiang, Cynthia Zhu

## Introduction
Our project aims to classify fruits and vegetables using handcrafted features–including color and texture quantified by Fast Fourier Transforms (FFT), Histogram of Gradients (HOG), and Local Binary Pattern (LBP) – and learned features from ResNet pretrained models. We classify the images using Kmeans, KNN, SVM, and CNN models. We compare performance between the models using combinations of our handcrafted and learned features. Our final results show that the best performing model was SVM using only the ResNet features.

## Dataset
The dataset comes from Fruits and Vegetables Image Recognition Dataset on Kaggle. The dataset contains 36 total labels. The original dataset was created by scraping Bing image search results. Due to the nature of the data collection, images were incredibly varied—some having outdoor backgrounds and some on white backgrounds, some with single or multiple pieces of the fruit or vegetable, and others being cartoonized, stylized, or a memeified version of the fruit or vegetable.

## Result
The ResNet feature set has a strong representation of our images, and we achieved 91.667% final test accuracy with ResNet features alone with SVM. With the addition of hand-crafted features, we were able to achieve 89.286% validation accuracy.

## Report
Please see [report.ipynb](https://github.com/CynYZhu/food_image_classification) for complete analysis. 
