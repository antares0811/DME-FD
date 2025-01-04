# Set up ISIC2018 and HAM10000 dataset
Folder Organization
```
|-data
    |-isic2018
        |-images
        |-labels
        |-fold*.txt
    |-ham10000
        |-images
        |-labels
        |-fold*.txt
```
#### Download Dataset Link ####
    ISIC2018: [Link](https://challenge.isic-archive.com/data/#2018) - (Task 1: Train + Val + Test)
    HAM10000: Image ([Link](https://challenge.isic-archive.com/data/#2018) - Task 3) + Label ([Link](https://www.kaggle.com/datasets/tschandl/ham10000-lesion-segmentations))

#### Set up ISIC2018 ####
    Move all extracted images into 'images' folder and labels into 'labels' folder
    Use the function process_isic2018 function in process_resize.py to resize images and labels and convert it to .npy file
    Move processed folder into data/isic2018

#### Set up HAM10000 ####
    Move all extracted images into 'images' folder and labels into 'labels' folder
    Use the function process_isic2018 function in process_resize.py to resize images and labels and convert it to .npy file
    Move processed folder into data/ham10000
