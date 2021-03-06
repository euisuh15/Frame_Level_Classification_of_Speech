---
typora-copy-images-to: images
---

# Frame_Level_Classification_of_Speech

Deep learning based neural network for speech recognition



## Getting Started

Install package manager pip, numpy, pandas, and pytorch by typing in the following command lines in cmd

```sh
python -m pip install --upgrade pip
pip install numpy
pip install pandas
pip install pytorch
```



## Dataset

14542 labels were used for training, 2200 for validation, and 2200 for testing.

![image-20210306131235320](C:\Users\euisu\OneDrive\Documents\11785_Introduction to Deep Learning\H1_Homework\P2_Git\images\image-20210306131235320.png)



## Usage

### preprocess.py

The given data is represented as the following - array of recordings/utterances. Each recording has a fixed feature length but different length of utterance.

<img src="C:\Users\euisu\OneDrive\Documents\11785_Introduction to Deep Learning\H1_Homework\P2_Git\images\image-20210306131501332.png" alt="image-20210306131501332" style="zoom: 67%;" />

There exists a relationship between the current phenome and the ones that come before and after it. Hence, when phenome is getting retrieved, a frame of phenome will be retrieved - the size of the frame is determined by the user. In order to make this process cheaper and efficient, phenomes are flattened and padded with zeros at each end like the following.

<img src="C:\Users\euisu\OneDrive\Documents\11785_Introduction to Deep Learning\H1_Homework\P2_Git\images\image-20210306120750391.png" alt="image-20210306120750391" style="zoom: 50%;" />

Same variable used in order to prevent memory running out.

### train.py

Includes functions used for training the model. 

### main.py

Essential tasks including data loading, model initializing, training, saving is all done in this file. The following is how the model was defined. Depending on the user's input on the number of neurons on each layers of the network, there exists a batch normalization, rectified linear unit (ReLU), and dropout layer.

<img src="C:\Users\euisu\OneDrive\Documents\11785_Introduction to Deep Learning\H1_Homework\P2_Git\images\carbon1.png" alt="carbon1" style="zoom:50%;" />

The purpose of putting batch normalization layer was mainly to increase the speed of training the model and to decrease the influence of the initial weights - which is expected to lead to higher accuracy of the model. Dropout layer was used in order to prevent overfitting of the model - very critical in speech recognition due to sample of data available.

### test.py

Includes functions used for testing the model. 

