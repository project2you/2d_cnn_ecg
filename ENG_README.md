
<div style="text-align: center">

# English (영어)

</div>

# Arrhythmia detection based on ECG Signal by Image using 2D Deep CNN

- 2D Image reference paper : [A Hybrid Deep CNN Model for Abnormal Arrhythmia Detection Based on Cardiac ECG Signal](https://www.mdpi.com/1424-8220/21/3/951)

It is a model that converts electrocardiogram data into graph images and distinguishes electrocardiograms in 2D CNN. The model referenced the model structure of 2D CNN in the paper above. There is a difference between the data entered in the thesis and the actual data, so it is not clearly observed. The thesis is for __reference__.


# Data

The image data converted into a graph exceeds 2 GB, so uploading it through LFS is also difficult. Now that you've uploaded your code and raw data, you can use the *cutting_graph.py* file to save the file. If the process is taking too long, you can download it via the link below.

- [Graph Input data - Google drive Link](https://drive.google.com/file/d/1DjuzXjQ21p3Bhuky8ojlvzzRiAnffzvP/view?usp=sharing)

If you show an example of the input data, the image below is input.
![Normal_beat_sliced_image](./docs/fig1.png)

The image size is a bit large, but anyway I make adjustments before going into the model. It is entered as an input value of size 256 * 256. In the past, there were unnecessary parts because the plot title was added by mistake, but that part was also corrected and now only the graph is saved. The photo above is just an example of a plain beat. Not all beats look so good...

## Model 구성

![2D_CNN_Model_Structure](https://www.mdpi.com/sensors/sensors-21-00951/article_deploy/html/images/sensors-21-00951-g005.png)

If there is a difference, the number of classes output at the end is different. There were 5 (N, S, V, F, Q), but in this paper, they were classified as 8. Almost in fact, it is a progressive and smart gambler in extracting features through CNN and Maxpool layers. When grayscaled, what the resulting model sees is a characteristic of the image.

# Result

## Confusion Matrix

![Confusion Matrix Fig.1](./docs/fig2.png)
![Confusion Matrix Fig.2](./docs/fig3.png)

Different results are expected depending on the weight of the data. Class S and Class F seem to not be able to capture well because they weigh significantly less than the other classes. Here's the summary report:
```
              precision    recall  f1-score   support

       0 = N       0.99      0.99      0.99     20059
       1 = S       0.94      0.86      0.90       610
       2 = V       0.95      0.96      0.95      1553
       3 = F       0.88      0.80      0.84       182
       4 = Q       0.98      0.98      0.98      2491

    accuracy                           0.99     24895
   macro avg       0.95      0.92      0.93     24895
weighted avg       0.99      0.99      0.99     24895
```

---

# GoogLeNet (Inception)

GoogleLeNet using Inception was constructed to form a 2D CNN model. The shape and size of the model consists of:

![GoogLeNet Shape](./docs/google_net_shape.png)

It's quite complicated, but if you briefly explain GoogLeNet and Inception, Inception is a model that has been released up to v3 so far. It is used like a block model, and the model using Inception is GoogLeNet. GoogLeNet != Inception so don't get confused.

Inception consists of the following:

![Inception](https://cdn.analyticsvidhya.com/wp-content/uploads/2018/10/Screenshot-from-2018-10-17-11-14-10.png)

As you can see, it is a continuation of the convolutional layer and the MaxPooling layer. A model that has been merged multiple times is called GoogleLeNet. It goes deep like the movie Inception. MaxPooling is to reduce the computational amount of the Convolution Layer. This is a way to reduce the amount of computation because it takes too long when the amount of computation is too much.

## Result

- Train and Validation Accuracy, Loss

![Train and Validation Loss](./docs/googleNetFig1.png)
![Train and validation accuracy](./docs/googleNetFig2.png)

- Confusion Matrix

![Confusion Matrix as count](./docs/googleNetFig3.png)
![Confusion Matrix as ratio](./docs/googleNetFig4.png)

```
             precision    recall  f1-score   support

       0 = N       0.98      1.00      0.99     20059
       1 = S       0.91      0.86      0.89       610
       2 = V       0.97      0.90      0.94      1553
       3 = F       0.70      0.80      0.75       182
       4 = Q       0.99      0.94      0.97      2491

    accuracy                           0.98     24895
   macro avg       0.91      0.90      0.90     24895
weighted avg       0.98      0.98      0.98     24895
```

## Result

The resulting value is the resulting value of the spectogram as input. In the past, a model with a raw graph image inserted was changed to a spectogram. Enter RGB as is, not GrayScale.

- Train and Validation Accuracy, Loss

![Train and Validation Loss](./docs/spectrogramFig1.png)
![Train and validation accuracy](./docs/spectrogramFig2.png)

- Confusion Matrix

![Confusion Matrix as count](./docs/spectrogramFig3.png)
![Confusion Matrix as ratio](./docs/spectrogramFig4.png)

```
            precision    recall  f1-score   support

       0 = N       0.99      0.99      0.99     20059
       1 = S       0.92      0.89      0.90       610
       2 = V       0.94      0.96      0.95      1553
       3 = F       0.86      0.77      0.81       182
       4 = Q       0.99      0.99      0.99      2491

    accuracy                           0.99     24895
   macro avg       0.94      0.92      0.93     24895
weighted avg       0.99      0.99      0.99     24895
```
