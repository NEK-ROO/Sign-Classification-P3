

# **P3 Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[explore]: ./write_up_images/explore1.png

[preprocess]: ./write_up_images/pre_origin.png

[test_all]: ./write_up_images/test_all.png
[test_pred]: ./write_up_images/test_pred.png
[test_top5]: ./write_up_images/test_top5.png

[exp1]: ./write_up_images/exploration_1.png
[exp2]: ./write_up_images/exploration_2.png
[exp3]: ./write_up_images/exploration_3.png
[exp4]: ./write_up_images/exploration_4.png
[exp5]: ./write_up_images/exploration_5.png
[exp6]: ./write_up_images/exploration_6.png

[brightness_hist]: ./write_up_images/brightness_hist.png
---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! And here is a link to my [project code](./Traffic_Sign_Classifier.ipynb)

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

**Explore Bad Patterns**  
Random checked 200 images and found out some patterns of images' crashing. And here are some examples of them.

| Index         	|     Problem	        					| 
|:---------------------:|:---------------------------------------------:| 
| 6223        			| High contrast (bright outside)								| 
| 29041				    | High contrast (bright inside) 							|
| 19591     				| Partly high contrast									|
| 28593					| Multiple signs											|
| 7657	      			| Fog (weather)					 				|
| 6302				    | Extremely dark      							|
| 15120				    | Dark      							|

![alt text][exp1]
![alt text][exp2]
![alt text][exp3]
![alt text][exp4]
![alt text][exp5]
![alt text][exp6]

**Brightness**  
First have a overview of brightness of training data. Because brightness was a avarage of all pixels, I could not catch a lot detail, but I noticed that variance of that was relatively high.

![alt text][brightness_hist]
Here is an exploratory visualization of the data set. Some Images are Dark, so some kinds of pre-processing about brightness might be needed.

![alt text][explore]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

I only balanced the brightness of images to make them not too bright or dark. 

**Grayscale** was so simple, it actually made the result better and training faster. But since I noticed some images were too dark and some were bright, I thought I should balance it and believed it would help my model work better.

Then I chose `CLAHE` for pre-processing because it could normalize the brightness according to the areas' brightness, which meant it could probably make dark areas brighter or bright areas darker. Here is examples of images pre-processed before and after.

![alt text][preprocess]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 brightness balanced gray image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x16 	 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x16 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 16x16x32 	 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x16 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 8x8x64         |
| RELU					|                                                 |
| Max pooling	      	| 2x2 stride,  outputs 4x4x64                     |
| Fully connected		| (1024, 312)     									|
| RELU           		|                									|
| Dropout       		| 0.8 keep probability                                 |
| Fully connected		| (312, 256)     									|
| RELU           		|                									|
| Dropout       		| 0.6 keep probability                                 |
| Fully connected		| (256, 128)     									|
| RELU           		|                									|
| Dropout       		| 0.4 keep probability                                 |
| Softmax				|        		


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the **cross entropy** as my model's loss function and initial the **learning rate** to **5e-4** (normal setting for `AdamOptimizer`). 

And `BATCH_SIZE` was 64, which was also a common choice. I haved tried to set BATCH_SIZE bigger to make the training faster but the learning rate was hard to configure.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 98.40%
* validation set accuracy of 93.70%
* test set accuracy of 92.57%

If an iterative approach was chosen:
* **What was the first architecture that was tried and why was it chosen?**  
LeNet was the first try on this dataset because MNIST is a very similar dataset, so I thought LeNet should give me a good baseline result.
* **What were some problems with the initial architecture?**  
It only got 82% accuracy on validation set, and it was far too away from the specification of 93%.
* **How was the architecture adjusted and why was it adjusted?**  
Almostly, I tried to built a deeper neural network, but training was slow, so I applied another useful algorithms named `dropout` to speed it up and for regularization.
* **Which parameters were tuned? How were they adjusted and why?**  
Learning rate was so important to fine tune it. And dropout rate was also needed to pay attention to.  
I set learning rate of 5e-4, and it took over 330 epochs to get to the best. Obviously, with a lower learning rate, it might be much longer than that. But in contrast, if the learning rate was too large, the training would be unstable and sometimes probably to step back to the initial point.
* **What are some of the important design choices and why were they chosen?**  
Convolutional layers could consider positional infomation naturally (compared to mlp layers), which should be the most important reason that CNN had been the most popular architecture in vision recognition.


If a well known architecture was chosen:
* **Why did you believe it would be relevant to the traffic sign application?**  
I just compared it to MNIST and cifar-10 dataset, no matter size of images or categories of images, they are very similar dataset, and which made me very confident in a deeper, AlexNet like, network should work well, too.
* **How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?**  
Both of them were higher than my baseline (LeNet), and relatively close to the accuracy of training set.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][test_all]

The first image might be difficult to classify because that was in a relatively wierd angle for training data, so it was not surprised that it did not make it. 

And the fifth image might also be difficult to classify because there is some snow over the sign.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

![alt text][test_pred]
Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work      		| Yield      									| 
| Road work     		| Road work 									|
| Stop					| Stop											|
| 30 km/h	      		| 30 km/h      					 				|
| Children crossing		| Children crossing      						|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 92.57%. It was reasonable because five images were too little to be a good test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .00         			| Road work   									| 
| .00     				| Right-of-way at the next intersection										|
| .00					| Keep left											|
| .00	      			| General caution					 				|
| 1.0				    | Yield      							|


For the second image:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .00         			| Traffic signals   									| 
| .00     				| Bicycles crossing										|
| .00					| Children crossing										|
| .20	      			| Bumpy road					 			       |
| .80				    | Road work      							|

For the third image: 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .00         			| Spped limit (30km/h)   									| 
| .00     				| Roundabout mandatory									|
| .00					| Priority road										|
| .00	      			| No entry      				 				|
| 1.0				    | Stop      							|

For the fourth image:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .00         			| Spped limit (60km/h)  									| 
| .00     				| Spped limit (20km/h)										|
| .00					| Spped limit (70km/h)										|
| .00	      			| Spped limit (50km/h)					 				|
| 1.0				    | Spped limit (30km/h)      							|

For the fifth image: 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .00         			| General caution  									| 
| .00     				| Right-of-way at the next intersection											|
| .00					| Road narrows on the right										|
| .00	      			| Bicycles crossing				 				|
| 1.0				    | Children crossing     							|

Probabilities of not correct classes werer too little (about 1e-10), so the probablities were approximately 1.0. But it was kind of wierd that the first image was classified to 'Yield' with probability of 1.0

