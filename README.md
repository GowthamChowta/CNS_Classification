# CNS_Classification
Classify the tissue images into organ systems



Deep learning for cellular image analysis.

The paper talks about advances of computer vision and how it can solve many problems involving biological image data. 
Deep learning refers to a set of machine learning techniques that can learn effective representations of data without much human effort. For example, when you take a classification task like SVM, Logistic regression lof of manual effort needs to be done to design good features to get good performance.
Challenges with respect to deep learning.
Good quality training data. 
Effective training of deep learning models on the data. Finding which model performs better on what data is a really hard problem and there are no easy solutions.  
Deployment of trained models on the new data. 
How do you approach a problem with deep learning?
A good idea is to always start with existing models. Say, you are working on a task of segmenting the images of filamentous bacteria or fluorescent microtubules, it is always recommended to start with existing models like U-Net, Mask R-CNN as they have readily available pre-trained weights. 

Start with existing software libraries which provide pre-trained models.
What to do when training data is limited and How do the neural networks avoid overfitting?
Neural networks are very powerful and often with a small dataset or complex models the data can easily overfit. Which means the data performs really good on the training data and really bad on the test data.
Data Augmentation:
We can perform operations such as rotation, flipping, zoom, adding random Gaussian noise to increase the image diversity in a limited dataset.
Transfer learning:
A deep learning model is already trained on a large dataset. Like training a VGG model on imagenet data. Once this training is done, we can use the weights of the layer as a starting point to get the latent features for our data or we can fine-tune the model on our data starting with pre-trained weights. 
Dropout:
Adding drop out layers turns off filters during training, which regularizes the network by forcing it to not overly rely on any one feature to make predictions. The number of features to drop can be adjusted by dropout rate. 
L2 regularization:
Regularization can be applied to each weight in the neural network. It can penalize large weights and reduce overfitting. 


Common mathematical concepts of deep learning models:
Convolutions extract local features in images. 
Activation functions make it possible for learning nonlinear relationships.
Pooling operations like max-pooling can be used to downsample an image. 
The above 3 features reduce the image size and produce lower dimensional representations of image. 
Architectural elements:
Separable convolutions:
	Seperable convolution perform the convolution operation on each channel separately which reduces the computation power while preserving the accuracy.
Residual Networks:

A direct connection between two different layers. This enables to the easy flow of gradients to initial layers which helps to train really deeper networks. 

Dense networks:


Dense networks allow each layer to see every prior layer which improves error propagation and encourages both feature reuse and parameter efficiency. 





Biological applications of Deep learning.
Image classification. 



Cellular image data on
Image classification:
Image segmentation

Instance segmentation
Unet - single cell analysis.
Pixel level annotation


Object tracking
The task is complex because there are many objects to track, often hundred to thousand. Objects can touch, disappear and merge or split. These issues have made it challenging to adapt existing object-tracking algorithms to biological data. 


