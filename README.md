# Deep-Learning 

## Synopsis

This is the summary of implementations for Deep Learning specialization by deeplearning.ai at coursera - https://www.coursera.org/specializations/deep-learning Source code is not posted on this public repository as per the coursera Deep Learning specialization honor code.

## Implementations (Author: Rohit Jain)

`python` `tensorflow` `keras`

**Neural Networks and Deep Learning**

* *Classifier to recognize cats - Logistic Regression with a Neural Network mindset*
	`Parameter initialization, compute cost function and its gradient, parameter update using gradient descent`

* *Planar data classification - Neural Network with a hidden layer*
	`Build neural network with a hidden layer. use a non-linear unit, Implement forward propagation and backpropagation, train neural network. See the impact of varying the hidden layer size, including overfitting.`

* *Cat vs Non-Cat Image classification - Deep Neural Network*
	`ReLU activation, implement all the functions required to build a deep neural network. Build a two-layer neural network, Build an L-layer deeper neural network`

**Improving Deep Neural Networks. Hyperparameter tuning, Regularization and Optimization**

* *Gradient Checking, Initialization and Regularization*
	`Gradient checking by backpropagation and numerical approximation. Implement Zeros initialization, Random initialization, He initialization, Xavier initialization, L2 regularization and Dropout.`

* *Optimization methods*
	`Implement Optimization algorithms mini-batch gradient descent, Momentum and Adam. Implement a model with each of these optimizers and observe the difference.`

* *Decipher hand gesture signs (Image Classification) - Neural Network in tensorflow*
	`Implement linear function, sigmoid, compute cost, one-hot encoding, initialization. Create the computation graph, create and initialize a session and run the session to execute the graph.`
   
**Structuring Machine Learning Projects**
* *Case Studies*
    *Bird recognition in the city of Peacetopia*
    `Evaluation metrics, satisficing and optimizing metrics, train/dev/test distributions, tuning dev/test sets and metrics, avoidable bias,understanding human-level performane and surpassing, improving your model`
    *Autonomous driving*
    `error analysis, cleaning incorrect labels, build and iterate quickly, train/test on different distributions, bias and variance with mismatched data distributions, address data mismatch, transfer learning, multi-task learning, end-to-end deep learning`
   
**Convolutional Neural Networks**
* *Image Classification - Convolution Neural Network*
	`Implement Convolution layer, zero Padding, single step convolution, convolution forward pass, convolution backward pass. Implement Pooling layer, Max Pooling, Average Pooling, forward pass, Pooling backward pass`

* *Face Recognition - Convolution Neural Network in Keras*
	`Build a model in keras, compile the model, train the model on train data, test the model on test data, visualize model summary and model graph`

* *Decipher hand gesture signs (Image Classification) - Residual Networks (ResNets) in Keras*
	`Implement Resnet identity-block and convolution-block, build Resnet model(50 layers), train the model on train data, test the model on test data, visualize model summary and model graph`

* *Autonomous driving application, Car detection - Object detection through YOLO model in tensorflow*
	`Implement YOLO box filters on class score threshold, Non-max suppression, Intersection over Union (IoU) thresholding, Process Deep CNN output through YOLO filtering, Test YOLO pretrained model on images`

* *Face verification and recognition system - Siamese Network*
	`Implement the triplet loss function, Use a pretrained model to map face images into 128-dimensional encodings, Use these encodings to perform face verification and face recognition`

* *Generate novel artistic images - Neural Style Transfer in tensorflow*
	`Compute content cost, compute style/gram matrix and style cost, Define the total cost to optimize, Implement Neural Style Transfer using VGG16 model pretrained on ImageNet database to synthesize new images`

**Sequence Models**

* *Build Recurrent Neural Network - RNN, LSTM* 
	`Implement one time-step of the RNN. forward pass for RNN, RNN bacward pass, implement LSTM cell for a single time-step, forward pass for LSTM, LSTM backward pass`

* *Generate dinosaur names - Character level language model RNN*
	`Implement Gradient clipping to avoid exploding gradients and Sampling to generate characters, Forward propagate RNN, compute loss, Backward propagate through time, compute gradients of loss wrt parameters, clip gradients, Update parameters using gradient descent, train the model, generate new names`

* *Generate Jazz music - RNN with LSTM in Keras*
	`Build the LSTM model, train the model on training data/jazz-music, Implement sampling for a sequence of musical values, define your inference model, Use the inference model to predict an output of new musical values which are then post-processed into midi music`

* *Solve word analogy problems - Word embeddings*
	`Load pre-trained word vectors and measure similarity using cosine similarity, Use word embeddings to solve word analogy problems, Modify word embeddings to reduce their gender bias`

* *Build an Emojifier, mapping sentence to emoji- RNN with LSTM in Keras*
	`Build a simple baseline classifier, build the Emojifier model, train the model to learn softmax parameters, examine test set performance, visualize confusion matrix. Build the Embedding layer in Keras using pre-trained word vectors, build Emojifier model in Keras, train the model and then evaluate/test on test data.`

* *Dates translator (Neural machine translation) - Attention model with bi-directional LSTM in Keras*
	`Implement one step attention, create Keras model, define loss, optimizer and metrics, fit the model and evaluate on test set, visualize the attention values in the network`

* *Trigger Word Detection (wake up upon hearing a certain word) - Uni-directional RNN with GRU in Keras*
	`Structure a speech recognition project, Synthesize and process audio recordings to create train/dev datasets, Train a trigger word detection model and make predictions`
