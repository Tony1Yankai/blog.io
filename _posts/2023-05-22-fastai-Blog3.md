### Continue to cover more information learned from fastai Course

# Learing rate

The deault learning rate in fastai model is 10^-3.
It is a good default value but sometimes we need to find the best learning rate for different models

- Big learning rate
  This will cause the loss becomes even bigger after each iterations. That's possible because the learning rate is too big.
  
- Small learning rate
If the learning rate is too small, it may prevent the model from converging. When the learning rate is too small, 
The step size of parameter updates becomes almost zero, resulting in insufficient progress during training and the inability to find suitable parameter configurations.

There is a attribute for the learner called lr_find(), it can find the suitable learning rate automatically. Documents related to this can be found:https://docs.fast.ai/examples/app_examples.html#keypoints
``` python
learn = vision_learner(dls, resnet18, y_range=(-1,1))
learn.lr_find()
```

# Overfitting and underfitting

- Thoery

Overfitting refers to the situation where the model excessively fits the training data, resulting in good performance on the training data but poor performance on unseen new data. This means that the model is too complex and over-adapts to the noise and details in the training data, failing to generalize to new data. Overfitted models may memorize the details and noise in the training data while ignoring the overall patterns and regularities.

Underfitting, on the other hand, refers to the situation where the model performs poorly on the training data, failing to capture the overall patterns and regularities. Underfitted models are often too simple to capture the complex relationships and patterns in the data. This results in poor performance on both the training and testing data, failing to reach the desired performance level.

- Validation and test datasets

Validation set is to access whether the model is overfitting or not. As mentioned before, the dataloader can take 20% of training dataset as the validation dataset:
```python
splitter=RandomSplitter(valid_pct=0.2, seed=42)
```

Test dataset is kind of another "validation datasets". It's more like an extra insurance when the model is also 'overfitting' on the validation datasets. So if the model performances well on both validation dataset and training dataset, it can be said that the model is a good model we found.

Here is two typical pictures show what is overfitting and underfitting in Lecture4

1. Underfitting:

![image](https://github.com/Tony1Yankai/blog.io/assets/132538779/baa2766a-3f22-4c95-999e-00f165e98e83)

2. Overfitting:

![image](https://github.com/Tony1Yankai/blog.io/assets/132538779/c2bb7a8b-57f2-4648-8d54-b45ef30aaa0e)


- Possible solution to avoid overfitting

Overfitting is a very significant problem in deep learning. Here are some possible solutions to avoid the problem:

1. Increase training data: By adding more training samples, the model can better learn the data distribution and features, reducing the risk of overfitting.

2. Data augmentation: Apply various random transformations and expansions, such as rotations, translations, scaling, and flips, to the training data. This increases the diversity of the data and helps the model generalize better.

3. Regularization: Regularization introduces penalty terms in the loss function to discourage complex model parameter configurations, thereby reducing overfitting. Common regularization methods include L1 regularization and L2 regularization.

4. Early stopping: Monitor the model's performance on a validation set and stop training when the performance no longer improves, thus avoiding overfitting.

5. Dropout: During training, randomly set a fraction of the neuron outputs to zero, reducing the complexity of the neural network and the interdependencies between neurons, thereby reducing overfitting risks.

6. Model simplification: Reduce the complexity of the model by decreasing the number of network layers, neurons, or other model components, thus reducing the risk of overfitting.

7. Ensemble learning: Combine predictions from multiple models, such as voting or averaging, to reduce the overfitting of individual models.

