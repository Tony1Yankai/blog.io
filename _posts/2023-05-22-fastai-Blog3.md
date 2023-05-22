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
