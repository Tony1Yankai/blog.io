# More advanced information in fastai Course

## Confusion matrix
Confusion matrix is a way to access the categorize performance of the training. 
Note: It is only meaningful when doinh the categorize work.

```python
  interp = ClassificationInterpretation.from_learner(learn)
  interp.plot_confusion_matrix()
```

And it will given the result like this(here is the confusion matrix generated from Question2):
![image](https://github.com/Tony1Yankai/blog.io/assets/132538779/a0393bc5-1594-4cc8-a431-5b45334b30ef)



## Clean the data after training the model
  This is a powerful functionality in fastai. After training the model, we can detect the predictions which have a bad result(i.e. high loss):
  - Show the high loss images
```python
  interp_animal = ClassificationInterpretation.from_learner(learn_animal)
  interp_animal.plot_top_losses(10, nrows = 2, figsize = (17,4))
```
And it will generate the output like:
![image](https://github.com/Tony1Yankai/blog.io/assets/132538779/2a70ea9d-7410-4f19-8070-a8bcd10e9bde)
This table shows the 10 images having the highest loss. 
The high loss prediction has two conditions

1. The prediction result is different from the actual
2. The prediction result is correct but it's not very condident.
 
 
- clean the data using a interact method
 
 Next we can clean the data by a GUI generated from the code:
 ```python
 cleaner = ImageClassifierCleaner(learn_animal)
 cleaner
 ```
 This will generate a GUI which allows you to check whether the labeled images are correct labeled or not:
 ![image](https://github.com/Tony1Yankai/blog.io/assets/132538779/51dfb9e4-8ad9-40f3-89c7-648826e0e730)

 After deleting or replacing the label of each wrong labeled images, we can retrain the model and see if it can be improved.
 
## Model in fastai
- This is a picture shows all the model in a single picture. The related code is in the link: https://www.kaggle.com/code/jhoward/which-image-models-are-best/
And it will show the result:
![image](https://github.com/Tony1Yankai/blog.io/assets/132538779/33f94f19-9e87-44a8-9817-c64a84ae66a2)

The x axis indicates the time needs to consume when training the model and the y axis indicates the accuracy.

## Loss Function
- Concept of loss function
  
  The loss function is used to measure the discrepancy between the predicted outputs of a model and the actual labels. In machine learning and deep learning, the objective of a model is to learn the mapping from input data to output labels, aiming to minimize the difference between the predictions and the ground truth labels. The loss function quantifies the inaccuracy or error of the predictions and serves as the objective function for model training. Through optimization algorithms, the model parameters are adjusted to minimize the value of the loss function.


Here is an example given by Jeremy in Lecture3:

  1. Create a quadratic function
  ``` python
  def f(x): return 3*x**2+2*x+1
  ```
  2. Adding the noise to the data to simulate the data in real life 
  
  ``` python
  def noise(x, scale): return normal(scale = scale, size = x.shape)
  def add_noise(x, mult, add): return x*(1+noise(x,mul))+noise(x,add)
 
  
  x = torch.linspace(-2,2, step = 20)[:,None] 
  y = add_noise(f(x),0.3,1.5)  
  ```
  
  3. define a loss function to access the difference between quadratic function and real data(the data adding the noise)
  
  ``` python
  #define the MSE loss function
  def mse(preds, act): return ((preds-act)**2).mean()
   #calculate the loss
  def quad_mse(params): 
    f = mk_quad(*params)
    return mse(f(x),y)
  ```
  
   4. define a tensor to represent the coefficients
   ``` python
  #define a tensor represents the coefficient of the quadratic function
  abc = torch.tensor([1.5,1.5,1.5])
  abc.requires_grad_()
  
  #calculate the loss
  loss = quad_mse(abc)
  # return the grad attribute after running the loss.backward()
  loss.backward()
  abc.grad
   
  with torch.no_grad():
    abc -= abc.grad*0.01 #-> #Here the 0.01 is known as learning rate#
    loss = quad_mse(abc)
    
  print(f'loss = {loss:.2f}')
  ```
   
   
   
   
  
  





