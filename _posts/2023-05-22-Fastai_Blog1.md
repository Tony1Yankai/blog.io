# Learning Blog for Fundamental Fastai Course


## What is neural network?

  Compared to machine learning, neural network can learn features instead of giving the features manully. Here is an architecture of neural network:
![image](https://github.com/Tony1Yankai/blog.io/assets/132538779/42ddfac8-0084-4255-948e-a3f15054a381)

Here are some further explanation for the picture:
- model: basically the model is some( or large amount of) mathematical equations
- result : based on the input to the model, there will be a prediction result represented the output of the model.
- loss : The loss will be the comparison between the actual result and the prediction result. So it is also used to access the performance of the prediction.
- weights: The weights will update based on the loss.

Once the weights are updated, then the model will use the new weight and input to start with the next iteration. After many iteration, ideally the loss will become very small.
At that time, it means the model has a goood prediction result, which is closly to the actual(correct) result. 


## Fastai Basic

- Fastai fundamental
  1. Fastai is a library based on pytorch. Pytorch is replacing the tensorflow recently.
  2. Fastai is a very powerful library. Compared to the traditional neural network coding, it has a large amount of API helps coder to build neural network model in a quick speed.
  3. Fastai resouce can be found on the fastai website in https://docs.fast.ai/

- Fastai basic API implementation
  ### Get data to model using dataloader
  
  - A API called *dataloader* can be used to get the training set to the model.Here is a basic example of dataloader:
  ```python
    dls = DataBlock(
      blocks = (ImageBlock, CategoryBlock),
      get_items = get_image_files,
      splitter = RandomSplitter(valid_put = 0.2, seed = 42),
      get_y = parent_label,
      items_tfms = [Resize(192, method = 'squish')]
    ).dataloader(path)
    
Here are some further explanation of the variable in dataloader:


|Variable| Explanation |
|-|-|
|blocks | define what the work the model does. Here ImageBlock indicates the input will be pictures, and the CategoryBlock indicates the model is to categorize the pictures |
| get_items | using get_image_files to get all the images in the specific pile folder |
| splitter | using RandomSplitter to  split the validation data from the datasets, here 20% data is used as validation set |
| get_y | getting all the label from the parent folder | 
|itmes_tfms| resize te images using 'squish' method |

more information about dataloader can be found in https://docs.fast.ai/data.load.html

  ### The API Fine_tune()
In general, the purpose of the fine_tune method is to utilize the feature representation capability of a pre-trained model through transfer learning and fine-tuning. 
It involves training the model on a new dataset to improve its performance and generalization.


    
    
    
    
    
