### More Fastai Course information

# Build the model from scratch

The library pytorch, numpy, pandas, matplotlib are 4 commonly used library for machine/deep learning when doing the data mining. 
The notebook can be seen through the link: https://www.kaggle.com/code/jhoward/linear-model-and-neural-net-from-scratch

Here is a roughly process of the steps:

- clean the data

1. Load the csv file 

``` python
df = pd.read_csv(path/'train.csv')
```

2. Find all the Null data and replace it

``` python
#find the null value numbers of each row
df.isna().sum()

#replace the null value with the same value type fitting in that position
modes = df.mode().iloc[0]
df.fillna(modes, inplace=True)

#check again whether the null value is existed
df.isna().sum() # This should return all 0
```

3. Clean the big data. Using the histogram to check the value attribution

``` python
df['Fare'].hist();

#take the logarithm to fix this and plus 1 to avoid log(0)
df['LogFare'] = np.log(df['Fare']+1)
```

4. Deal with the data which can't do multiplication

``` python
df = pd.get_dummies(df, columns=["Sex","Pclass","Embarked"])
added_cols = ['Sex_male', 'Sex_female', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
df[added_cols].head()
```

5. Create the independent (predictors) and dependent (target) variables

``` python
from torch import tensor
t_dep = tensor(df.Survived)
indep_cols = ['Age', 'SibSp', 'Parch', 'LogFare'] + added_cols
t_indep = tensor(df[indep_cols].values, dtype=torch.float)
```

- Setting up a linear model

1.  a coefficient is needed for each column in t_indep. We'll pick random numbers in the range (-0.5,0.5)

``` python
torch.manual_seed(442)
n_coeff = t_indep.shape[1]
coeffs = torch.rand(n_coeff)-0.5

t_indep*coeffs
t_indep = t_indep / vals
```

2. Set a loss function

``` python
loss = torch.abs(preds-t_dep).mean()

#calculating predictions, and loss
def calc_preds(coeffs, indeps): return (indeps*coeffs).sum(axis=1)
def calc_loss(coeffs, indeps, deps): return torch.abs(calc_preds(coeffs, indeps)-deps).mean()
```

3. gradient decent step

``` python
coeffs.requires_grad_()
loss = calc_loss(coeffs, t_indep, t_dep)
loss.backward()

with torch.no_grad():
    coeffs.sub_(coeffs.grad * 0.1)
    coeffs.grad.zero_()
    print(calc_loss(coeffs, t_indep, t_dep))
```

4. training the linear model

``` python
from fastai.data.transforms import RandomSplitter
trn_split,val_split=RandomSplitter(seed=42)(df)

trn_indep,val_indep = t_indep[trn_split],t_indep[val_split]
trn_dep,val_dep = t_dep[trn_split],t_dep[val_split]
len(trn_indep),len(val_indep)

#define some functions 
def update_coeffs(coeffs, lr):
    coeffs.sub_(coeffs.grad * lr)
    coeffs.grad.zero_()
    
def one_epoch(coeffs, lr):
    loss = calc_loss(coeffs, trn_indep, trn_dep)
    loss.backward()
    with torch.no_grad(): update_coeffs(coeffs, lr)
    print(f"{loss:.3f}", end="; ")

def init_coeffs(): return (torch.rand(n_coeff)-0.5).requires_grad_()

#train the model
def train_model(epochs=30, lr=0.01):
    torch.manual_seed(442)
    coeffs = init_coeffs()
    for i in range(epochs): one_epoch(coeffs, lr=lr)
    return coeffs
 
 #train the model and print the loss each poch
coeffs = train_model(18, lr=0.2) 
```

The output loss:
![image](https://github.com/Tony1Yankai/blog.io/assets/132538779/717f73a7-754a-45db-bf64-87aa8f6dc877)
We can see the loss does decreases.

- Measure the acurracy

```python
preds = calc_preds(coeffs, val_indep)

#check the average accuracy
results.float().mean()
```

- Using sigmoid to cast the output from 0 to 1

``` python
# modify the calc_preds
def calc_preds(coeffs, indeps): return torch.sigmoid((indeps*coeffs).sum(axis=1))
coeffs = train_model(lr=100)

```
