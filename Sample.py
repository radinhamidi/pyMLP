from Core.MLP import *
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

data = load_iris()
x = data['data']
y = data['target']
y = y.reshape(y.__len__(), 1)

enc = OneHotEncoder()
enc.fit(y)
y = enc.transform(y).toarray()

train_ratio = 0.8
validation_ratio = 0.1
train_x, validate_x, test_x, ids = crossValidate(x, train_ratio, validation_ratio)
train_y = y[ids[0:int(y.__len__() * train_ratio)]]
validate_y = y[ids[int(y.__len__() * train_ratio):int(y.__len__() * (train_ratio + validation_ratio))]]
test_y = y[ids[int(y.__len__() * (train_ratio + validation_ratio)):]]

mlp = MLP(hidden_layers=[10],
          n_classes=len(train_y[0]),
          n_features=len(train_x[0]),
          activation_mode=Activation_Mode.Sigmoid,
          error_mode=Error_Mode.MSE,
          steepest_descent=Steepest_Descent.Off,
          gradient_mode=Gradient_Mode.Stochastic,
          weight_damp_factor=1
          )

mlp.train(train_x, train_y, validate_x, validate_y,
          max_epoches=100,
          eta=0.1,
          momnentum=0.9,
          accuracy_threshold=90,    # in percent
          batch_length=5
          )

q = int(input('Save model?\n1. Yes\n2. No'))
if q==1:
    model_name=input('model name:')
    mlp.save_model(model_name)