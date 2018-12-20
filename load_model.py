from util import *
from MLP import *
import glob

model = MLP([10],2,1,Activation_Mode.Sigmoid,Error_Mode.MSE,Steepest_Descent.Off,Gradient_Mode.Batch)
model_names = []
for path in glob.glob('./models/*.npy'):
    model_names.append(path)
print('Please enter you model number form list below:')
for i, path in enumerate(model_names):
    print('{}. {}'.format(i,path))
model_number = int(input('?'))
model.load_model(model_names[model_number])

data_set_number = int(input('Choose dataset:\n1. Digits\n2. Alphabets & Digits'))

if data_set_number==1:
    data = np.load("./Dataset/digits_samples.npy")
    y = np.load("./Dataset/digits_classes.npy")
    x = data.reshape((1700 * 10, 1024))
elif data_set_number==2:
    data = np.load("./Dataset/characters_digits_samples.npy")
    y = np.load("./Dataset/characters_digits_classes.npy")
    x = data.reshape((1700*46, 1024))

svd_acc = float(input('enter SVD compression accuracy:'))
x, [U, S, V] = SVD_compress(x, svd_acc)
x = normalize(x)

print('*'*10,'\nOverall Accuracy for dataset is:',model.test(x,y))