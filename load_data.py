from scipy import misc
import glob
from util import *

## Digits Only
n= 10
data_df = []
classes_df = []
for i in range(n):
    lst = []
    for image_path in glob.glob("./datasets/digit/digit_{}/*.png".format(i)):
        lst.append(misc.imread(image_path))
        classes_df.append(i)
    data_df.append(lst)
    print("dataset of digit {} imported.".format(i))
data_df = np.asarray(data_df)
classes_df = np.asarray(classes_df)

targets = []
for i, t in enumerate(classes_df):
    temp = np.zeros(10)
    temp[t] = 1
    targets.append(temp)
classes_df = np.asarray(targets)

np.save("./dataset/digits_samples",data_df)
np.save("./dataset/digits_classes",classes_df)
print('Data saved as csv file under name "digits_samples.npy"')
print('Classes saved as csv file under name "digits_classes.npy"')

### All
# n_digits = 10
# n_characters = 36
# data_df = []
# classes_df = []
# for i in range(n_digits):
#     lst = []
#     for image_path in glob.glob("./dataset/digit/digit_{}/*.png".format(i)):
#         lst.append(misc.imread(image_path))
#         classes_df.append(i)
#     data_df.append(lst)
#     print("dataset of digit {} imported.".format(i))
#
# for i in range(n_characters):
#     lst = []
#     for character_path in glob.glob('./dataset/character/character_{}_*/'.format(i+1)):
#         for image_path in glob.glob(character_path+'*.png'):
#             lst.append(misc.imread(image_path))
#             classes_df.append(i+n_digits)
#     data_df.append(lst)
#     print("dataset of character {} imported.".format(i+1))
#
# data_df = np.asarray(data_df)
# classes_df = np.asarray(classes_df)
#
# targets = []
# for i, t in enumerate(classes_df):
#     temp = np.zeros(n_characters+n_digits)
#     temp[t] = 1
#     targets.append(temp)
# classes_df = np.asarray(targets)
#
# np.save("./dataset/characters_digits_samples",data_df)
# np.save("./dataset/characters_digits_classes",classes_df)
# print('Data saved as csv file under name "characters_digits_samples.npy"')
# print('Classes saved as csv file under name "characters_digits_classes.npy"')