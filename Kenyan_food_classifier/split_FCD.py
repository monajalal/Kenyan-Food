import os
import random
from shutil import copyfile

path = 'images/'
dirs = os.listdir(path)
table = {}
for i in dirs:
    table[i] = os.listdir(path+i)


for i in table:
    random.shuffle(table[i])
    print(i, len(table[i]))

split = ['train/', 'test/']

train_files = []
for i in table:
    for j in table[i]:
        train_files.append(i+"/"+j)

random.shuffle(train_files)

k = 5

folds = [train_files[int(i*len(train_files)/k)
                         :int((i+1)*len(train_files)/k)] for i in range(k)]

for _k in range(k):
    new = "FCD_"+str(_k)+"/"

    if not os.path.exists(new):
        os.makedirs(new)

    for i in split:
        if not os.path.exists(new+i):
            os.makedirs(new+i)

    for i in split:
        for j in table:
            if not os.path.exists(new+i+j):
                os.makedirs(new+i+j)

    for ind in range(k):
        if ind == _k:
            for i in folds[ind]:
                copyfile(path+i,new+split[1]+i)
        else:
            for i in folds[ind]:
                copyfile(path+i,new+split[0]+i)

