import os
import shutil

jobss = []
file1 = open(os.path.join("/data/users4/ziqbal5/abc/MILC", 'output2.txt'), 'r+')
lines = file1.readlines()
lines = [line.replace(' ', '') for line in lines]

start = '/data/users4/ziqbal5/abc/MILC/Data/'
folder2 = '/data/users4/ziqbal5/abc/MILC/training_output/'

for line in lines:
    jobss.append(str(line.rstrip('\n')))

#print(len(jobss))
n_col = len(jobss) #4  # (*2)
Ad = []
for i in range(n_col):
    for dirpath, dirnames, filenames in os.walk(start):
        for dirname in dirnames:
            if dirname.startswith(jobss[i]):
                filename = os.path.join(dirpath, dirname)
                Ad.append(filename)
Ad2 = []
for i in range(n_col):
    for dpath, _, fnames in os.walk(folder2):

        for fname in fnames:
            if fname.startswith(jobss[i]):
                fname = os.path.join(dpath, fname)
                Ad2.append(fname)

for _, data in enumerate(Ad2):
    print(data)
    data = str(data)
    os.remove(data)
for _, data in enumerate(Ad):
    print(data)
    data = str(data)
    shutil.rmtree(data)