# Reading file types in subfolder
from os import walk
folder_project = './myfolder'
wavfiles = []
for (root, dirs, files) in walk(folder_project + '/audio'):
    wavfiles.extend(['{}/{}'.format(root, f) for f in files if f.endswith(".wav")])