import os
from pathlib import Path
from sklearn.model_selection import train_test_split

abs_path = '/media/sien/DATA/DATA/dataset/korean_summary'
spesific_path = 'korean_lang'
data_dir = os.path.join(abs_path,spesific_path)

def get_data(path : str):
    '''
        :param: path
        :return: raw text data
    '''
    full = []
    for dir, _ , path_list in os.walk(path):
        for name in path_list:
            full_path = os.path.join(dir,name)
            full.append(full_path)
    return full

def preprocessing(data) :
    pass


