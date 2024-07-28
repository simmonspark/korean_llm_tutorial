import os
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

abs_path = '/media/sien/DATA/DATA/dataset/korean_summary'
spesific_path = 'korean_lang'
data_dir = os.path.join(abs_path, spesific_path)


def get_data(path: str):
    '''
        :param: path
        :return: raw text data
    '''
    full = []
    for dir, _, path_list in os.walk(path):
        for name in path_list:
            full_path = os.path.join(dir, name)
            full.append(full_path)
    return full


def preprocessing(path_list):
    assert type(path_list) == list, "Input should be a list of file paths."

    input_data = []
    labels = []

    for path in path_list:
        with open(path, 'r', encoding='utf-8') as file:
            raw_json = json.load(file)
            print('---processing raw json as input...')
            # Navigating through the JSON data to find the utterances and summaries
            for dialogue in raw_json['data']:
                summary = dialogue['body']['summary']  # Extracting the summary
                for utterance in dialogue['body']['dialogue']:
                    print(utterance['utterance'])
                    input_data.append(utterance['utterance'])
                    labels.append(summary)

    return input_data, labels



preprocessing(get_data(data_dir))