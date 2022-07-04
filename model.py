#Dataset link: https://s3-ap-southeast-1.amazonaws.com/he-public-data/dataset52a7b21.zip

import pandas as pd
import csv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from tqdm import tqdm
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed


def save_to_file(data, f):
    import pickle
    if "io" not in str(type(f)):
        f = open(f, "wb")
    pickle.dump(data, f)
    f.close()

def load_from_file(f):
    import pickle
    if "io" not in str(type(f)):
        f = open(f, "rb")
    data = pickle.load(f)
    f.close()
    return data

def get_words_from_dataset(dataset):
    data_list = (dataset['TITLE'].tolist() + dataset['DESCRIPTION'].tolist() + 
                    dataset['BULLET_POINTS'].tolist() + dataset['BRAND'].tolist())
    words_list = Counter()
    print('Getting words to make a dict')
    for item in tqdm(data_list):
        if type(item) is str:
            words_list.update(re.findall(r'\w+', item))
    
    save_to_file(words_list, 'words.pickle')

    top_words_list = [word for word, word_count in words_list.most_common(200000)]
    return top_words_list

def make_one_hot_encoder(feature_words):
    feature_words = [[x] for x in feature_words]
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(feature_words)
    return enc

def encode_training_data_element(words, encx):
    if words:
        encoded = encx.transform(words)
        encoded_element = sparse.csr_matrix(encoded.sum(axis=0))
        return encoded_element
    else:
        return None

def get_row_data_lists(data):
    data_rows = []
    rows, cols = data.shape
    print('Getting row data')
    for item in tqdm(range(rows)):
        temp_list = Counter()
        temp_list.clear()
        row_words = []
        if(type(data.iloc[item]['TITLE']) is str):
            temp_list.update(re.findall(r'\w+', data.iloc[item]['TITLE']))
        if(type(data.iloc[item]['DESCRIPTION']) is str):
            temp_list.update(re.findall(r'\w+', data.iloc[item]['DESCRIPTION']))
        if(type(data.iloc[item]['BULLET_POINTS']) is str):
            temp_list.update(re.findall(r'\w+', data.iloc[item]['BULLET_POINTS']))
        if(type(data.iloc[item]['BRAND']) is str):
            temp_list.update(re.findall(r'\w+', data.iloc[item]['BRAND']))
        row_words_list = [[word] for word in temp_list]
        data_rows.append(row_words_list)
    return data_rows

def make_one_hot_labels(id_list):
    enc = OneHotEncoder(handle_unknown='error')
    enc.fit(id_list)
    return enc

def get_unique_ids(data):
    id_list = data['BROWSE_NODE_ID'].tolist()
    id_list = [str(x) for x in id_list]
    id_list = list(set(id_list))
    id_list = [[x] for x in id_list]

def encode_labels(id, ency):
    if id:
        encoded_element = ency.transform([[id]])
        return encoded_element
    else:
        return None

def main():
    data = pd.read_csv("train.csv", escapechar = "\\", quoting = csv.QUOTE_NONE)
    words_list = get_words_from_dataset(data)

    #-------------------------------------------

    encx = make_one_hot_encoder(words_list)
    save_to_file(encx, 'encx.pickle')

    row_data_lists = get_row_data_lists(data)
    print('Encoding training data')
    processedX = Parallel(n_jobs=1)(delayed(encode_training_data_element)(element, encx) 
                                    for element in tqdm(row_data_lists))
    X = sparse.vstack(processedX)

    #-------------------------------------------

    unique_id_list = get_unique_ids(data)
    ency = make_one_hot_labels(unique_id_list)
    save_to_file(ency, 'ency.pickle')

    y = []
    id_list = data['BROWSE_NODE_ID'].tolist()
    id_list = [str(x) for x in id_list]
    print('Encoding training labels')
    for index in tqdm(range(len(id_list))):
        encoded_y = encode_labels(id_list[index], ency)
        y.append(encoded_y)
    y = sparse.vstack(y).toarray()

    #-------------------------------------------

    seed = 1
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=seed)
    y_train, y_test = train_test_split(y, test_size=0.2, random_state=seed)

    rforest = RandomForestClassifier(max_depth=8, n_estimators=100, random_state=seed)
    rforest.fit(X_train, y_train)

    print("Accuracy: ", rforest.score(X_test, y_test))

if __name__ == '__main__':
    main()