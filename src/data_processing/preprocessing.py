import os
from fnmatch import filter
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist
import csv

def get_dataset_text(path,  pattern = "*"):
    """
    get the complete concatenated text from the siamese dataset
    :param path: path to the file or folder containing the dataset batches
    :param pattern: pattern to select the type of file
    :return: text
    """
    if os.path.isdir(path):
        temp_file_list = filter(os.listdir(path), pat=pattern)
        file_list = [path+'/'+file for file in temp_file_list]
    elif os.path.isfile(path):
        file_list = [path]
    else:
        print("Error! Path is neither a folder nor a file! Check Path")
    text = ""
    for file in file_list:
        print(file)
        data = pd.read_csv(open(file))
        p1 = " ".join(data['para1_text'].to_list())
        p2 = " ".join(data['para2_text'].to_list())
        text = text + p1 + " " + p2
    return text


def get_word_index_list(text , n = 100, remove_stopwords = True, remove_punc = False):
    """
    Selects most frequent n words from given text.
    :param text: given text
    :param remove_stopwords: flag indicating whether to remove stopwords
    :param n: number of words to return
    :return: word index list of selected most frequent n words
    """
    word_tokens = word_tokenize(text.lower())
    print(word_tokens)
    filtered_word_tokens = word_tokens
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        filtered_word_tokens = [word for word in word_tokens if not word in stop_words]
    if remove_punc:
        filtered_word_tokens = [word for word in filtered_word_tokens if word.isalpha()]
    fdist = FreqDist(filtered_word_tokens)
    return fdist.most_common(n)


def write_index_to_file(word_index, file_path):
    with open(file_path, 'w', newline='') as file:
        wr = csv.writer(file, quoting=csv.QUOTE_ALL)
        wr.writerow(word_index)


def read_index_as_string(file_path):
    with open(file_path) as file:
        contents = file.read() + '\n'
    return contents