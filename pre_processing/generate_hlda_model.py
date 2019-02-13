
from utils.file_operation import write_file
import numpy as np


def model_temp(content, out_path):
    hlda_model = []
    word_list = set(" ".join(content).split(" "))
    if "" in word_list:
        word_list.remove("")
    for sentence in content:
        sen_word = sentence.split(" ")
        union_word = set(sen_word)
        if "" in union_word:
            union_word.remove("")
        sen_model = str(len(union_word))
        for word in union_word:
            idx = np.where(np.array(list(word_list)) == word)
            sen_model += " " + str(idx[0][0]) + ":" + str(sen_word.count(word))
        hlda_model.append(sen_model)
    write_file(hlda_model, out_path + "/model.temp", False)
    write_file(word_list, out_path + "/words.temp", False)
