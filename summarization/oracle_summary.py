# Coding: utf-8
from utils.data_initial import read_file
from utils.data_initial import write_file
import os
from rouge_win import run_rouge as rouge
import numpy as np


def get_oracle(doc_path, rouge_path="./data/rouge_backup/oracle/zh"):
    """

    :param doc_path: segmented and sentence splited document path.
    :param rouge_path: segmented summary path.
    :return:
    """
    doc_content = read_file(doc_path + "/word_segment.temp")
    print rouge_path + "/models/" + os.path.basename(doc_path) + "_summary.txt"
    summary_content = " ".join(read_file(rouge_path + "/models/" +
                                         os.path.basename(doc_path) + "_summary.txt"))
    len_threshold = len(summary_content.replace(" ", ""))
    sum_len = 0
    rouge_1 = []
    result = []
    for sentence in doc_content:
        write_file([sentence], rouge_path + "/systems/" + os.path.basename(doc_path) + ".txt", False)
        rouge.run_rouge(rouge_path)
        rouge_1.append(float(read_file(rouge_path + "/output/" + os.path.basename(doc_path) + ".txt.out")[2].split(" ")[3]))
    print np.array(rouge_1) / max(rouge_1)
    while sum_len < len_threshold:
        idx = int(rouge_1.index(max(rouge_1)))
        result.append(idx)
        sum_len += len(doc_content[idx].replace(" ", ""))
        rouge_1[idx] = 0.0
    result.sort()
    print result


