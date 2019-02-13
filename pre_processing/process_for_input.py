from utils.file_operation import read_file
from utils.file_operation import write_file
import jieba


def get_lda_input_format():
    file_content = read_file("..\\data\\sample_datas\\evasampledata4-TaskAA.txt")
    lda_format = list()
    cur_target = ""
    count = 0
    for cur_line in file_content:
        if cur_line == "":
            continue
        (id_, target, text, stance) = cur_line.split("\t")
        if cur_target == "":
            cur_target = target
            count = 0
        if cur_target != target:
            cur_target = target
            count = 0
        if count < 500:
            lda_format.append(text)
            count += 1
#        if target != "IphoneSE":
#            continue
#        lda_format.append(" ".join(jieba.cut(text)))
    write_file(lda_format, "../nlpcc2016.txt")

if __name__ == "__main__":
    get_lda_input_format()
