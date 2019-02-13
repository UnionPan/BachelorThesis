# coding=gbk
import os
import string
from utils.file_operation import *
from utils.segmenter import StanfordSegmenter
from utils.rosette_api import api
# offset n: removing head n lines of original paper.
OFFSET = 1


class FileMerger(object):
    def __init__(self):
        self.__merged_paper = None
        self.__filtered_paper = None
        self.__sub_paper_len = [0]
        self.__seg = None
        self.__stop_word = None
        self.__titles = None

    def get_filtered_paper(self):
        if self.__filtered_paper is None:
            log.error("filtered paper is None")
            exit()
        return self.__filtered_paper

    def get_merged_paper(self):
        if self.__merged_paper is None:
            log.error("filtered paper is None")
            exit()
        return self.__merged_paper

    def get_sub_paper_len(self):
        return self.__sub_paper_len

    def get_titles(self):
        return self.__titles

    def filter_stop_word(self, segmented_sen):
        filtered_sen = []
        if self.__stop_word is None:
            self.__stop_word = read_file("./third_part/dict/stop_list.txt")
        for word in segmented_sen:
            if word in self.__stop_word:
                continue
            filtered_sen.append(word)
        return filtered_sen

    @staticmethod
    def filter_bracket(sentence):
        if "(" in sentence:
            return sentence[:sentence.find("(")] + sentence[sentence.rfind(")") + 1:]
        else:
            return sentence

    def merge_mss_2017_ros(self, dir_path):
        tmp = read_file(os.path.join("%s/%s.txt" % (dir_path, os.path.basename(dir_path))))
        self.__titles = [tmp[0]]
        self.__merged_paper = []
        self.__filtered_paper = []
        word_content = api.tokenize("".join(tmp[OFFSET:]))
        sentence_content = api.sen_tagging(" ".join(word_content))
        for sentence in sentence_content:
            filtered_sen = self.filter_stop_word(sentence.split(" "))
            if len(filtered_sen) == 0:
                continue
            self.__merged_paper.append(sentence)
            self.__filtered_paper.append(" ".join(filtered_sen))

    def merge_mss_2017(self, dir_path):
        tmp = read_file(os.path.join("%s/%s.txt" % (dir_path, os.path.basename(dir_path))))
        self.__titles = [tmp[0]]
        self.__merged_paper = []    # original paper filtered line
        self.__filtered_paper = []  # paper removed stopwords
        file_content = tmp[OFFSET:]
        filtered_file = []
        for paragraph in file_content:
            paragraph = paragraph.replace(". ", ".\n")
            for cur_sen in paragraph.split("\n"):
                sentence = self.filter_bracket(cur_sen)
                sentence = sentence.encode("utf-8").translate(string.maketrans("", ""), string.punctuation).decode("utf-8")
                seged_sen = sentence.split(" ")
                filtered_sen = self.filter_stop_word(seged_sen)
                if len(filtered_sen) == 0:
                    continue
                filtered_file.append(" ".join(filtered_sen))
                self.__merged_paper.append(cur_sen)
                self.__filtered_paper.append(" ".join(filtered_sen))

    def merge_mms_2015(self, path, language):
        if self.__seg is None:
            self.__seg = StanfordSegmenter()
        self.__merged_paper = []    # original paper filtered line
        self.__filtered_paper = []  # paper removed stopwords
        self.__titles = []
        self.__root_path = path
        for i in range(10):
            child_path = os.path.join("%s/%s%d.%s" %
                        (self.__root_path, os.path.basename(self.__root_path), i, language))
            tmp = read_file(child_path)
            self.__titles.append(" ".join(self.__seg.seg_string(tmp[0])))
            file_content = tmp[OFFSET:]
            cur_file = []
            filtered_file = []
            for paragraph in file_content:
                paragraph = paragraph.replace(u"。", u"。\n")
                paragraph = paragraph.replace(u"？", u"？\n")
                paragraph = paragraph.replace(u"！", u"！\n")
                paragraph = paragraph.replace(u"；", u"；\n")
                for sentence in paragraph.split("\n"):
                    seged_sen = self.__seg.seg_string(sentence)
                    filtered_sen = self.filter_stop_word(seged_sen)
                    if len(filtered_sen) == 0:
                        continue
                    filtered_file.append(" ".join(filtered_sen))
                    cur_file.append(sentence)
                    self.__merged_paper.append(" ".join(seged_sen))
                    self.__filtered_paper.append(" ".join(filtered_sen))
            self.__sub_paper_len.append(len(self.__merged_paper))
            log.debug("\n".join(cur_file))
#        log.debug("original paper filtered line: \n" + "\n".join(self.__merged_paper))
#        log.debug("paper removed stop words: \n" + "\n".join(self.__filtered_paper))

if __name__ == "__main__":
    a = FileMerger("./data/multiLing2015Experiment/SourceTextsWithRemoveStop/chinese/M000")
    a.merge_mms_2015("chinese")
