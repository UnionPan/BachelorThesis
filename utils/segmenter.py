# coding: 'utf-8'
from base_class.jpype_unit import *
from utils.log_custom import log
import nltk


class NLTKSegmenter(object):
    def __init__(self):
        print ""

    def seg_string(self, in_str):
        nltk.word_tokenize(in_str, "")
        print ""


class StanfordSegmenter(UseJavaUnit):
    def __init__(self):
        UseJavaUnit.__init__(self)
        crf_classifier = JPackage("edu.stanford.nlp.ie.crf").CRFClassifier
        self.__data_path = "third_part/java/stanford_seg/data"
        self.__props = java.util.Properties()
        self.__props.setProperty("sighanCorporaDict", self.__data_path)
        self.__props.setProperty("serDictionary", self.__data_path + "/dict-chris6.ser.gz")
        self.__props.setProperty("inputEncoding", "UTF-8")
        self.__props.setProperty("sighanPostProcessing", "true")
        self.__segmenter = crf_classifier(self.__props)
        self.__segmenter.loadClassifierNoExceptions(self.__data_path + "/ctb.gz", self.__props)

    def seg_string(self, in_string):
        if not isinstance(in_string, unicode):
            raise TypeError("type must be unicode")
        return list(self.__segmenter.segmentString(in_string))

    def seg_paper(self, paper_list):
        segmented_paper = []
        for sentence in paper_list:
            segmented_paper.append(self.seg_string(sentence))
        return segmented_paper
