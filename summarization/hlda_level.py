from base_class.jpype_unit import *


class LevelScore(UseJavaUnit):
    def __init__(self):
        UseJavaUnit.__init__(self)
        abstract_htw = JPackage('summaryByFeatures').Abstract_htw
        self.__class_for_level = abstract_htw()
        self.__file = None
        self.__score_map = None

#    def get_sentence_level_score(self, idx):
#        idx_java = java.lang.Integer(idx)
#        return self.__score_map.get(idx_java).floatValue()

    def get_paper_level_score(self, file_path, run_name="run000"):
        self.__file = java.io.File(file_path)
        self.__score_map = self.__class_for_level.getLevelScore(self.__file, run_name)
        level_list = []
        for i in range(self.__score_map.size()):
            idx_java = java.lang.Integer(i)
            level_list.append(self.__score_map.get(idx_java).floatValue())
        return level_list

