from jpype import *


class UseJavaUnit(object):
    def __init__(self):
        self.__jvm_path = getDefaultJVMPath()
        if not isJVMStarted():
            class_path = "-Djava.class.path="
            class_path += "./third_part/java/hLDAForSummary.jar"
            class_path += ";third_part/java/stanford_seg/stanford-segmenter-3.6.0.jar"
            class_path += ";third_part/java/stanford_seg/stanford-segmenter-3.6.0-sources.jar"
            startJVM(self.__jvm_path, "-ea", class_path)

    def __del__(self):
        if isJVMStarted():
            shutdownJVM()

