# Coding: utf-8
from utils.log_custom import log
import re
import os
import platform
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


__pattern = re.compile(r"\r|\r\n|\n")


def read_file(file_path):
    file_ans = list()
    try:
        file_object_ = open(file_path, 'rb')
    except IOError as e:
        log.error(e)
    else:
        try:
            file_content = file_object_.read()
            file_content = __pattern.sub('\n', file_content)
            for cur_line in file_content.split('\n'):
                if cur_line == "":
                    continue
                file_ans.append(cur_line.strip().decode("utf-8"))
        finally:
            file_object_.close()
    return file_ans


def write_file(file_conent, file_path, if_convert=True):
    try:
        output = open(file_path, 'w')
    except IOError as e:
        log.error(e)
        exit()
    finally:
        for cur_line in file_conent:
            new_line = (cur_line + u"\n").encode("utf-8")
            # if if_convert:
            #     output.write(new_line.decode('utf-8').encode('GB2312', 'ignore'))
            # else:
            #     output.write(new_line)
            output.write(new_line)
        output.close()


def __remove_all(path):
    for file in os.listdir(path):
        file_path = os.path.join("%s/%s" % (path, file))
        if os.path.isfile(file_path):
            os.remove(file_path)
        else:
            __remove_all(file_path)
    os.removedirs(path)


def flush_dic(path):
    if not os.path.exists(path):
        os.makedirs(path)
    elif os.path.isdir(path):
        for file in os.listdir(path):
            file_path = os.path.join("%s/%s" % (path, file))
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                __remove_all(file_path)
    else:
        raise IOError(path + " is not dictionary")


def parse_xml(path):
    tree = ET.ElementTree(path)
    tree.getroot()


if __name__ == "__main__":
    flush_dic("F:/new_tests/")
#    for line in read_file("..\\data\\sample_datas\\evasampledata4-TaskAA.txt"):
#        print line

