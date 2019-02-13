# coding=utf-8
from utils.file_operation import read_file
from utils.file_operation import write_file
from utils.rosette_api import api
from nltk.tokenize import word_tokenize
from utils.parse_xml import ParseXML
from utils.log_custom import log
from pre_processing.generate_hlda_model import model_temp
import os
import time


# rosette_api
def __processing_using_ros(original_path="../data/origin",
                           data_backup_path="../data/ros_result"):
    px = ParseXML()
    for cur_file in os.listdir(original_path):
        dir_name = cur_file.split(".")[0]
        if dir_name == "test":
            continue
        out_path = os.path.join("%s/%s/" % (data_backup_path, dir_name))
        if os.path.exists(out_path + "/tokenized_body.temp") and \
                os.path.exists(out_path + "/tokenized_title.temp") and False:
            continue
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        #px.parse(os.path.join("%s" % (cur_file)))
        # cur_content = read_file(os.path.join("%s/%s" % (lang_path, cur_file)))
        contents = px.all_content
        print contents
        titles = px.title
        word_segmented = api.tokenize(u" ".join(contents))
        write_file(contents, os.path.join("%s/%s.txt" % (out_path, dir_name)), False)
        write_file(word_segmented, out_path + "/tokenized_body.temp", False)
        write_file(word_segmented, out_path + "/lemmatized_body.temp", False)
        word_segmented = api.tokenize(u" ".join(titles))
        write_file(word_segmented, out_path + "/tokenized_title.temp", False)
        write_file(word_segmented, out_path + "/lemmatized_title.temp", False)



# ["ar", "cs", "de", "el", "en", "es", "fa", "fr", "it", "ja", "ko", "nl", "no", "pl", "pt", "ro", "ru", "th", "tr", "zh"]:
# ["af", "az", "bg", "bs", "ca", "eo", "eu", "fi", "hr", "id", "jv", "ka", "li", "lv", "mr", "ms", "nn", "simple", "sk", "sr", "tt", "uk"]:


def __processing_using_nltk(original_path,
                            data_backup_path):
    px = ParseXML()
    for cur_file in os.listdir(original_path):
        dir_name = cur_file.split(".")[0]
        if dir_name == "test":
            continue
        out_path = os.path.join("%s/%s/" % (data_backup_path, dir_name))
        if os.path.exists(out_path + "/tokenized_body.temp") and \
                os.path.exists(out_path + "/tokenized_title.temp") and False:
            continue
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        #px.parse(os.path.join("%s/%s" % (original_path, cur_file)))
        cur_content = read_file(os.path.join("%s/%s" % (original_path, cur_file)))
        contents = cur_content
        #print contents
        titles = cur_content[0]
        print titles
        word_segmented = word_tokenize(" ".join(contents))
        write_file(contents, os.path.join("%s/%s.txt" % (out_path, dir_name)), False)
        write_file(word_segmented, out_path + "/tokenized_paper.temp", False)
        write_file(word_segmented, out_path + "/lemmatized_paper.temp", False)
        write_file(word_tokenize(" ".join(titles)), out_path + "/tokenized_title.temp", False)
        write_file(word_tokenize(" ".join(titles)), out_path + "/lemmatized_title.temp", False)
        write_file(word_tokenize(" ".join(cur_content)), out_path + "/tokenized_body.temp", False)
        write_file(word_tokenize(" ".join(cur_content)), out_path + "/lemmatized_body.temp", False)


def ini_mss2015_data(root_path, out_path):
    #__processing_using_ros()
    #__processing_using_nltk()
    os.makedirs(out_path)
    #__stop_word = read_file("./third_part/dict/stop_list.txt")
    for cur_file in os.listdir(root_path):
        out_dir_name = cur_file
        out_dir_path = os.path.join("%s/%s" % (out_path, out_dir_name))
        os.mkdir(out_dir_path)
        content = read_file(root_path + "/" + cur_file + "/" + cur_file + ".txt")
        write_file(content, out_dir_path + "/" + out_dir_name + ".txt", False)
        # start generate temp file
        tokenized_paper = read_file(root_path + "/" + cur_file + "/lemmatized_body.temp")
        remove_stop = []
        segmented_paper = []
        no_bracket_str = []
        section_set = []
        tmp_str = ""
        tmp_removed_str = ""
        tmp_no_bracket_str = ""
        __brackets = False
        tmp_int = 0
        for word in tokenized_paper:
            if word == "(" or word == u"（":
                __brackets = True
            elif word == ")" or word == u"）":
                __brackets = False
            #if word not in __stop_word:
            tmp_removed_str += word + " "
            if __brackets:
                tmp_str += word + " "
                continue
            if word != "#":
                tmp_no_bracket_str += word + " "
                tmp_str += word + " "
            if word.endswith(".") or word in [u"。", u"！", u"？", u"；", u"#"]:
                if tmp_removed_str != "":
                    segmented_paper.append(tmp_str)
                    remove_stop.append(tmp_removed_str)
                    no_bracket_str.append(tmp_no_bracket_str)
                    tmp_int += 1
                if word == "#":
                    section_set.append(str(tmp_int - 1))
                tmp_str = ""
                tmp_removed_str = ""
                tmp_no_bracket_str = ""
        section_set.append(str(len(segmented_paper)))
        write_file(remove_stop, out_dir_path + "/RemoveStop.temp", False)
        write_file(segmented_paper, out_dir_path + "/word_segment.temp", False)
        write_file(no_bracket_str, out_dir_path + "/word_remove_bracket.temp", False)
        titles = read_file(root_path + "/" + cur_file + "/tokenized_title.temp")
        write_file([" ".join(titles)], out_dir_path + "/titles.temp", False)
        write_file(section_set, out_dir_path + "/sec_idx.temp", False)
        model_temp(segmented_paper, out_dir_path)

    return ""









if __name__ == "__main__":
    __processing_using_nltk()
    #__processing_using_ros()
    # ini_kernel_rouge()
    ini_mss2015_data("../data/ros_result","../data/ros_result_with_stop")
    # __processing_using_nltk()
