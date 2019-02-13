# coding: UTF-8
from utils.data_initial import ini_mss2015_data
from summarization import summary_zoe as summary
# import rouge_win.run_rouge as rouge
from utils.log_custom import *
from utils.file_operation import read_file
from utils.file_operation import write_file
# from pre_processing import merging_document
# from summarization import hlda_analysis
# from utils.data_initial import ini_rouge_data
import os
import codecs
from utils.data_initial import __processing_using_nltk
from utils.data_initial import ini_mss2015_data


def __extract_rouge_result(result_path):
    log.info("extracting rouge result to log.")
    if not os.path.isdir(result_path + "output/"):
        log.error("result path is not a dictionary")
    for cur_file in os.listdir(result_path + "output/"):
        cur_rouge_value = read_file(os.path.join("%s/%s/%s" % (result_path, 'output', cur_file)))
        log.debug(cur_file + "\n" + "\n".join(cur_rouge_value))


def analyse_rouge_value(file_path, rouge_n):
    log.info("analysing rouge result ...")
    analysed_log = []
    for cur_log_file in os.listdir(file_path):
        if not cur_log_file.endswith(".log"):
            continue
        log_content = read_file(os.path.join("%s/%s" % (file_path, cur_log_file)))
        file_name = os.path.basename(cur_log_file)
        tmp_log = ""
        for i in range(len(log_content)):
            if log_content[i].endswith("configure_all.out"):
                tmp_log = "all\t"
                tmp_log += "\t".join(file_name.replace(".log", "").split(".")[3:])
                tmp_log += "\t" + log_content[i + 2].split(" ")[3]
                tmp_log += "\t" + log_content[i + 3].split(" ")[3]
                tmp_log += "\t" + log_content[i + 4].split(" ")[3]
                break
        analysed_log.append(tmp_log)
    write_file(analysed_log, "./data/log_analysis/ana.log", False)


def __launch_summary(iter__):
    # train data
    # original_path = "./data/MultiLing2015-MSS-ROS/eval_with_stop_token_with_title"
    # test data
    original_path = "../data/ros_result"
    conf = codecs.open('configure.txt', 'r', 'utf-8')
    text = conf.readlines()
    __processing_using_nltk(str(text[0]).strip('\r\n'), str(text[1]).strip('\r\n'))
    ini_mss2015_data(str(text[2]).strip('\r\n'), str(text[3]).strip('\r\n'))
    for i in range(iter__):
        log.info("iteration " + str(i + 1))
        if summary_object.stop_word_method == "remove_stop":
            data_path = str(text[3]).strip('\r\n')
        else:
            data_path = str(text[3]).strip('\r\n')
        if not os.path.exists(data_path):
            print "not exists"
            #ini_mss2015_data(original_path, data_path)
            exit()
        # start to get summary
        print data_path
        summary_object.launch_multiling_single_summary(data_path)
        #for lang in os.listdir(rouge_dir):
            #rouge.run_rouge(rouge_dir + lang + "/")
            #__extract_rouge_result(rouge_dir + lang + "/")
        # backup log file

if __name__ == '__main__':
    summary_object = summary.MultiDocumentSummary()
    stop_method = ["remove_stop", "with_stop"]
    # candidate =  ["DR", "CLU-DPP", "RANDOM"]
    f_method = ["QD"]  # f_method = ["QD", "DM"]
    s_method = [""]  # , "hDPP"] # s_method = ["", "hDPP", "OneInDoc", "quality"]
    for f in f_method:
        for s in s_method:
            log.info(f + "\t" + s)
            for i in range(1):
                # tmp = ""
                # tmp += str(((i + 1) & 0x10) >> 4) + "-"
                # tmp += str(((i + 1) & 0x08) >> 3) + "-"
                # tmp += str(((i + 1) & 0x04) >> 2) + "-"
                # tmp += str(((i + 1) & 0x02) >> 1) + "-"
                # tmp += str((i + 1) & 0x01)
                # feature merge switch: position, length, similarity, coverage, level
                # tmp = "0-0-0-0-1"
                tmp = ["1-0-0-0-0", "0-1-0-0-0", "0-0-1-0-0", "0-0-0-1-0", "0-0-0-0-1"][i]  # , "0-0-0-0-0"]
                log.info("feature merge setting: " + tmp)
                summary_object.set_merger(tmp)
                summary_object.set_methods(f, s, "with_stop")#, candidate_="CLU-DPP")  # , candidate_="RANDOM")
                __launch_summary(1)
