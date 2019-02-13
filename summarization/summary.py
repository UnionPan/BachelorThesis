from utils.log_custom import log
from utils.file_operation import write_file
from utils.file_operation import read_file
from utils.data_initial import ini_rouge_data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from summarization.hlda_level import LevelScore
from pre_processing.merging_document import FileMerger
from summarization import hlda_analysis
from path_sum import PathSum
import dpp_sample.sample_dpp as ds
import numpy as np
import os
import sys
import ConfigParser
DATA = "mss2017"


class MultiDocumentSummary(object):
    def __init__(self):
        """
        document summarizing class
        Use Determinantal Point Processes
        """
        print 'document summarization'
        # Used for reading configure
        self.__cf = None
        # must be initialed kernel variable
        self.__key_word = None
        self.__paper = None
        self.__doc_matrix_ = None
        self.__paper_original = None
        self.__sub_paper_len = None
        self.__quality = None
        self.__level_tmp = None
        # title of all papers
        self.__titles = None
        # name of current paper for extracting summary
        self.__paper_name = ""
        # root rouge dictionary
        self.__rouge_path = None
        # self.__target_len_dir = "./data/MultiLing2017-MSS-ROS/2017_test_data/target-length/"
        self.__target_len_dir = "./data/data_for_rouge/target-length/"
        # used only for hlda modeling and calculate level feature
        # stored abstract path of current file
        self.__child_path = ""
        # used for storing rouge configuration
        self.__all_conf = []
        # biggest summary length
        self.max_sum_len__ = 250
        self.__target_len = None
        # used for file merger
        # should be removed
        self.__all_file = FileMerger()
        # abstract methods variable
        self.quality_method__ = ""
        self.distance_method__ = ""
        self.__feature_method = ""
        self.summary_method = ""
        self.stop_word_method = ""
        self.candidate_method = ""
        # feature merge switch: position, length, similarity, coverage, level
        self.feature_merge = "0-0-0-0-1"

    def set_merger(self, merger):
        self.feature_merge = merger

    def set_methods(self, method__, sum_method__="", stop_word="remove_stop", candidate_="DR"):
        """
        :param method__: QD, use quality and distance; DM, direct matrix
        :param sum_method__:  QC, DM,
                            hDPP use hierarchy DPP sampling sentence, until 250  words.
                            OneInDoc, select one sentence in each doc each time.
        :param stop_word: remove_stop, with_stop
        :param candidate_: candidate set construct method
        :return: null
        """
        if self.__cf is None:
            self.__cf = ConfigParser.ConfigParser()
            self.__cf.read("db_config.ini")
        self.stop_word_method = stop_word
        self.__feature_method = method__
        self.candidate_method = candidate_
        self.quality_method__ = ""
        if candidate_ == "CLU-DPP":
            self.summary_method = "CLU-DPP"
        elif candidate_ == "RANDOM":
            self.summary_method = "QD"
        elif sum_method__ == "":
            self.summary_method = method__
        else:
            self.summary_method = sum_method__

    @staticmethod
    def __feature_normalization(tmp_array):
        log.info("feature normalization: sigmoid")
        # sigmoid
        if isinstance(tmp_array, list):
            return (np.array(tmp_array) / np.max(tmp_array)).tolist()
            # return [np.exp(-1.0 * x) for x in tmp_array]
        else:
            return tmp_array / np.max(tmp_array)
            # return np.exp(-1.0 * tmp_array)

    def __quality_initial_position(self):
        if self.feature_merge.split("-")[0] == "0":
            return
        log.info("quality calculation: position")
        if self.quality_method__ == "":
            self.quality_method__ += "pos"
        else:
            self.quality_method__ += "-pos"

        tmp_quality = np.zeros([len(self.__paper)])
        for i in range(len(self.__paper)):
            tmp_quality[i] = 1.0 - float(i) / float(len(self.__paper))
        # tmp_quality = self.__feature_normalization(tmp_quality)
        self.__quality += tmp_quality * float(self.feature_merge.split("-")[0])

    def __quality_initial_length(self):
        if self.feature_merge.split("-")[1] == "0":
            return
        log.info("quality calculation: length")
        if self.quality_method__ == "":
            self.quality_method__ += "len"
        else:
            self.quality_method__ += "-len"
        tmp_quality = np.zeros([len(self.__paper)])
        for i in range(len(self.__paper)):
            tmp_quality[i] = len(self.__paper[i].replace(" ", ""))
        mean = tmp_quality.sum() / float(len(self.__paper))
        var = np.cov(tmp_quality)
        for i in range(len(self.__paper)):
            tmp_quality[i] = np.exp(
                (-1 * np.square(tmp_quality[i] - mean)) / var)
        # tmp_quality = self.__feature_normalization(tmp_quality)
        self.__quality += tmp_quality * float(self.feature_merge.split("-")[1])

    def __quality_initial_similarity(self):
        if self.feature_merge.split("-")[2] == "0":
            return
        log.info("quality calculation: similarity")
        if self.quality_method__ == "":
            self.quality_method__ += "sim"
        else:
            self.quality_method__ += "-sim"
        tmp_quality = np.zeros([len(self.__paper)])
        '''
        calculate quality use similarity
        '''
        title = " ".join(self.__titles)
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        corpus = [title]
        corpus.extend(self.__paper)
        tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
        # word = vectorizer.get_feature_names() # all words
        weight = tfidf.toarray()
        for i in range(len(self.__paper)):
            tmp_quality[i] = np.array(weight[i+1]).dot(np.array(weight[0]))
        tmp_quality = self.__feature_normalization(tmp_quality)
        self.__quality += tmp_quality * float(self.feature_merge.split("-")[2])

    def __quality_initial_coverage(self):
        """
        initial quality list use sentence coverage feature
        :return: null
        """
        if self.feature_merge.split("-")[3] == "0":
            return
        log.info("quality calculation: sentence coverage")
        if self.quality_method__ == "":
            self.quality_method__ += "cov"
        else:
            self.quality_method__ += "-cov"
        tmp_quality = np.zeros([len(self.__paper)])
        sen_num = len(self.__paper)
        union_paper = " ".join([" ".join(set(sen.split(" "))) for sen in self.__paper]).split(" ")
        for i in range(sen_num):
            word_list = self.__paper[i].split(" ")
            word_in_sen = [union_paper.count(cur_word) / float(sen_num) for cur_word in word_list]
            tmp_quality[i] = np.sum(word_in_sen) / len(word_list)
        tmp_quality = self.__feature_normalization(tmp_quality)
        self.__quality += tmp_quality * float(self.feature_merge.split("-")[3])

    def __quality_initial_level(self):
        if self.feature_merge.split("-")[4] == "0":
            return
        log.info("quality calculation: hLDA level")
        log.info("get level score: " + self.__child_path)
        if self.quality_method__ == "":
            self.quality_method__ += "lev"
        else:
            self.quality_method__ += "-lev"
        if self.__level_tmp is None:
            self.__level_tmp = LevelScore()
        tmp_level = self.__level_tmp.get_paper_level_score(self.__child_path)
        tmp_level = self.__feature_normalization(tmp_level)
        print len(tmp_level)
        print len(self.__quality)
        for i in range(len(self.__quality) - len(tmp_level)):
            tmp_level.append(0.0)
        if len(self.__quality) - len(tmp_level) < 0:
            tmp_level = tmp_level[:len(self.__quality)]
        self.__quality += (np.array(tmp_level) * float(self.feature_merge.split("-")[4]))

    def __quality_calculating(self, idx):
        """
        calculate quality using different methods
        :param idx: index of matrix element
        :return:
        """
        if self.__paper is None:
            log.error("")
            sys.exit()
        if self.__quality is not None:
            return self.__quality[idx]
        self.__quality = np.zeros([len(self.__paper)])
        self.__quality_initial_length()
        self.__quality_initial_coverage()
        self.__quality_initial_position()
        self.__quality_initial_level()
        self.__quality_initial_similarity()
#        self.__quality = self.__feature_normalization(self.__quality)
        # self.__quality /= 2.0
        return self.__quality[idx]

    def __similarity_calculating(self, idx_i, idx_j):
        if idx_i + idx_j <= 0:
            log.info("distance calculation: JACCARD")
            self.distance_method__ = "jaccard"
        inter_ = set(self.__paper[idx_i].split(" ")
                     ).intersection(self.__paper[idx_j].split(" "))
        union_ = set(self.__paper[idx_i].split(" ")
                     ).union(self.__paper[idx_j].split(" "))
        # print 1 - (float(len(inter_))) / float(len(union_))
        return (float(len(inter_))) / float(len(union_))

    def __cal_matrix_element(self, idx_1, idx_2):
        return self.__quality_calculating(idx_1) * \
               self.__similarity_calculating(idx_1, idx_2) * \
               self.__quality_calculating(idx_2)

    def __get_doc2vec_matrix(self, path):
        log.info('use word2vec')
        self.quality_method__ = "word2vec"
        self.distance_method__ = "100"
        word2vec_matrix = read_file(path)
        word2vec_matrix = word2vec_matrix[2:len(word2vec_matrix)-1]
        self.__key_word = [vec_.split(u" ")[0] for vec_ in word2vec_matrix]
        log.debug("word2vec key words: \n" + "\t".join(self.__key_word))
        word2vec = np.array([(vec_.encode("utf-8")).split(" ")[1:] for vec_ in word2vec_matrix])
        word2vec = word2vec.astype(np.float64)
        print word2vec.shape
        print len(self.__key_word)
        return word2vec.dot(word2vec.transpose()) * 1000

    def __cal_matrix(self, file_name=""):
        log.info("extract feature from pre-defined setting!")
        if self.__feature_method == "QD":
            paper_len = len(self.__paper)
            matrix_l = np.zeros([paper_len, paper_len])
            for i in range(paper_len):
                for j in range(paper_len):
                    # print "element %d, %d" % (i, j)
                    num = self.__cal_matrix_element(i, j)
                    matrix_l[i][j] = num
                    matrix_l[j][i] = num
        elif self.__feature_method == "DM":
#            if file_name == "":
#                log.error("file name is empty, please check!")
#                return []
#            file_path = os.path.join("./data/word2vec/remove_stop/%s.vec" % file_name)
            file_path = self.__child_path + "word_segment.vec"
            matrix_l = self.__get_doc2vec_matrix(file_path)
        else:
            log.error("self.__feature_method is " + self.__feature_method)
            return []
#        matrix_l = self.__feature_normalization(matrix_l)
        if self.summary_method == "hDPP":
            self.__doc_matrix_ = matrix_l
        return matrix_l

    def __cal_candidate_set(self):
        matrix_l = self.__cal_matrix()
        subset_ = []
        eigenvalue = []
        try:
            if self.candidate_method == "DR":
                subset_, eigenvalue = ds.sample(matrix_l)
            elif self.candidate_method == "CLU-DPP":
                cluster = hlda_analysis.sentence_cluster(self.__child_path, "run000")
                # debug hLDA message, include: total cluster number, each cluster sentence,
                i = 0
                tmp = ""
                log.info("cluster number: " + str(len(cluster)))
                for sen_list in cluster:
                    tmp += "\n cluster: " + str(i) + "\tsentence_num is " + str(len(sen_list)) + "\n"
                    tmp += "\n".join(np.array(self.__paper_original)[sen_list])
                    i += 1
                log.debug(tmp)
                # begin calculate and get sentence
                for i in range(len(cluster) / 2):
                    sen_list = cluster[i]
                    tmp_matrix = matrix_l[sen_list][:, sen_list]
                    tmp_set, eig = ds.sample(tmp_matrix)
                    if len(sen_list) < 10:
                        subset_.append(sen_list)
                        eigenvalue.append(eig)
                        continue
                    subset_.append(np.array(sen_list)[tmp_set].tolist())
                    eigenvalue.append(np.array(eig)[tmp_set].tolist())
            elif self.candidate_method == "RANDOM":
                for i in range(20):
                    subset_.append(np.random.randint(0, len(self.__paper_original)))
            else:
                raise RuntimeError("value error: " + self.candidate_method)
        except RuntimeError as e:
            log.error(e)
        finally:
            return subset_, eigenvalue

    def __get_word_frequency(self, key_word):
        frequency = []
        orig_doc_ = " ".join(self.__paper).split(" ")
        union_word = set(orig_doc_)
        for word in key_word:
            frequency.append(word + '\t' + str(orig_doc_.count(word)))
        frequency.append("not in word")
        for word in union_word:
            if word in key_word:
                continue
            frequency.append(word + '\t' + str(orig_doc_.count(word)))
        log.debug("word frequency: \n" + "\n".join(frequency))
        return frequency

    def __construct_summary(self, sentence_subset, eig, lang="zh"):
        summary = []
        sum_length = 0
        if self.summary_method == "QD":
            for sentence_idx in sentence_subset:
                if sum_length < self.max_sum_len__:
                    tmp_sen = self.__paper_original[sentence_idx]
                    # if lang in ['ja', 'th', 'zh']:
                    #     summary.append(" ".join([i for i in tmp_sen]))
                    # else:
                    #     summary.append(tmp_sen)
                    summary.append(tmp_sen)
                else:
                    break
                sum_length += len(self.__paper_original[sentence_idx])
            quality = np.array(self.__quality)
            quality[sentence_subset] = -999
            while sum_length < self.max_sum_len__:
                max_quality = np.where(quality == np.max(quality))
                tmp_summary = np.array(self.__paper_original)[max_quality]
                tmp_sen = "\n".join(tmp_summary.tolist())
                summary.append(tmp_sen)
                sum_length += len(tmp_sen)
                quality[max_quality] = -999
            summary = [" ".join(("\n".join(summary)[:self.max_sum_len__]).split("\n"))]
        elif self.summary_method == "DM":
            print len(self.__key_word)
            key_word = set(np.array(self.__key_word)[sentence_subset])
            self.__get_word_frequency(key_word)
            common_number = np.zeros([len(self.__paper_original)])
            for i in range(len(self.__paper_original)):
                common_number[i] = len(key_word.intersection(set(self.__paper_original[i].split(" "))))
            while sum_length < self.max_sum_len__:
                b = np.where(common_number == np.max(common_number))
                common_number[b] = 0
                for sen in np.array(self.__paper_original)[b].tolist():
                    summary.append(sen)
                    sum_length = len(summary)
                    if sum_length > self.max_sum_len__:
                        break
            summary = [" ".join(("\n".join(summary)[:self.max_sum_len__]).split("\n"))]
        elif self.summary_method == "newDM":
            key_word = set(np.array(self.__key_word)[sentence_subset])
            self.__get_word_frequency(key_word)
            common_number = np.zeros([len(self.__paper_original)])
            selected_sen = []
            while sum_length < self.max_sum_len__ and len(key_word) > 0:
                for i in range(len(self.__paper_original)):
                    if i in selected_sen:
                        continue
                    common_number[i] = len(key_word.intersection(set(self.__paper_original[i].split(" "))))
                b = np.where(common_number == np.max(common_number))
                common_number[b] = 0
                key_word -= key_word.intersection(set(self.__paper_original[b[0][0]].split(" ")))
                selected_sen += b[0].tolist()
                for sen in np.array(self.__paper_original)[b].tolist():
                    summary.append(sen)
                    sum_length = len(summary)
                    if sum_length > self.max_sum_len__:
                        break
            summary = [" ".join(("\n".join(summary)[:self.max_sum_len__]).split("\n"))]
        elif self.summary_method == "hDPP":
            tmp_summary = np.array(self.__paper_original)[sentence_subset]
            matrix_l = self.__doc_matrix_
            while len(("".join(tmp_summary)).replace(" ", "")) > self.max_sum_len__:
                matrix_l = matrix_l[sentence_subset][:, sentence_subset]
                sentence_subset, eigenvalue = ds.sample(matrix_l)
                tmp_summary = tmp_summary[sentence_subset]
            # tmp_sen = ("".join(tmp_summary)).replace(" ", "")
            # sum_length = len(tmp_sen)
            # summary.append(" ".join([i for i in tmp_sen]))
            tmp_sen = (" ".join(tmp_summary))
            sum_length = len(tmp_sen.replace(" ", ""))
            summary.append(tmp_sen)
        elif self.summary_method == "OneInDoc":
            sen_sub = []
            tmp_b = set(range(len(self.__paper_original))) - set(sentence_subset)
            sentence_subset += np.sort(list(tmp_b)).tolist()
            for i in range(len(self.__sub_paper_len) - 1):
                idx = np.where(np.array(sentence_subset) >= self.__sub_paper_len[i])
                tmp = np.array(sentence_subset)[idx]
                idx = np.where(tmp < self.__sub_paper_len[i+1])
                if tmp[idx].size == 0:
                    continue
                sen_sub.append(list(tmp[idx]))
            log.debug("splited sentence idx is: " + str(sen_sub))
            if_stop = False
            while sum_length < self.max_sum_len__ and not if_stop:
                if_stop = True
                for li in sen_sub:
                    if len(li) == 0:
                        continue
                    if_stop = False
                    tmp_sen = self.__paper_original[li[0]].replace(" ", "")
                    sum_length += len(tmp_sen)
                    if sum_length > self.max_sum_len__:
                        # summary.append(" ".join([i for i in tmp_sen]))
                        summary.append(self.__paper_original[li[0]])
                        break
                    else:
                        # summary.append(" ".join([i for i in tmp_sen]))
                        summary.append(self.__paper_original[li[0]])
                        li.remove(li[0])
            summary = [" ".join(("\n".join(summary)[:self.max_sum_len__]).split("\n"))]
        elif self.summary_method == "quality":
            quality = np.array(self.__quality)
            while sum_length < self.max_sum_len__:
                max_quality = np.where(quality == np.max(quality))
                tmp_summary = np.array(self.__paper_original)[max_quality]
                # tmp_sen = "".join(tmp_summary.tolist()).replace(" ", "")
                # summary.append(" ".join([i for i in tmp_sen]))
                tmp_sen = "\n".join(tmp_summary.tolist())
                summary.append(tmp_sen)
                sum_length += len(tmp_sen)
#                print max_quality
#                print quality
#                print summary
                quality[max_quality] = -999
#                print quality
            summary = ("\n".join(summary)[:self.max_sum_len__]).split("\n")
        elif self.summary_method == "CLU-DPP":
            while sum_length < self.max_sum_len__:
                tmp_len = 0
                for i in range(len(sentence_subset)):
                    # sen_num = int(float(len(subset)**2)/len(self.__paper_original))
                    subset = sentence_subset[i]
                    tmp_eig = eig[i]
                    tmp_len += len(subset)
                    if len(subset) == 0:
                        continue
                    idx = np.where(np.max(tmp_eig) == tmp_eig)[0][0]
                    tmp_sen = self.__paper_original[subset[idx]]
                    tmp_eig.remove(tmp_eig[idx])
                    subset.remove(subset[idx])
                    summary.append(tmp_sen)
                    sum_length += len(tmp_sen)
                    if sum_length > self.max_sum_len__:
                        break
                if tmp_len == 0:
                    break
            summary = [" ".join(("\n".join(summary)[:self.max_sum_len__]).split("\n"))]
        elif self.summary_method == "PathSumm":
            sentence_subset = range(len(self.__paper))
            path_sum = PathSum(self.__child_path, sentence_subset)
            while sum_length < self.max_sum_len__:
                sen_idx = path_sum.get_next_sentence()
                summary.append(self.__paper_original[sen_idx])
                sum_length += len(self.__paper_original[sen_idx])
            summary = [" ".join(("\n".join(summary)[:self.max_sum_len__]).split("\n"))]
        else:
            return ""

        log.debug("summary length is: " + str(sum_length))
        log.debug("generated summary: \n" + " ".join(summary))
        return summary

    def get_mss_paper_summary(self, lang, file_name, if_write_file=True):
        """
        generate summary for one paper, single document summarization
        :param lang:
        :param file_name: current file name, used for write summary answer
        :param if_write_file: whether write generated summary to answer file named file_name
        :return:
        """
        # initial
        self.__quality, self.__paper_name = None, file_name
        self.quality_method__ = ""
        '''
        if DATA == "mms2015":
            self.__all_file.merge_mms_2015(os.path.dirname(self.__child_path), "chinese")
        elif DATA == "mss2017":
            if lang in ["vi", "ka"]:
                self.__all_file.merge_mss_2017(os.path.dirname(self.__child_path))
            else:
                self.__all_file.merge_mss_2017_ros(os.path.dirname(self.__child_path))
        self.__paper_original = self.__all_file.get_merged_paper()
        if self.stop_word_method == "remove_stop":
            self.__paper = self.__all_file.get_filtered_paper()
        elif self.stop_word_method == "with_stop":
            self.__paper = self.__all_file.get_merged_paper()
        self.__titles = self.__all_file.get_titles()
        # used for generate hLDA input file and calculate level method.
        if (not os.path.exists(self.__child_path + "model.temp")) or False:
            write_file(self.__paper, self.__child_path + "RemoveStop.temp", False)
            write_file(self.__paper_original, self.__child_path + "word_segment.temp", False)
            model_temp(self.__paper, self.__child_path)
            return ""
        '''
        if self.stop_word_method == "remove_stop":
            self.__paper = read_file(self.__child_path + "RemoveStop.temp")
        elif self.stop_word_method == "with_stop":
            self.__paper = read_file(self.__child_path + "word_segment.temp")
        self.__titles = read_file(self.__child_path + "titles.temp")
        self.__paper_original = read_file(self.__child_path + "word_segment.temp")
        self.__sub_paper_len = [int(i) for i in read_file(self.__child_path + "sec_idx.temp")]
        # extract sentence
        feature_subset, eig = self.__cal_candidate_set()
#        feature_subset = range(len(self.__paper_original))
#        eig = []
        log.error("results is: ")
        log.info(feature_subset)
        log.debug(eig)
        # use feature list to extract summary
        summary = self.__construct_summary(feature_subset, eig, lang)
        if if_write_file:
            if file_name == '':
                log.error("file name is empty")
                return ""
            # write answer to file for ROUGE
            answer_path = self.__rouge_path + lang + "/systems/"
            write_file(summary, os.path.join('%s%s.txt' % (answer_path, file_name)), False)
            '''
            # generate gold summary split by CHAR
            gold_path = self.__rouge_path + lang + "/models/"
            if not os.path.exists(gold_path):
                os.makedirs(gold_path)
            tmp_name = lang + "/" + file_name + "_summary.txt"
            abs_human = read_file('./data/MultiLing2015-MSS/multilingMss2015Eval/summary/' + tmp_name)
            if not os.path.exists(gold_path + file_name + "_summary.txt") and lang != "vi" and lang != 'ka':
                write_file([" ".join(api.tokenize("\n".join(abs_human)))],
                           gold_path + file_name + "_summary.txt", False)
            if lang == "vi":
                write_file(abs_human, gold_path + file_name + "_summary.txt", False)
            # generate configure file of each document for ROUGE
            conf_path = self.__rouge_path + lang + "/configure/"
            if not os.path.exists(conf_path):
                os.makedirs(conf_path)
            tmp_conf_ = answer_path + file_name + ".txt " + gold_path + file_name + "_summary.txt"
            self.__all_conf.append(tmp_conf_)
            write_file([tmp_conf_], os.path.join('%s/%s.txt' % (conf_path, file_name)), False)
            '''

        return "".join(summary)

    def launch_multi_paper_summary(self, paper_path, name_pattern, if_write_file=True):
        """
        get multi-document summary, these document belongs to the same topic
        :param paper_path: path of paper path, it must be a dictionary
        :param name_pattern: common part of document name
        :param if_write_file: whether write summary to file, reserved interface
        :return: summary content with pre-determined length
        """
        self.__paper = None
        self.__quality = None
        return ''

    def launch_multiling_single_summary(self, dic_path):
        self.__rouge_path = ini_rouge_data(name_suffix=self.feature_merge + "-" + "-" + self.summary_method)
        path_dir = os.listdir(dic_path)
        for cur_lang in path_dir:
            if cur_lang not in ["zh", "en"]:
                continue
            lang_dir = os.path.join("%s/%s" % (dic_path, cur_lang))
            self.__all_conf = []
            # get target length of current language
            self.__target_len = dict()
            for line in read_file(self.__target_len_dir + cur_lang + ".txt"):
                self.__target_len[line.split("_")[0]] = int(line.split(",")[1])
            # get summary of current file(cur_file)
            for cur_file in os.listdir(lang_dir):
                self.max_sum_len__ = self.__target_len[cur_file]
                child_path = os.path.join('%s/%s/%s/' % (dic_path, cur_lang, cur_file))
                self.__child_path = child_path
                log.info(child_path)
                self.get_mss_paper_summary(cur_lang, cur_file)
            # write_file(self.__all_conf, self.__rouge_path + cur_lang + "/configure/.configure_all_" + cur_lang + ".txt", False)
            # if not os.path.exists(self.__rouge_path + cur_lang + "/output"):
            #     os.makedirs(self.__rouge_path + cur_lang + "/output")
        return self.__rouge_path
