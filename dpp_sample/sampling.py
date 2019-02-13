# -*- coding: utf-8 -*-
from utils.log_custom import log
import dpp_alex.sampling as ds
import numpy as np
import ml_method.sentimentNew as sn


class DppSampling(object):
    def __init__(self, data_size, feature_size, start_idx):
        self.__data_size = data_size
        self.__feature_size = feature_size
        self.__test_data = []
        self.__test_label = []
        self.__f_x_test = np.zeros([data_size, feature_size])
        self.__parameter_theta = None
        self.__initial(start_idx)
        self.__best_answer = []
        self.__similarity = None
        self.__best_answer_eigenvalue = []
        self.__best_float = 0.0
        self.__sentiment = sn.sentimentAnalysis('third_part/dict/', False)
        # parameter size is the same as quality vector

    def __initial(self, start_idx):
#        doc2vec_file = open("data/input_data/qualityForTrain.vec", "r")
        doc2vec_file = open("data/input_data/qualityForTrain.vec", "r")
        i = 0
        for vector in doc2vec_file.readlines():
            if i < start_idx:
                i += 1
                continue
            list_vector = vector.strip().split(" ")
            self.__f_x_test[i - start_idx][:] = np.array(list_vector[:])
            i += 1
            if i - start_idx >= self.__data_size:
                break
        doc2vec_file.close()

        file = open("./data/input_data/taskAAForDoc2VecTrainData.txt", "r")
#        file = open("./data/input_data_test/all_test_data.txt", "r")
        datas = [sentence.strip().decode("utf-8").split(" ") for sentence in file.readlines()]
        self.__test_data = [datas[i + start_idx] for i in range(self.__data_size)]
        log.info(len(self.__test_data))
#        log.info(self.__test_data)
        file.close()
        file = open("./data/input_data/taskAAForDoc2VecTrainLabel.txt", "r")
        datas = [sentence.strip() for sentence in file.readlines()]
        self.__test_label = [datas[i + start_idx] for i in range(self.__data_size)]
        file.close()

    def __calculate_quality(self, element_idx):
#        return np.exp(self.__f_x_test[element_idx])
#        return np.exp(self.__sentiment.getSentimentScore((" ".join(self.__test_data[element_idx])).encode('utf-8')))
        return np.exp(0.5 * self.__parameter_theta.dot(self.__f_x_test[element_idx]))

    def __similarity__method(self, idx_i, idx_j):
        inter_ = set(self.__test_data[idx_i]).intersection(self.__test_data[idx_j])
        union_ = set(self.__test_data[idx_i]).union(self.__test_data[idx_j])
        return float(len(inter_)) / float(len(union_))

    def __calculate_similarity(self, idx_i, idx_j):
        '''
        if self.__similarity is None:
            log.info("initiating similarity")
            self.__similarity = self.__f_x_test.dot(
                self.__f_x_test.transpose())
        return self.__similarity[idx_i][idx_j]
        '''
        if self.__similarity is None:
            log.info("initiating similarity")
            self.__similarity = np.zeros([self.__data_size, self.__data_size])
            for i in range(self.__data_size):
                for j in range(i, self.__data_size):
                    self.__similarity[i][j] = self.__similarity__method(i, j)
                    self.__similarity[j][i] = self.__similarity[i][j]
        return self.__similarity[idx_i][idx_j]

    def __calculate_matrix_element(self, element_index_i, element_index_j):
        return self.__calculate_quality(element_index_i) * \
                   self.__calculate_quality(element_index_j) * \
                   self.__calculate_similarity(element_index_i, element_index_j)

    def sampling(self, parameter, attitude):
        self.__parameter_theta = parameter
        matrix_l = np.zeros([self.__data_size, self.__data_size])
        for row_i in range(self.__data_size):
            for col_j in range(row_i, self.__data_size):
                num = self.__calculate_matrix_element(row_i, col_j)
                matrix_l[row_i][col_j] = num
                matrix_l[col_j][row_i] = num
        list_y, eigenvalue = ds.sample(matrix_l)
        '''
        (eigenvalue, feature_vector) = np.linalg.eig(matrix_l)

        j = list()
        for x in range(eigenvalue.size):
            random_a = random.randrange(1, 11)
            if (eigenvalue[x] / (eigenvalue[x] + 1)) * 10 > random_a:
                j.append(x)

        matrix_y = matrix_l[j][:, np.array(j)]
        det = np.linalg.det(matrix_y)
        list_y = j
        det = 0
        while np.abs(det) > 0.000001:
            log.debug("det: " + str(det))
            v_dem = np.sqrt(matrix_y.size)
            for i in range(int(v_dem)):
                prop = 0.0
                # column j of feature_vector is the feature of element j
                for j in range(int(v_dem)):
                    prop += feature_vector[j][i] ** 2
                prop /= det
                log.debug("prop: " + str(prop))
                random_a = random.randrange(1, 11)
                if prop * 10 > random_a:
                    list_y.append(i)
                    j.pop(i)
                    feature_vector = feature_vector[j][:, j]
                    log.info(feature_vector)
        '''
        ans = []
        new_f = 0.0
        for attitude in ['FAVOR', 'AGAINST', 'NONE']:
            ans = []
            for i in range(self.__data_size):
                if i in list_y:
                    ans.append(attitude)
                else:
                    ans.append("NONEs")
            new_f = self.get_f_score(ans, attitude)
            if self.__best_float <= new_f:
                self.__best_float = new_f
                self.__best_answer = ans
                self.__best_answer_eigenvalue = eigenvalue / np.sum(eigenvalue)
            log.debug("best_matrix function is: ")
    #            log.debug(matrix_l.tolist())
#            log.debug("best_float is: " + str(ans))
            log.debug("best_answer is: " + str(ans))

        ans = []
        print len(list_y)
#        list_y = list(set(range(self.__data_size)) - set(list_y))
        print len(list_y)
        list_y = range(self.__data_size)
        for i in list_y:
            ans.append(self.__test_label[i])
        tmp = []
        for i in range(len(list_y)):
            if ans[i] == 'FAVOR':
                tmp.append(i)
        tmp_ans = []
        for i in tmp:
            tmp_ans.append(np.array([self.__f_x_test[i, 0], self.__f_x_test[i, 1], eigenvalue[i]]).tolist())
        tmp = tmp_ans
#        tmp = list(eigenvalue[tmp])
#        tmp = self.__f_x_test[tmp, :]
        log.debug(tmp)
        tmp = []
        for i in range(len(list_y)):
            if ans[i] == 'AGAINST':
                tmp.append(i)
        tmp_ans = []
        for i in tmp:
            tmp_ans.append(np.array([self.__f_x_test[i, 0], self.__f_x_test[i, 1], eigenvalue[i]]).tolist())
        tmp = tmp_ans
#        tmp = list(eigenvalue[tmp])
#        tmp = self.__f_x_test[tmp, :]
        log.debug(tmp)
        tmp = []
        for i in range(len(list_y)):
            if ans[i] == 'NONE':
                tmp.append(i)
        tmp_ans = []
        for i in tmp:
            tmp_ans.append(np.array([self.__f_x_test[i, 0], self.__f_x_test[i, 1], eigenvalue[i]]).tolist())
#            tmp_ans.append(self.__calculate_quality(i))
        tmp = tmp_ans
#        tmp = list(eigenvalue[tmp])
#        tmp = self.__f_x_test[tmp, :]
        log.debug(tmp)

        log.debug(list(eigenvalue))
        log.debug(ans)
        return new_f, list_y, eigenvalue

    def sample_for_test(self, parameter):
        self.__parameter_theta = parameter["FAVOR"]
        size_ = self.__data_size
        matrix_l = np.zeros([size_, size_])
        for row_i in range(size_):
            for col_j in range(row_i, size_):
                num = self.__calculate_matrix_element(row_i, col_j)
                matrix_l[row_i][col_j] = num
                matrix_l[col_j][row_i] = num
        tmp_ans = dict()
        list_y, eigenvalue = ds.sample(matrix_l)
        tmp_ans["FAVOR"] = list_y

        self.__parameter_theta = parameter["AGAINST"]
        tmp = range(size_)
        diff = list(set(tmp)-set(list_y))
        self.__f_x_test = self.__f_x_test[diff][:]
        self.__similarity = self.__similarity[diff][:, diff]
        size_ = len(diff)
        log.debug(self.__f_x_test)
        for row_i in range(size_):
            for col_j in range(row_i, size_):
                num = self.__calculate_matrix_element(row_i, col_j)
                matrix_l[row_i][col_j] = num
                matrix_l[col_j][row_i] = num
        list_y, eigenvalue = ds.sample(matrix_l)

        tmp_ans["AGAINST"] = []
        tmp_ans["NONE"] = []
        for i in range(size_):
            if i in list_y:
                tmp_ans["AGAINST"].append(diff[i])
            else:
                tmp_ans["NONE"].append(diff[i])
        return tmp_ans

    def sample_k_test(self, parameter, sentiment_test):
        self.__parameter_theta = parameter
        matrix_l = np.zeros([self.__data_size, self.__data_size])
        for row_i in range(self.__data_size):
            for col_j in range(row_i, self.__data_size):
                num = self.__calculate_matrix_element(row_i, col_j)
                print num
                matrix_l[row_i][col_j] = num
                matrix_l[col_j][row_i] = num
        size_ = self.__data_size
        ans = range(size_)
        '''
        matrix_k = matrix_l
        diff = range(size_)
        for i in range(10):
            center_idx, vec = ds.sample_k(matrix_k, 2)
            tmp = range(size_)
            log.info("center_idx: " + str(diff[center_idx[0]]) + " " + str(diff[center_idx[1]]))
            if i > 2:
                ans.append(diff[center_idx[0]])
                ans.append(diff[center_idx[1]])
                log.info("".join(self.__test_data[diff[center_idx[0]]]))
                log.info(self.__test_label[diff[center_idx[0]]])
                log.info("".join(self.__test_data[diff[center_idx[1]]]))
                log.info(self.__test_label[diff[center_idx[1]]])
            diff = list(set(tmp)-set(center_idx))
            self.__f_x_test = self.__f_x_test[diff][:]
            matrix_k = matrix_k[diff][:, diff]
            size_ = len(diff)
        log.info(matrix_l.shape)
        '''
        center_idx, vec = ds.sample_k(matrix_l, 3)
        log.debug(vec)
        final_answer_ = np.array(ans)[center_idx]
        log.info(final_answer_)
        log.info("".join(self.__test_data[final_answer_[0]]))
        log.info(self.__test_label[final_answer_[0]])
        log.info("".join(self.__test_data[final_answer_[1]]))
        log.info(self.__test_label[final_answer_[1]])
        log.info("".join(self.__test_data[final_answer_[2]]))
        log.info(self.__test_label[final_answer_[2]])
        cluster_1 = []
        cluster_2 = []
        cluster_3 = []
        score_0 = sentiment_test[center_idx[0]]
        score_1 = sentiment_test[center_idx[1]]
        score_2 = sentiment_test[center_idx[2]]
        label_1 = ''
        label_0 = ''
        label_2 = ''
        if score_0 > score_1:
            if score_0 > score_2:
                label_0 = 'FAVOR'
                if score_1 > score_2:
                    label_1 = 'NONE'
                    label_2 = 'AGAINST'
                else:
                    label_1 = 'AGAINST'
                    label_2 = 'NONE'
            else:
                label_2 = 'FAVOR'
                label_0 = 'NONE'
                label_1 = 'AGAINST'
        else:
            if score_1 > score_2:
                label_1 = 'FAVOR'
                if score_0 > score_2:
                    label_0 = 'NONE'
                    label_2 = 'AGAINST'
                else:
                    label_0 = 'AGAINST'
                    label_2 = 'NONE'
            else:
                score_2 = 'FAVOR'
                score_0 = 'AGAINST'
                score_1 = 'NONE'

        cluster_1.append(center_idx[0])
        cluster_2.append(center_idx[1])
        cluster_3.append(center_idx[2])
        for i in range(self.__data_size):
#            sim_1 = matrix_l[i][i] * matrix_l[center_idx[0]][center_idx[0]] - matrix_l[i][center_idx[0]] ** 2
#            sim_2 = matrix_l[i][i] * matrix_l[center_idx[1]][center_idx[1]] - matrix_l[i][center_idx[1]] ** 2
#            sim_3 = matrix_l[i][i] * matrix_l[center_idx[2]][center_idx[2]] - matrix_l[i][center_idx[2]] ** 2
#            sim_1 = np.sum(np.square(vec[:][i] - vec[:][center_idx[0]]))
#            sim_2 = np.sum(np.square(vec[:][i] - vec[:][center_idx[1]]))
#            sim_3 = np.sum(np.square(vec[:][i] - vec[:][center_idx[2]]))
            sim_1 = self.__similarity[i][center_idx[0]]
            sim_2 = self.__similarity[i][center_idx[1]]
            sim_3 = self.__similarity[i][center_idx[2]]
            if sim_1 > sim_2:
                if sim_1 > sim_3:
                    cluster_1.append(i)
                else:
                    cluster_3.append(i)
            elif sim_2 > sim_3:
                cluster_2.append(i)
            else:
                cluster_3.append(i)
        ans = []
        for i in range(self.__data_size):
            print "i: " + str(i)
            if i in cluster_1:
#                ans.append(self.__test_label[center_idx[0]])
                ans.append(label_0)
            elif i in cluster_2:
                ans.append(label_1)
            else:
                ans.append(label_2)

        log.info('FINAL' + str(ans))
        for label in ["FAVOR", "AGAINST", "NONE"]:
            self.get_f_score(ans, label)

    def get_best_float(self):
        return self.__best_float

    def get_best_eigenvalue(self):
        return self.__best_answer_eigenvalue

    def get_best_answer(self):
        return self.__best_answer

    def get_f_score(self, experiment_label, attitude, i, j):
        count = 0
        answer_count = 0
        exp_count = 0
        for idx in range(j):
            idx += i
            if self.__test_label[idx] == attitude:
                answer_count += 1
                if experiment_label[idx] == attitude:
                    count += 1
            if experiment_label[idx] == attitude:
                exp_count += 1
        if count == 0 or exp_count == 0 or answer_count == 0:
            return 0.0
        precision = float(count) / exp_count
        recall = float(count) / answer_count
        log.debug("precision is: " + str(precision))
        log.debug("recall is: " + str(recall))
        f = precision * recall * 2 / (recall + precision)
        log.debug("f is: " + str(f))
        return f
