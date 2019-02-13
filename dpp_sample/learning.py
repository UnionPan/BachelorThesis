# -*- coding: utf-8 -*-

from utils.log_custom import log
import copy
from utils.constants import *
import numpy as np
import random
import sampling as sp


# Jaccard similarity
# keyword
class DPPLearning(object):
    def __init__(self, data_size, feature_size):
        log.info("dpp learning module")
        self.__data_size = data_size
        self.__feature_size = feature_size
        self.__train_label = []
        self.__train_data = []
#        self.__parameter_theta = np.zeros(feature_size)
        self.__parameter_theta = None
        self.__matrix_l = np.zeros([data_size, data_size])
        self.__similarity = None
        self.__f_x = np.zeros([data_size, self.__feature_size])
        self.__label = np.zeros(data_size)
        self.__step = 0.01
        self.__tmp_answer = dict()
        self.__final_parameter = dict()
        self.__tmp_eigenvalue = dict()
        self.__sample = None

    def __initiate_f_x(self):
        log.info("initial f(x)")
        doc2vec_file = open("data/input_data/qualityForTrain.vec", "r")
        i = 0
        for vector in doc2vec_file.readlines():
            list_vector = vector.strip().split(" ")
            self.__f_x[i][:] = np.array(list_vector[:])
            i += 1
            if i >= self.__data_size:
                break
        doc2vec_file.close()

        sentence_file = open("data/input_data/taskAAForDoc2VecTrainData.txt")
        self.__train_data = [sentence.strip().decode("utf-8").split(" ") for sentence in sentence_file.readlines()]
        sentence_file.close()
        train_label_file = open("data/input_data/taskAAForDoc2VecTrainLabel.txt")
        self.__train_label = [sentence.strip() for sentence in train_label_file.readlines()]
        train_label_file.close()

    def __calculate_quality(self, element_idx):
        return np.exp(-0.5 * self.__parameter_theta.dot(self.__f_x[element_idx]))

    # one-zero similarity
    def __similarity__method(self, idx_i, idx_j):
        inter_ = set(self.__train_data[idx_i]).intersection(self.__train_data[idx_j])
        union_ = set(self.__train_data[idx_i]).union(self.__train_data[idx_j])
        return float(len(inter_)) / float(len(union_))
        count = 0.0
        for word_i in self.__train_data[idx_i]:
            if word_i in self.__train_data[idx_j]:
                count += 1.0
                break
        return float(count) / (np.sqrt(len(self.__train_data[idx_i])) * np.sqrt(len(self.__train_data[idx_j])))

    def __calculate_similarity(self, idx_i, idx_j):
        if self.__similarity is None:
            log.info("initiating similarity")
            self.__similarity = self.__f_x.dot(self.__f_x.transpose().reshape(self.__data_size, self.__feature_size))
        return self.__similarity[idx_i][idx_j]
        '''
        if self.__similarity is None:
            log.info("initiating similarity")
            self.__similarity = np.zeros([self.__data_size, self.__data_size])
            for i in range(self.__data_size):
                for j in range(self.__data_size):
                    self.__similarity[i][j] = self.__similarity__method(i, j)
#                log.debug(self.__similarity[i][:].tolist())
        return self.__similarity[idx_i][idx_j]
        '''
    def __calculate_matrix_element(self, element_index_i, element_index_j):
        return self.__calculate_quality(element_index_i) * \
            self.__calculate_quality(element_index_j) * \
            self.__calculate_similarity(element_index_i, element_index_j)

    def calculate_gradient(self, attitude):
        # compute L(x; theta) as in equation (155)
        log.debug(self.__parameter_theta)
        log.info("computing matrix L")
        for row_i in range(self.__data_size):
            for col_j in range(self.__data_size):
                self.__matrix_l[row_i][col_j] = self.__calculate_matrix_element(row_i, col_j)

        # Eigendecompose L(x; theta)
        log.info("engendecomposing")
        log.debug("matrix value: " + str(np.linalg.det(self.__matrix_l)))
        (eigenvalue, feature_vector) = np.linalg.eig(self.__matrix_l)
        for i in range(len(eigenvalue)):
            eigenvalue[i] = float(eigenvalue[i])
            if np.abs(eigenvalue[i]) < 0.000000001:
                eigenvalue[i] = 0.0
        log.debug("eigenvalue")
        log.debug(eigenvalue)

        # calculate K_ii
        log.info("calculating Kii")
        vector_k = np.zeros(self.__data_size)
        log.debug("feature value matrix")
        for i in range(self.__data_size):
            for j in range(self.__data_size):
                vector_k[i] += ((eigenvalue[j] / (eigenvalue[j] + 1)) * (feature_vector[i][j] ** 2))

#        log.debug("Kii: " + str(vector_k))
        # calculate gradient
        log.info("calculating gradient")
        sigma_sub_f_x = np.zeros(self.__feature_size)
        for index in range(self.__data_size):
            if self.__train_label[index] == attitude:
                sigma_sub_f_x += self.__f_x[index]

        sigma_kii_f_x = np.zeros(self.__feature_size)
        for i in range(self.__data_size):
            sigma_kii_f_x += vector_k[i] * self.__f_x[i]

        return sigma_sub_f_x - sigma_kii_f_x

    @staticmethod
    def __whether_end(gradient):
        for grad_double in gradient:
            if np.abs(grad_double) > 0.001:
                return False
        return True

    def learning(self):
        for label in ["FAVOR", "AGAINST", "NONE"]:
            self.__learning_single_attitude(label)
        right = 0.0
        ans = []
        log.info("answer")
        print self.__tmp_answer
        log.info(self.__tmp_answer["FAVOR"])
        log.info(self.__tmp_eigenvalue["FAVOR"])
        log.info(self.__tmp_answer["AGAINST"])
        log.info(self.__tmp_eigenvalue["AGAINST"])
        log.info(self.__tmp_answer["NONE"])
        log.info(self.__tmp_eigenvalue["NONE"])
        tmp = []
        for i in range(100):
            a = list()
            a.append(self.__tmp_answer["FAVOR"][i])
            a.append(self.__tmp_answer["AGAINST"][i])
            a.append(self.__tmp_answer["NONE"][i])
            count = 0
            label = ""
            for j in range(3):
                if a[j] == "NONEs":
                    count += 1
                else:
                    label = a[j]
            tmp.append(str(self.__tmp_eigenvalue["FAVOR"][i]) + '\t' + str(self.__tmp_eigenvalue["AGAINST"][i]) +
                       '\t' + str(self.__tmp_eigenvalue["NONE"][i]))
            if count == 2:
                right += 1
                ans.append(label)
            else:
                favor_value = self.__tmp_eigenvalue["FAVOR"][i]
                against_value = self.__tmp_eigenvalue["AGAINST"][i]
                none_value = self.__tmp_eigenvalue["NONE"][i]
                if favor_value > against_value:
                    if favor_value > none_value:
                        ans.append("FAVOR")
                    else:
                        ans.append("NONE")
                else:
                    if against_value > none_value:
                        ans.append("AGAINST")
                    else:
                        ans.append("NONE")

        log.info(tmp)
        log.info("final_answer: " + str(ans))
        log.info("final parameters: ")
        log.info(self.__final_parameter)
        for label in ["FAVOR", "AGAINST", "NONE"]:
            self.__sample.get_f_score(ans, label)

    def __learning_single_attitude(self, attitude):
        """
        learning parameters __parameter_theta, which is the parameters of quality model
        :param input_x: input data of X
        :param label_y: label y
        :return: L-matrix
        """
        log.info('learning: ' + attitude)
        self.__parameter_theta = np.random.random(size=self.__feature_size)
        self.__sample = sp.DppSampling(100, self.__feature_size)
        self.__initiate_f_x()
        grad = self.calculate_gradient(attitude)
        best_f = 0.0
        iter_count = 0
        log.debug("grad")
        log.info(grad)
        while (not self.__whether_end(grad)) and iter_count < 1000:
            log.info("interation: " + str(iter_count))
            new_f, ignore, ignore_value = self.__sample.sampling(self.__parameter_theta, attitude)
            if new_f > best_f:
                best_f = new_f
                self.__final_parameter[attitude] = copy.deepcopy(self.__parameter_theta)
            log.debug("grad")
            log.info(grad)
            log.debug("parameter")
            log.debug(self.__parameter_theta)
            self.__parameter_theta = self.__parameter_theta - self.__step * grad
            grad = self.calculate_gradient(attitude)
            iter_count += 1

        self.__tmp_eigenvalue[attitude] = copy.deepcopy(self.__sample.get_best_eigenvalue())
        self.__tmp_answer[attitude] = copy.deepcopy(self.__sample.get_best_answer())

