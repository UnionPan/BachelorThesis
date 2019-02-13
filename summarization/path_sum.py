# coding=utf-8
import numpy as np
from utils.file_operation import read_file


class PathSum(object):
    def __init__(self, file_path, candidate):
        self.paper = 1
        self.path_list = dict()
        self.sen_levels = []
        self.__ori_allocation = None
        self.__cur_allocation = None
        self.__candidata = candidate
        self.__root_path = file_path
        self.__init(file_path)

    def __init(self, root_path):
        """
        :param root_path, file path of hLDA results, must contains runXXX
        :return:
        """
        run_path = root_path + "/run000"
        # path assign
        mode_assign = read_file(run_path + "/mode.assign")
        # word level assign
        mode_levels = read_file(run_path + "/mode.levels")
        # word list
        word_list = read_file(root_path + "/words.temp")

        # sentences paths
        self.path_list = dict()
        for idx in range(len(mode_assign)):
            line = mode_assign[idx]
            new_path = " ".join(line.split(" ")[2:])
            if new_path not in self.path_list:
                self.path_list[new_path] = []
            if idx in self.__candidata:
                self.path_list[new_path].append(idx)
        self.__ori_allocation = np.array([float(len(self.path_list[i]))for i in self.path_list])
        self.__ori_allocation /= (np.sum(self.__ori_allocation))
        self.__cur_allocation = np.zeros(self.__ori_allocation.shape).tolist()
        # print path
        # for path in sorted(self.path_list.items(), key=lambda x: len(x[1])):
        #     print path[0], "\t: ", str(len(path[1])), path[1]
        # for path in self.path_list:
            # if self.path_list[path]
            # print path, len(self.path_list[path]), self.path_list[path]
        # sentence levels
        self.sen_levels = []
        for i in range(len(mode_levels)):
            self.sen_levels.append([])
            word_level = mode_levels[i].split(" ")
            for j in range(3):
                tmp = []
                for word in word_level:
                    w2l = word.split(":")
                    if w2l[1] == str(j):
                        tmp.append(w2l[0])
                self.sen_levels[i].append(tmp)

    def get_next_sentence(self):
        while True:
            minium = 999999999
            tmp = np.array(self.__cur_allocation)
            idx = -1
            for i in range(len(self.__ori_allocation)):
                tmp[i] += 1.0
                deviation = sum(np.abs(tmp / sum(tmp) - self.__ori_allocation))
                if deviation < minium:
                    minium = deviation
                    idx = i
                    self.__cur_allocation = tmp.tolist()
                tmp[i] -= 1.0
            if len(self.path_list.values()[idx]) != 0:
                results = self.path_list.values()[idx][0]
                self.path_list.values()[idx].remove(results)
                return results
            elif sum([len(tmp) for tmp in self.path_list.values()]) == 0:
                return -1

    def test(self):
        """
        :param root_path, file path of hLDA results, must contains runXXX
        :return:
        """
        run_path = self.__root_path + "/run000"
        # path assign
        mode_assign = read_file(run_path + "/mode.assign")
        # word level assign
        mode_levels = read_file(run_path + "/mode.levels")
        # word list
        word_list = read_file(self.__root_path + "/words.temp")
        # mode.temp
        model_temp = read_file(self.__root_path + "model.temp")

        # sentences paths
        self.path_list = dict()
        self.nodes = dict()
        self.node_word_freq = dict()
        for idx in range(len(mode_assign)):
            line = mode_assign[idx]
            new_path = " ".join(line.split(" ")[2:])
            if new_path not in self.path_list:
                self.path_list[new_path] = []
            if idx in self.__candidata:
                self.path_list[new_path].append(idx)
        self.__ori_allocation = np.array([float(len(self.path_list[i]))for i in self.path_list])
        self.__ori_allocation /= (np.sum(self.__ori_allocation))
        self.__cur_allocation = np.zeros(self.__ori_allocation.shape).tolist()
        # print path
        # for path in sorted(self.path_list.items(), key=lambda x: len(x[1])):
        #     print path[0], "\t: ", str(len(path[1])), path[1]

        for level in range(3):
            for path in mode_assign:
                cur_node = path.split(" ")[2:level + 3]
                if " ".join(cur_node) not in self.nodes:
                    self.nodes[" ".join(cur_node)] = []
                    self.node_word_freq[" ".join(cur_node)] = []
                for i in range(len(mode_levels[int(path.split(" ")[0])].split(" "))):
                    word = mode_levels[int(path.split(" ")[0])].split(" ")[i]
                    if int(word.split(":")[1]) == level:
                        self.nodes[" ".join(cur_node)].append(word_list[int(word.split(":")[0])])
                        # self.node_word_freq[" ".join(cur_node)].append(model_temp[int(path.split(" ")[0])].split(" ")[i + 1])
        for node in self.nodes:
            print node, ": ", "\t".join(self.nodes[node])# , "\t".join(self.node_word_freq[node])
        # for path in self.path_list:
        #     print path, ": ", len(self.path_list[path]), ", ", self.path_list[path]
        #     for sen_idx in self.path_list[path]:
        #         leves = mode_levels[sen_idx]


if __name__ == "__main__":
    print "path_sum"
    test = PathSum("../data/MultiLing2015-MSS-ROS/eval_with_stop_with_title/en/0fb33cd018ad2920a6c4fcfaba506f06/",
                   [8, 9, 13, 17, 19, 20, 25, 29, 32, 36, 44, 51, 55, 57, 70, 72, 76, 79, 85, 86, 91, 97, 98, 102, 111, 113, 115, 116, 117, 119, 121, 124, 129, 135, 140, 150, 151, 152, 154, 172, 173, 175, 176, 179, 183, 185])
    test.test()
