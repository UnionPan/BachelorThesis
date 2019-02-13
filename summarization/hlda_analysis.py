from utils.file_operation import read_file
import numpy as np
import os


def sentence_cluster(mode_path, run_name):
    assign_path = mode_path + "/" + run_name + "/mode.assign"
    mode_assign = read_file(assign_path)
    cluster_set = [" ".join(cur_line.split(" ")[2:]) for cur_line in mode_assign]
    ans = []
    for cluster_id in set(cluster_set):
        idx = np.where(np.array(cluster_set) == cluster_id)[0]
        sen_cluster = [int(line.split(" ")[0]) for line in np.array(mode_assign)[idx]]
        ans.append(sen_cluster)
    ans.sort(key=len, reverse=True)
    print ans
    return ans


if __name__ == "__main__":
    sentence_cluster("../data/MultiLing2015-MSS-ROS/eval_with_stop_with_title/en/0f8047e125d506e389b7f2d2f2d7f289",
                     "run000")
