from utils.file_operation import read_file
from utils.file_operation import write_file
import numpy as np


def get_memog_ans():
    memog_res = read_file("../data/2017_results/MSS2017_MeMoG_CI_March30.csv")
    res_value = np.array([line.replace(" ", "").split(",")[4] for line in memog_res])
    team = np.array([line.replace(" ", "").split(",")[0] for line in memog_res])
    priority = np.array([line.replace(" ", "").split(",")[1] for line in memog_res])
    lang = np.array([line.replace(" ", "").split(",")[2] for line in memog_res])
    ans = []
    memog_answer_value = []
    tmp_set = set()
    for cur_lang in set(lang):
        print cur_lang
        idx = np.where(lang == cur_lang)[0]
        cur_value = res_value[idx]
        cur_priority = priority[idx]
        cur_team = team[idx]
        tmp_ans = cur_lang + '\t'
        tmp_len = 0
        while tmp_len < len(cur_value):
            idx_max = np.where(cur_value == max(cur_value))[0]
            tmp_len += len(idx_max)
            for idx_1 in idx_max:
                tmp_ans += cur_team[idx_1] + "-" + cur_priority[idx_1] + '\t'
            cur_value[idx_max] = -1
        ans.append(tmp_ans)
    tmp_set = set([team[i] + "-" + priority[i] for i in range(len(team))])
    write_file(ans, "../memog_ans_march30.txt")
    final_ans = []
    tmp_set = sorted(list(tmp_set))
    final_ans.append("lang,"+",".join(tmp_set))
    for cur_ans in ans:
        ans_list = cur_ans.split('\t')
        print ans_list
        final_tmp_ans = ["" for i in range(len(tmp_set) + 1)]
        final_tmp_ans[0] = ans_list[0]
        for i in range(len(tmp_set)):
            if tmp_set[i] in ans_list:
                final_tmp_ans[i+1] = str(np.where(np.array(ans_list) == tmp_set[i])[0][0])
        final_ans.append(",".join(final_tmp_ans))
    write_file(final_ans, "../final_memog_ans_march30.csv")
    print "end"


def get_memog_value():
    memog_res = read_file("../data/2017_results/MSS2017_MeMoG_CI_March30.csv")
    res_value = np.array([line.replace(" ", "").split(",")[4] for line in memog_res])
    team_priority = np.array([line.replace(" ", "").split(",")[0] + "-" + line.replace(" ", "").split(",")[1] for line in memog_res])
    lang = np.array([line.replace(" ", "").split(",")[2] for line in memog_res])
    ans = []

    # get team priority
    tmp_team_priority = []
    for cur_lang in set(lang):
        idx = np.where(lang == cur_lang)[0]
        if len(tmp_team_priority) < len(team_priority[idx]):
            tmp_team_priority = team_priority[idx]
    tmp_team_priority = sorted(tmp_team_priority)
    print tmp_team_priority
    print len(tmp_team_priority)

    ans.append("lang\t" + "\t".join(tmp_team_priority))
    for cur_lang in set(lang):
        print cur_lang
        tmp_ans = cur_lang
        idx = np.where(lang == cur_lang)[0]
        cur_lang_value = res_value[idx]
        print team_priority[idx]
        for cur_team_priority in tmp_team_priority:
            idx_1 = np.where(team_priority[idx] == cur_team_priority)[0]
            if len(idx_1) == 0:
                tmp_ans += "\t-"
                continue
            tmp_ans += "\t" + str(float(cur_lang_value[idx_1[0]]) * 100)
        ans.append(tmp_ans)
    write_file(ans, "../final_memog_value_march30.csv")
    print "end"


def get_rouge_ans():
    memog_res = read_file("../data/2017_results/MSS2017_ROUGE_1.5.7_CI.csv")
    ori_res_value = np.array([line.replace(" ", "").split(",")[4] for line in memog_res])
    ori_team = np.array([line.replace(" ", "").split(",")[0] for line in memog_res])
    ori_priority = np.array([line.replace(" ", "").split(",")[1] for line in memog_res])
    ori_lang = np.array([line.replace(" ", "").split(",")[2] for line in memog_res])
    rouge_n = np.array([line.replace(" ", "").split(",")[3] for line in memog_res])
    for cur_rouge in ["ROUGE-1", "ROUGE-2", "ROUGE-3", "ROUGE-4"]:
        ans = []
        tmp_set = set()
        rouge_idx = np.where(rouge_n == cur_rouge)[0]
        lang = ori_lang[rouge_idx]
        priority = ori_priority[rouge_idx]
        team = ori_team[rouge_idx]
        res_value = ori_res_value[rouge_idx]
        for cur_lang in set(lang):
            print cur_lang
            idx = np.where(lang == cur_lang)[0]
            cur_value = res_value[idx]
            cur_priority = priority[idx]
            cur_team = team[idx]
            tmp_ans = cur_lang + '\t'
            tmp_len = 0
            while tmp_len < len(cur_value):
                idx_min = np.where(cur_value == max(cur_value))[0]
                tmp_len += len(idx_min)
                for idx_1 in idx_min:
                    tmp_ans += cur_team[idx_1] + "-" + cur_priority[idx_1] + '\t'
                cur_value[idx_min] = -1
            ans.append(tmp_ans)
        tmp_set = set([team[i] + "-" + priority[i] for i in range(len(team))])
        write_file(ans, "../" + cur_rouge + "_ans.txt")
        final_ans = []
        final_value = []
        tmp_set = sorted(list(tmp_set))
        final_ans.append("lang,"+",".join(tmp_set))
        for cur_ans in ans:
            ans_list = cur_ans.split('\t')
            print ans_list
            final_tmp_ans = ["" for i in range(len(tmp_set) + 1)]
            final_tmp_ans[0] = ans_list[0]
            for i in range(len(tmp_set)):
                if tmp_set[i] in ans_list:
                    final_tmp_ans[i+1] = str(np.where(np.array(ans_list) == tmp_set[i])[0][0])
            final_ans.append(",".join(final_tmp_ans))
        write_file(final_ans, "../final_" + cur_rouge + "_ans.csv")
    print "end"


def get_rouge_value():
    memog_res = read_file("../data/2017_results/MSS2017_ROUGE_1.5.7_CI.csv")
    ori_res_value = np.array([line.replace(" ", "").split(",")[4] for line in memog_res])
    ori_team_priority = np.array(
        [line.replace(" ", "").split(",")[0] + "-" + line.replace(" ", "").split(",")[1] for line in memog_res])
    ori_lang = np.array([line.replace(" ", "").split(",")[2] for line in memog_res])
    rouge_n = np.array([line.replace(" ", "").split(",")[3] for line in memog_res])
    for cur_rouge in ["ROUGE-1", "ROUGE-2", "ROUGE-3", "ROUGE-4"]:
        ans = []
        # get team priority
        tmp_team_priority = []
        rouge_idx = np.where(rouge_n == cur_rouge)[0]
        lang = ori_lang[rouge_idx]
        team_priority = ori_team_priority[rouge_idx]
        res_value = ori_res_value[rouge_idx]
        for cur_lang in set(lang):
            idx = np.where(lang == cur_lang)[0]
            if len(tmp_team_priority) < len(team_priority[idx]):
                tmp_team_priority = team_priority[idx]
        tmp_team_priority = sorted(tmp_team_priority)
        print tmp_team_priority
        print len(tmp_team_priority)

        ans.append("lang\t" + "\t".join(tmp_team_priority))
        for cur_lang in set(lang):
            print cur_lang
            tmp_ans = cur_lang
            idx = np.where(lang == cur_lang)[0]
            cur_lang_value = res_value[idx]
            print team_priority[idx]
            for cur_team_priority in tmp_team_priority:
                idx_1 = np.where(team_priority[idx] == cur_team_priority)[0]
                if len(idx_1) == 0:
                    tmp_ans += "\t-"
                    continue
                tmp_ans += "\t" + str(float(cur_lang_value[idx_1[0]]) * 100)
            ans.append(tmp_ans)
        write_file(ans, "../final_" + cur_rouge + "_value.csv")
    print "end"


if __name__ == "__main__":
    get_rouge_value()
    # get_memog_value()
    # get_memog_ans()
    # get_rouge_ans()
