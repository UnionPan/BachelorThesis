
def get_f_score(answer_label, experiment_label):
    count = 0
    answer_count = 0
    exp_count = 0
    for idx in range(len(experiment_label)):
        if answer_label[idx] == "FAVOR":
            answer_count += 1
        if experiment_label[idx] == "FAVOR":
            exp_count += 1
        if answer_label[idx] == experiment_label[idx]:
            count += 1

    precision = float(count) / exp_count
    recall = float(count) / answer_count
    f = precision * recall * 2 / (recall + precision)
    print "precision is: " + str(precision)
    print "recall is: " + str(recall)
    print "f is: " + str(f)
    return f
