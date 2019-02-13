# coding=utf-8
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


if __name__ == "__main__":
    print "\n".join(word_tokenize(
        u"Berlin nằm ở phía Đông Bắc nước Đức và được bao quanh bởi tiểu bang Brandenburg. Berlin cách biên giới với Ba Lan 70 km và là một trong những khu vực đông dân cư nhất nước Đức."))

