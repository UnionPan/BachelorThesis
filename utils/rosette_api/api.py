# coding=utf-8
import json
from utils.file_operation import write_file
import os

from rosette.api import API, DocumentParameters, MorphologyOutput


def tokenize(tokens_data, key='c573c91d6f074690b5bb6af453fed596'):
    # create an API instance
    print "tokenize"
    api = API(user_key=key)
    params = DocumentParameters()
    params["content"] = tokens_data
    results = api.tokens(params)
    ans = json.dumps(results, indent=2, ensure_ascii=False, sort_keys=True).encode('utf-8').decode('utf-8')
    ans_json = json.loads(ans)
    print "tokenize finished"
    return ans_json['tokens']


# bef6edaff421021d87b86f1aedf3d2a0
def lemmatize(lemma_data, key='c573c91d6f074690b5bb6af453fed596'):
    print "lemmas"
    api = API(user_key=key)
    params = DocumentParameters()
    params["content"] = lemma_data
    results = api.morphology(params, MorphologyOutput.LEMMAS)
    ans = json.dumps(results, indent=2, ensure_ascii=False, sort_keys=True).encode('utf-8').decode('utf-8')
    ans_json = json.loads(ans)
    print "lemmas finished"
    tokens = ans_json['tokens']
    lemmas = ans_json['lemmas']
    for i in range(len(tokens)):
        if lemmas[i] is None:
            lemmas[i] = tokens[i]
    return tokens, lemmas


def sen_tagging(tokens_data, key='c573c91d6f074690b5bb6af453fed596'):
    # create an API instance
    print "sen_tagging"
    api = API(user_key=key)
    params = DocumentParameters()
    params["content"] = tokens_data
    results = api.sentences(params)
    ans = json.dumps(results, indent=2, ensure_ascii=False, sort_keys=True).encode('utf-8')
    ans_json = json.loads(ans)
    print "sen_tagging finished"
    return ans_json['sentences']


if __name__ == "__main__":
    test_con = "ამერიკის კალათბურთის ასოციაცია (ინგლ. Basketball Association of America, BAA) 1946 წელს ყინულის ჰოკეის არენების მფლობელების მიერ შეიქმნა. 1946 წლის 1 ნოემბერს, ტორონტოში ონტარიოს შტატში, კანადაში პირველი მატჩი შედგა, რომელზეც „ტორონტო ჰასკისმა“ „ნიუ-იორკ ნიკერბოკერსს“ მეიფლ ლივზ გარდენში უმასპინძლა — ამჟამად ეს მოვლენა აღიარებულია, როგორც NBA-ის ისტორიაში პირველი მატჩი. "
    result = tokenize(test_con)
    print result
    print type(result)
    print ("\n".join(result))

