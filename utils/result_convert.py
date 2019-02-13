# coding: UTF-8
import shutil
import os
from file_operation import read_file
from file_operation import write_file


def cjq():
	result_path = "./multiling2017_summarization"
	for runs in os.listdir(result_path):
		run_path = result_path + "/" + runs
		for lang in os.listdir(run_path):
			lang_path = run_path + "/" + lang
			for file in os.listdir(lang_path):
				file_path = lang_path + "/"
				print file_path + file
				if file.endswith(".temp"):
					shutil.move(file_path + file, file_path + file.replace(".temp", ".txt"))
					file = file.replace(".temp", ".txt")
				if lang in ["zh", "ja"]:
					content = read_file(file_path + file)
					new_content = []
					for sentence in content:
						new_content.append(sentence.replace(" ", ""))
					write_file(new_content, file_path + file, False)

def yz():
	result_path = "./zyz"
	for runs in os.listdir(result_path):
		run_path = result_path + "/" + runs
		for lang in os.listdir(run_path):
			lang_path = run_path + "/" + lang + "/systems"
			if not os.path.exists(lang_path):
				lang_path = run_path + "/" + lang
			ori_lang_path = run_path + "/" + lang
			for file in os.listdir(lang_path):
				file_path = lang_path + "/"
				print file_path + file
				if file.endswith(".temp"):
					shutil.move(file_path + file, file_path + file.replace(".temp", ".txt"))
					file = file.replace(".temp", ".txt")
				if lang in ["zh", "ja"]:
					content = read_file(file_path + file)
					new_content = []
					for sentence in content:
						new_content.append(sentence.replace(" ", ""))
					write_file(new_content, ori_lang_path + "/" + file, False)
				else:
					content = read_file(file_path + file)
					write_file(content, ori_lang_path + "/" + file, False)
			if os.path.exists(run_path + "/" + lang + "/systems"):
				shutil.rmtree(lang_path)

if __name__ == '__main__':
	cjq()
	yz()
	