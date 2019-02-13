from utils.file_operation import read_file


def get_hlda_message(path):
    mode = read_file(path + "/mode")
    mode_assign = read_file(path + "/mode.assign")
