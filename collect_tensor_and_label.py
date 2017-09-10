import os
import sys


def get_indexes(first_line, delimiter, header):
    first_line_split = first_line.split(delimiter)
    for i, item in enumerate(first_line_split):
        if item == header:
            return i


def collect_tensor_and_label(path_list, summary, target_file_name):
    for path in path_list:
        if os.path.isfile(path):
            folder, file_name = os.path.split(path)
            if file_name == target_file_name:
                with open(path) as tensor_and_label_file:
                    first_line = tensor_and_label_file.readline()
                    first_line = first_line.strip('\n')
                    tensor_file_index = get_indexes(first_line, ',', 'filename')
                    chessman_index = get_indexes(first_line, ',', 'chessman')
                    state_index = get_indexes(first_line, ',', 'state')
                    x_index = get_indexes(first_line, ',', 'x')
                    y_index = get_indexes(first_line, ',', 'y')
                    class_index = get_indexes(first_line, ',', 'class')
                    for line in tensor_and_label_file:
                        line = line.strip('\n')
                        line_split = line.split(',')
                        tensor_file = line_split[tensor_file_index]
                        if not os.path.isfile(tensor_file) or os.path.splitext(tensor_file)[1] != ".npy":
                            print(tensor_file + " not exist!")
                            continue
                        chessman = line_split[chessman_index]
                        state = line_split[state_index]
                        x = line_split[x_index]
                        y = line_split[y_index]
                        class_belong = line_split[class_index]
                        summary.append((tensor_file, chessman, state, x, y, class_belong))
        else:
            sub_path_list = os.listdir(path)
            sub_path_list = [os.path.join(path, sub_path) for sub_path in sub_path_list]
            summary = collect_tensor_and_label(sub_path_list, summary, target_file_name)
    return summary


if __name__ == '__main__':

    label_file_name = sys.argv[1]
    output_file = sys.argv[2]
    path_list = sys.argv[3:]
    summary = collect_tensor_and_label(path_list, [], label_file_name)
    with open(output_file, 'w') as collect_file:
        collect_file.write('filename, chessman, state, x, y, class\n')
        for item in summary:
            item = [str(o) for o in item]
            collect_file.write(','.join(item) + '\n')
