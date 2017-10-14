import datetime
import os
import math
import multiprocessing
from multiway_tree import merge_tree
import matplotlib as mpl
import matplotlib.pyplot as plt


def process_sub_list(process_one_file_func, mr_sub_list, *args):
    for file in mr_sub_list:
        return_val = process_one_file_func(file, *args)
    return return_val


def one_task(result_queue, process_one_file_func, mr_sub_list, *args):
    result_queue.put(process_sub_list(process_one_file_func, mr_sub_list, *args))


def process_file_list(mr_folder, process_one_file_func, is_multiprocess=True, *args):
    """

    :param mr_folder:
    :param process_one_file_func:

    :param is_multiprocess:
    :param args:
    :return:

    """
    statistic_tree = {}
    files = os.listdir(mr_folder)
    files = [os.path.join(mr_folder, file) for file in files]
    if is_multiprocess:
        result_queue = multiprocessing.Manager().Queue()
        pool_num = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(pool_num)
        pool_size = int(math.ceil(len(files) / float(pool_num)))
        print('%s: start process mr.' % datetime.datetime.now())
        for i in range(0, len(files), pool_size):
            file_sub_list = files[i: i + pool_size]
            temp = [result_queue, process_one_file_func, file_sub_list]
            temp.extend(list(args))
            pool.apply_async(one_task, args=tuple(temp))
        pool.close()
        pool.join()
        print('%s: start merge result.' % datetime.datetime.now())
        while not result_queue.empty():
            value = result_queue.get(True)
            statistic_tree = merge_tree(statistic_tree, value)
        print('%s: end merge result.' % datetime.datetime.now())
    else:
        for i, file in enumerate(files):
            print('%s: (%d|%d) read %s' % (datetime.datetime.now(), i + 1, len(files), file))
            statistic_tree = process_one_file_func(file, *args)
    return statistic_tree


def show_channels(channels, is_show_image=False, is_show_immediately=True, is_show_colorbar=True, save_fig=None, *args,
                  **kwargs):
    if not is_show_image:
        return
    mpl.rcParams.update({'font.size': 7, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})
    shape = channels.shape
    channel_num = shape[0]
    if len(shape) == 2:
        plt.imshow(channels, *args, **kwargs)
        if is_show_colorbar:
            plt.colorbar()
    else:
        row = int(math.sqrt(channel_num))
        column = int(math.ceil(channel_num / row))
        fig, axes = plt.subplots(row, column)
        for i in range(channel_num):
            cur_ax = axes if channel_num == 1 else axes[i] if channel_num <= 3 else axes[
                int(i / column), int(i % column)]
            im = cur_ax.imshow(channels[i, :, :], *args, **kwargs)
            if is_show_colorbar:
                fig.colorbar(im, ax=cur_ax)
    if save_fig:
        fig = plt.gca()
        fig.set_size_inches(18.5, 10.5)
        fig.savefig(save_fig)
    if is_show_immediately:
        plt.show()
    return


def index_split(index_str):
    indexes = []
    index_str_split = index_str.split('-')
    for index_section in index_str_split:
        index_section_split = index_section.split(':')
        section = list(map(int, index_section_split))
        if len(section) == 2:
            section = list(range(*section))
        indexes += section
    return indexes
