def add_node(tree, keys_list, value, add_func=lambda a, b: a + b):
    cur_key = keys_list[0]
    tree_child = tree.get(cur_key)
    if len(keys_list) == 1:
        if tree_child is None:
            tree[cur_key] = value
        else:
            tree[cur_key] = add_func(tree_child, value)
        return tree
    if tree_child is None:
        tree_child = {}
    keys_list_child = keys_list[1:]
    tree[cur_key] = add_node(tree_child, keys_list_child, value, add_func)
    return tree


def merge_tree(foo, bar, add_func=lambda a, b: a + b):
    if foo is None:
        return bar
    if bar is None:
        return foo
    if not isinstance(foo, dict) and not isinstance(bar, dict):
        return add_func(foo, bar)
    merged_tree = {}
    for k in set(foo.keys()) | set(bar.keys()):
        merged_tree[k] = merge_tree(foo.get(k), bar.get(k), add_func)
    return merged_tree


def read_tree(input_file, type_list, delimiter=','):
    tree = {}
    input_file = open(input_file, 'rU')
    input_file.readline()

    for line in input_file:
        line_split = line.split(delimiter)
        for i in range(len(type_list)):
            line_split[i] = type_list[i](line_split[i])
        add_node(tree, line_split[:-1], line_split[-1])
    return tree


def write_tree(tree, output_file, first_line_list, delimiter=','):
    output = open(output_file, 'w')
    output.write(delimiter.join(first_line_list) + '\n')
    write_tree_iteration(tree, output, [], delimiter)
    output.close()


def write_tree_iteration(tree, output, line_list, delimiter=','):
    if not isinstance(tree, dict):
        pop_num = 1
        if isinstance(tree, list):
            tree = [str(n) for n in tree]
            line_list.extend(tree)
            pop_num = len(tree)
        else:
            line_list.append(str(tree))
        output.write(delimiter.join(line_list) + '\n')
        for i in range(pop_num):
            line_list.pop()
        return
    for k, v in tree.items():
        line_list.append(str(k))
        write_tree_iteration(v, output, line_list, delimiter)
        line_list.pop()


def tree2matrix(tree, matrix, index):
    if not isinstance(tree, dict):
        matrix[index[0], index[1]] = tree
        index[0] += 1
        if index[0] < len(matrix):
            matrix[index[0], :] = matrix[index[0] - 1, :]
        return matrix
    for k, v in tree.items():
        matrix[index[0], index[1]] = int(k)
        index[1] += 1
        tree2matrix(v, matrix, index)
        index[1] -= 1
    return matrix


def get_tree_size(tree):
    layer_num, item_num, max_layer_num = traverse_tree(tree, 1, 0, 0)
    return item_num, max_layer_num


def traverse_tree(tree, layer_num, item_num, max_layer_num):
    v = tree.get(next(iter(tree)))
    if not isinstance(v, dict):
        item_num += len(tree)
        if max_layer_num < layer_num + 1:
            max_layer_num = layer_num + 1
        return layer_num, item_num, max_layer_num
    for k, v in tree.items():
        layer_num += 1
        layer_num, item_num, max_layer_num = traverse_tree(v, layer_num, item_num, max_layer_num)
        layer_num -= 1
    return layer_num, item_num, max_layer_num
