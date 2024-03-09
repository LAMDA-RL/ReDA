from offlinerl.data.meta import load_meta_buffer

interval = 20000
env_1_idx = [500000 // interval]
env_3_idx = [100000 // interval, 500000 // interval, 900000 // interval]
env_5_idx = [100000 // interval, 200000 // interval, 500000 // interval, 800000 // interval, 900000 // interval]
env_10_idx = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 48]

task_map = {
    '1': env_1_idx,
    '3': env_3_idx,
    '5': env_5_idx,
    '10': env_10_idx,
}


def task_generation(task, level_type='1', param_type='grav', degrees=[0.5, 1.0, 1.5], with_eval=False):
    task_data_list = []
    task_data_list_eval = []
    idx_eval = []
    for idx, param in enumerate(degrees):
        task_data_list.append((task_map[str(level_type)], param_type, param))
    for idx in range(50):
        if idx not in task_map[str(level_type)]:
            idx_eval.append(idx)
    for idx, param in enumerate(degrees):
        task_data_list_eval.append((idx_eval, param_type, param))
    meta_buffer = load_meta_buffer(task, task_data_list)
    meta_buffer_eval = load_meta_buffer(task, task_data_list_eval)
    print(meta_buffer.keys())
    print(meta_buffer_eval.keys())
    if with_eval:
        return meta_buffer, meta_buffer_eval

    return meta_buffer

# task_map = {
#     'r': "random",
#     'm': "medium",
#     'e': "expert",
#     'me': "medium-expert",
#     'mr': "medium-replay",
#     'fr': "full-replay",
# }
#
#
# def task_generation(task, level_type='random', param_type='grav', degrees=[0.5, 1.0, 1.5]):
#     levels = [task_map[level.strip()] for level in level_type.split('-')]
#     task_data_list = []
#     for idx, param in enumerate(degrees):
#         if idx >= len(levels):
#             level = levels[-1]
#         else:
#             level = levels[idx]
#
#         task_data_list.append((level, param_type, param))

    # if level_type == 'other0':
    #     task_data_list = [
    #         ('random', 'grav', 0.5),
    #         ('expert', 'grav', 1.0),
    #         ('random', 'grav', 1.5),
    #     ]
    # else:
    #     task_data_list = []
    #     for deg in degrees:
    #         task_data_list.append((level_type, param_type, deg))

    # meta_buffer = load_meta_buffer(task, task_data_list)
    #
    # return meta_buffer

