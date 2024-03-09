import os.path

from torch.utils.tensorboard import SummaryWriter
import time


class Logger:
    def __init__(self, args):
        self.args = args
        self.tb_path = args.tb_path

        self.seed = args.seed
        self.task = args.env
        self.type = args.type
        self.degree = args.degree

        self.history_dict = {}

        time_tuple = time.localtime(time.time())
        date = ""
        for t in time_tuple:
            date += (str(t) + "-")
        self.tb_path = os.path.join(self.tb_path, "sac_{}_{}_{}_{}_{}".format(self.task, self.type, self.degree, self.seed, self.args.utd))
        if not os.path.exists(self.tb_path):
            os.makedirs(self.tb_path, exist_ok=True)
        self.writer = SummaryWriter(self.tb_path)

    def add_scalar(self, c, dict_data, iter=None):
        for key in dict_data.keys():
            if iter is not None:
                self.writer.add_scalar("{}/{}".format(c, key), dict_data[key], iter)
            else:
                if key in self.history_dict.keys():
                    self.history_dict[key] += 1
                else:
                    self.history_dict[key] = 0
                self.writer.add_scalar("{}/{}".format(c, key), dict_data[key], self.history_dict[key])

    def add_list(self, c, dict_data):
        for key in dict_data.keys():
            for data in dict_data[key]:
                log_key = "{}/{}".format(c, key)
                if log_key not in self.history_dict.keys():
                    self.history_dict[log_key] = 0
                else:
                    self.history_dict[log_key] += 1
                self.writer.add_scalar(log_key, data, self.history_dict[log_key])