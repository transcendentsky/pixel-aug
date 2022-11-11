import pandas as pd
import torch
import os


class TBLogger:
    def __init__(self, logdir, *args, **kwargs):
        from tensorboardX import SummaryWriter
        print(f"Use Tensorboard, log at '{os.path.join(logdir, 'tb')}'")
        self.tb_logger = SummaryWriter(logdir=os.path.join(logdir, "tb"))
        self.step = 0

    def record(self, d, step=-1, tag="std"):
        if step < 0:
            step = self.step
            self.step += 1
        to_record = {}
        for k, v in d.items():
            if isinstance(v, (float, int, )) or torch.is_tensor(k):
                to_record[k] = v
        self.tb_logger.add_scalars(tag, to_record, global_step=step)


def df2record(df):
    data = df.to_dict('split')
    # print(data)
    keys = data['columns']
    values = data['data']
    record_list = []
    for i in range(len(values)):
        record = {k:v for k, v in zip(keys, values[i])}
        # print(record)
        record_list.append(record)
    return record_list

if __name__ == '__main__':
    from tutils import print_dict, tdir

    # data = pd.read_csv
    # data = pd.read_csv("D:\Documents\git-clone\oneshot-landmark\landmark\code/runs/ana/max_sim/collect_sim/record_test_linear.csv")
    data = pd.read_csv('/home1/quanquan/code/landmark/code/runs/ana/max_sim/collect_sim/record_test_linear.csv')
    # data = data.to_dict('split')
    # print_dict(data)
    data = df2record(data)
    # print(data
    tblogger = TBLogger(tdir("./tb_debug/"))
    for one_data in data:
        tblogger.record(one_data)


