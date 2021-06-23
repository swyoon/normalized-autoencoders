import numpy as np
from metrics import averageMeter


class BaseLogger:
    """BaseLogger that can handle most of the logging
    logging convention
    ------------------
    'loss' has to be exist in all training settings
    endswith('_') : scalar
    endswith('@') : image
    """
    def __init__(self, tb_writer):
        """tb_writer: tensorboard SummaryWriter"""
        self.writer = tb_writer
        self.train_loss_meter = averageMeter()
        self.val_loss_meter = averageMeter()
        self.d_train = {}
        self.d_val = {}

    def process_iter_train(self, d_result):
        self.train_loss_meter.update(d_result['loss'])
        self.d_train = d_result

    def summary_train(self, i):
        self.d_train['loss/train_loss_'] = self.train_loss_meter.avg 
        for key, val in self.d_train.items():
            if key.endswith('_'):
                self.writer.add_scalar(key, val, i)
            if key.endswith('@'):
                if val is not None:
                    self.writer.add_image(key, val, i)

        result = self.d_train
        self.d_train = {}
        return result

    def process_iter_val(self, d_result):
        self.val_loss_meter.update(d_result['loss'])
        self.d_val = d_result

    def summary_val(self, i):
        self.d_val['loss/val_loss_'] = self.val_loss_meter.avg 
        l_print_str = [f'Iter [{i:d}]']
        for key, val in self.d_val.items():
            if key.endswith('_'):
                self.writer.add_scalar(key, val, i)
                l_print_str.append(f'{key}: {val:.4f}')
            if key.endswith('@'):
                if val is not None:
                    self.writer.add_image(key, val, i)

        print_str = ' '.join(l_print_str)

        result = self.d_val
        result['print_str'] = print_str
        self.d_val = {}
        return result
