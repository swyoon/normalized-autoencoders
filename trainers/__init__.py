from trainers.logger import BaseLogger 
from trainers.base import BaseTrainer
from trainers.nae import NAETrainer, NAELogger


def get_trainer(cfg):
    # get trainer by specified `trainer` field
    # if not speficied, get trainer by model type
    trainer_type = cfg.get('trainer', None)
    arch = cfg['model']['arch']
    device = cfg['device']
    if trainer_type == 'nae':
        trainer = NAETrainer(cfg['training'], device=device)
    else:
        trainer = BaseTrainer(cfg['training'], device=device)
    return trainer


def get_logger(cfg, writer):
    logger_type = cfg['logger']
    if logger_type == 'nae':
        logger = NAELogger(writer)
    else:
        logger = BaseLogger(writer)
    return logger 
