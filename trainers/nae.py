import os
import numpy as np
import time
from metrics import averageMeter
from trainers.base import BaseTrainer
from trainers.logger import BaseLogger
from optimizers import get_optimizer
import torch
from torch.optim import Adam
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
from utils import roc_btw_arr


class NAETrainer(BaseTrainer):
    def train(self, model, opt, d_dataloaders, logger=None, logdir='', scheduler=None, clip_grad=None):
        cfg = self.cfg
        best_val_loss = np.inf
        time_meter = averageMeter()
        i = 0
        indist_train_loader = d_dataloaders['indist_train']
        indist_val_loader = d_dataloaders['indist_val']
        oodval_val_loader = d_dataloaders['ood_val']
        oodtarget_val_loader = d_dataloaders['ood_target']
        no_best_model_tolerance = 3
        no_best_model_count = 0

        n_ae_epoch = cfg.ae_epoch
        n_nae_epoch = cfg.nae_epoch
        ae_opt = Adam(model.parameters(), lr=cfg.ae_lr)
        # nae_opt = Adam([{'params': list(model.encoder.parameters()) + list(model.decoder.parameters())},
        #                 {'params': model.temperature_, 'lr': cfg.temperature_lr}], lr=cfg.nae_lr)

        if cfg.fix_D:
            if hasattr(model, 'temperature_'):
                nae_opt = Adam(list(model.encoder.parameters()) + [model.temperature_], lr=cfg.nae_lr)
            else:
                nae_opt = Adam(model.encoder.parameters(), lr=cfg.nae_lr)
        elif cfg.small_D_lr:
            print('small decoder learning rate')
            nae_opt = Adam([{'params': model.encoder.parameters(), 'lr': cfg.nae_lr},
                             {'params': model.decoder.parameters(), 'lr': cfg.nae_lr / 10}])
        else:
            nae_opt = Adam([{'params': list(model.encoder.parameters()) + list(model.decoder.parameters())},
                            {'params': model.temperature_, 'lr': cfg.temperature_lr}], lr=cfg.nae_lr)
            # nae_opt = Adam(model.parameters(), lr=cfg.nae_lr)

        '''AE PASS'''
        if 'load_ae' in cfg:
            n_ae_epoch = 0
            model.load_state_dict(torch.load(cfg['load_ae'])['model_state'])
            print(f'model loaded from {cfg["load_ae"]}')

        for i_epoch in range(n_ae_epoch):

            for x, y in indist_train_loader:
                i += 1

                model.train()
                x = x.to(self.device)
                y = y.to(self.device)

                start_ts = time.time()
                d_train = model.train_step_ae(x, ae_opt, clip_grad=0.1)  # todo: clip_grad
                time_meter.update(time.time() - start_ts)
                logger.process_iter_train(d_train)

                if i % cfg.print_interval == 0:
                    d_train = logger.summary_train(i)
                    print(f"Iter [{i:d}] Avg Loss: {d_train['loss/train_loss_']:.4f} Elapsed time: {time_meter.sum:.4f}")
                    time_meter.reset()

                if i % cfg.val_interval == 0:
                    model.eval()
                    for val_x, val_y in indist_val_loader:
                        val_x = val_x.to(self.device)
                        val_y = val_y.to(self.device)

                        d_val = model.validation_step(val_x, y=val_y)
                        logger.process_iter_val(d_val)
                    d_val = logger.summary_val(i)
                    val_loss = d_val['loss/val_loss_']
                    print(d_val['print_str'])
                    best_model = val_loss < best_val_loss

                    if i % cfg.save_interval == 0 or best_model:
                        self.save_model(model, logdir, best=best_model, i_iter=i)
                    if best_model:
                        print(f'Iter [{i:d}] best model saved {val_loss} <= {best_val_loss}')
                        best_val_loss = val_loss
                    else:
                        no_best_model_count += 1
                        if no_best_model_count > no_best_model_tolerance:
                            break

            if no_best_model_count > no_best_model_tolerance:
                print('terminating autoencoder training since validation loss does not decrease anymore')
                break

        '''EBAE PASS'''
        # load best autoencoder model
        #  model.load_state_dict(torch.load(os.path.join(logdir, 'model_best.pkl'))['model_state'])
        # print('best model loaded')
        i = 0
        for i_epoch in tqdm(range(n_nae_epoch)):
            for x, _ in tqdm(indist_train_loader):
                i += 1

                x = x.cuda(self.device)
                if cfg.get('flatten', False):
                    x = x.view(len(x), -1)
                d_result = model.train_step(x, nae_opt)
                logger.process_iter_train_nae(d_result)

                if i % cfg.print_interval == 1:
                    logger.summary_train_nae(i)
 
                if i % cfg.val_interval == 1:
                    '''AUC'''
                    in_pred = self.predict(model, indist_val_loader, self.device)
                    ood1_pred = self.predict(model, oodval_val_loader, self.device)
                    auc_val = roc_btw_arr(ood1_pred, in_pred)
                    ood2_pred = self.predict(model, oodtarget_val_loader, self.device)
                    auc_target = roc_btw_arr(ood2_pred, in_pred)
                    d_result = {'nae/auc_val': auc_val, 'nae/auc_target': auc_target}
                    print(logger.summary_val_nae(i, d_result))
                    torch.save({'model_state': model.state_dict()}, f'{logdir}/nae_iter_{i}.pkl')

            torch.save(model.state_dict(), f'{logdir}/nae_{i_epoch}.pkl')
        torch.save(model.state_dict(), f'{logdir}/nae.pkl')

        '''EBAE sample'''
        nae_sample = model.sample(n_sample=30, device=self.device, replay=True)
        img_grid = make_grid(nae_sample['sample_x'].detach().cpu(), nrow=10, range=(0, 1))
        logger.writer.add_image('nae/sample', img_grid, i + 1)
        save_image(img_grid, f'{logdir}/nae_sample.png')

        '''AUC'''
        in_pred = self.predict(model, indist_val_loader, self.device)
        ood1_pred = self.predict(model, oodval_val_loader, self.device)
        auc_val = roc_btw_arr(ood1_pred, in_pred)
        ood2_pred = self.predict(model, oodtarget_val_loader, self.device)
        auc_target = roc_btw_arr(ood2_pred, in_pred)
        d_result = {'nae/auc_val': auc_val, 'nae/auc_target': auc_target}
        print(d_result)

        return model, auc_val

    def predict(self, m, dl, device, flatten=False):
        """run prediction for the whole dataset"""
        l_result = []
        for x, _ in dl:
            with torch.no_grad():
                if flatten:
                    x = x.view(len(x), -1)
                pred = m.predict(x.cuda(device)).detach().cpu()
            l_result.append(pred)
        return torch.cat(l_result)


class NAELogger(BaseLogger):
    def __init__(self, tb_writer):
        super().__init__(tb_writer)
        self.train_loss_meter_nae = averageMeter()
        self.val_loss_meter_nae = averageMeter()
        self.d_train_nae = {}
        self.d_val_nae = {}

    def process_iter_train_nae(self, d_result):
        self.train_loss_meter.update(d_result['loss'])
        self.d_train_nae = d_result

    def summary_train_nae(self, i):
        d_result = self.d_train_nae
        writer = self.writer
        writer.add_scalar('nae/loss', d_result['loss'], i)
        writer.add_scalar('nae/energy_diff', d_result['pos_e'] - d_result['neg_e'], i)
        writer.add_scalar('nae/pos_e', d_result['pos_e'], i)
        writer.add_scalar('nae/neg_e', d_result['neg_e'], i)
        writer.add_scalar('nae/encoder_l2', d_result['encoder_norm'], i)
        writer.add_scalar('nae/decoder_l2', d_result['decoder_norm'], i)
        if 'neg_e_x0' in d_result:
            writer.add_scalar('nae/neg_e_x0', d_result['neg_e_x0'], i)
        if 'neg_e_z0' in d_result:
            writer.add_scalar('nae/neg_e_z0', d_result['neg_e_z0'], i)
        if 'temperature' in d_result:
            writer.add_scalar('nae/temperature', d_result['temperature'], i)
        if 'sigma' in d_result:
            writer.add_scalar('nae/sigma', d_result['sigma'], i)
        if 'delta_term' in d_result:
            writer.add_scalar('nae/delta_term', d_result['delta_term'], i)
        if 'gamma_term' in d_result:
            writer.add_scalar('nae/gamma_term', d_result['gamma_term'], i)


        '''images'''
        x_neg = d_result['x_neg']
        recon_neg = d_result['recon_neg']
        img_grid = make_grid(x_neg, nrow=10, range=(0, 1))
        writer.add_image('nae/sample', img_grid, i)
        img_grid = make_grid(recon_neg, nrow=10, range=(0, 1), normalize=True)
        writer.add_image('nae/sample_recon', img_grid, i)

        # to uint8 and save as array
        x_neg = (x_neg.permute(0,2,3,1).numpy() * 256.).clip(0, 255).astype('uint8')
        recon_neg = (recon_neg.permute(0,2,3,1).numpy() * 256.).clip(0, 255).astype('uint8')
        # save_image(img_grid, f'{writer.file_writer.get_logdir()}/nae_sample_{i}.png')
        np.save(f'{writer.file_writer.get_logdir()}/nae_neg_{i}.npy', x_neg)
        np.save(f'{writer.file_writer.get_logdir()}/nae_neg_recon_{i}.npy', recon_neg)


    def summary_val_nae(self, i, d_result):
        l_print_str = [f'Iter [{i:d}]']
        for key, val in d_result.items():
            self.writer.add_scalar(key, val, i)
            l_print_str.append(f'{key}: {val:.4f}')
        print_str = ' '.join(l_print_str)
        return print_str






