import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import mat_utils
import utils

from torch.autograd import Variable
from builtins import zip as xzip


class Trainer:

    def __init__(self, args, model, data_info):
        self.model = model
        self.data_info = data_info

        self.list_params = list(model.parameters())

        if args.half_acc:
            self.copy_params = [param.clone().detach() for param in self.list_params]
            self.model = self.model.half()

            for param in self.copy_params:
                param.requires_grad = True
                param.grad = param.data.new_zeros(param.size())

            self.optimizer = optim.Adam(self.copy_params, args.learn_rate, weight_decay = args.weight_decay)
        else:
            self.optimizer = optim.Adam(self.list_params, args.learn_rate, weight_decay = args.weight_decay)

        self.depth = args.depth
        self.num_joints = args.num_joints
        self.side_in = args.side_in
        self.stride = args.stride
        self.depth_range = args.depth_range

        self.half_acc = args.half_acc
        self.joint_space = args.joint_space
        self.do_track = args.do_track

        self.learn_rate = args.learn_rate
        self.num_epochs = args.n_epochs
        self.grad_norm = args.grad_norm
        self.grad_scaling = args.grad_scaling

        self.thresh = dict(
            solid = args.thresh_solid,
            close = args.thresh_close,
            rough = args.thresh_rough
        )
        self.criterion = nn.__dict__[args.criterion + 'Loss'](reduction = 'mean').cuda()


    def joint_train(self, epoch, data_loader, cuda_device):
        n_batches = len(data_loader)

        cam_loss_avg = 0
        mat_loss_avg = 0
        recon_loss_avg = 0
        total = 0

        side_out = (self.side_in - 1) / self.stride + 1

        do_track = self.do_track and (epoch != 1)

        for i, (image, true_cam, true_mat, true_val, intrinsics) in enumerate(data_loader):

            image = image.to(cuda_device)

            true_cam = true_cam.to(cuda_device)
            true_mat = true_mat.to(cuda_device)
            true_val = true_val.to(cuda_device)

            intrinsics = intrinsics.to(cuda_device)

            batch = image.size(0)

            cam_feat, mat_feat = self.model(image)

            heat_mat = mat_utils.to_heatmap(mat_feat, self.num_joints, side_out, side_out)

            heat_cam = utils.to_heatmap(cam_feat, self.depth, self.num_joints, side_out, side_out)

            spec_mat = mat_utils.decode(heat_mat, self.side_in)

            mat_loss = self.criterion(spec_mat.view(-1, 2)[true_val.view(-1)], true_mat.view(-1, 2)[true_val.view(-1)])

            mat_loss_avg += mat_loss.item() * batch

            key_index = self.data_info.key_index

            relat_cam = utils.decode(heat_cam, self.depth_range)

            relat_cam = relat_cam - relat_cam[:, key_index:key_index + 1]

            spec_cam = relat_cam + true_cam[:, key_index:key_index + 1]

            cam_loss = self.criterion(spec_cam.view(-1, 3)[true_val.view(-1)], true_cam.view(-1, 3)[true_val.view(-1)])

            cam_loss_avg += cam_loss.item() * batch

            loss = cam_loss + mat_loss

            if do_track:
                recon_cam = utils.get_recon_cam(spec_mat, relat_cam, intrinsics)

                recon_loss = self.criterion(recon_cam.view(-1, 3)[true_val.view(-1)], true_cam.view(-1, 3)[true_val.view(-1)])

                recon_loss_avg += recon_loss.item() * batch

                loss = loss * 0.5 + recon_loss

            self.optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(self.list_params, self.grad_norm)
            self.optimizer.step()

            total += batch

            message = '| train Epoch[%d] [%d/%d]' % (epoch, i, n_batches)
            message += '  Cam Loss: %1.4f' % (cam_loss.item())
            message += '  Mat Loss: %1.4f' % (mat_loss.item())

            if do_track:
                message += '  Recon Loss: %1.4f' % (recon_loss.item())

            print(message)

        cam_loss_avg /= total
        mat_loss_avg /= total
        recon_loss_avg /= total

        message = '=> train Epoch[%d]  Cam Loss: %1.4f  Mat Loss: %1.4f' % (epoch, cam_loss_avg, mat_loss_avg)

        if do_track:
            message += '  Recon Loss: %1.4f' % (recon_loss_avg)

        print('\n' + message + '\n')

        return dict(cam_train_loss = cam_loss_avg, mat_train_loss = mat_loss_avg, recon_train_loss = recon_loss_avg)


    def cam_train(self, epoch, data_loader, cuda_device):
        n_batches = len(data_loader)

        loss_avg = 0
        total = 0

        side_out = (self.side_in - 1) / self.stride + 1

        for i, (image, true_cam, true_val) in enumerate(data_loader):

            image = image.to(cuda_device)

            true_cam = true_cam.to(cuda_device)
            true_val = true_val.to(cuda_device)

            batch = image.size(0)

            cam_feat = self.model(image)

            heat_cam = utils.to_heatmap(cam_feat, self.depth, self.num_joints, side_out, side_out)

            key_index = self.data_info.key_index

            relat_cam = utils.decode(heat_cam, self.depth_range)

            relat_cam = relat_cam - relat_cam[:, key_index:key_index + 1]

            spec_cam = relat_cam + true_cam[:, key_index:key_index + 1]

            loss = self.criterion(spec_cam.view(-1, 3)[true_val.view(-1)], true_cam.view(-1, 3)[true_val.view(-1)])

            loss_avg += loss.item() * batch

            self.optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(self.list_params, self.grad_norm)
            self.optimizer.step()

            total += batch

            print('| train Epoch[%d] [%d/%d]  Loss %1.4f' % (epoch, i, n_batches, loss.item()))

        loss_avg /= total

        print('\n=> train Epoch[%d]  Cam Loss: %1.4f\n' % (epoch, loss_avg))

        return dict(cam_train_loss = loss_avg)


    def train(self, epoch, data_loader):
        self.model.train()
        self.adapt_learn_rate(epoch)

        if self.joint_space:
            return self.joint_train(epoch, data_loader, torch.device('cuda'))
        else:
            return self.cam_train(epoch, data_loader, torch.device('cuda'))


    def joint_test(self, epoch, test_loader, cuda_device):
        n_batches = len(test_loader)

        cam_loss_avg = 0
        mat_loss_avg = 0
        total = 0

        side_out = (self.side_in - 1) / self.stride + 1

        mat_stats = []
        cam_stats = []
        det_stats = []

        for i, (image, true_cam, true_mat, back_rotation, true_val, intrinsics) in enumerate(test_loader):

            image = image.half().to(cuda_device) if self.half_acc else image.to(cuda_device)

            true_cam = true_cam.to(cuda_device)
            true_mat = true_mat.to(cuda_device)
            true_val = true_val.to(cuda_device)

            batch = image.size(0)

            with torch.no_grad():
                cam_feat, mat_feat = self.model(image)

                if self.half_acc:
                    cam_feat = cam_feat.float()
                    mat_feat = mat_feat.float()

                heat_mat = mat_utils.to_heatmap(mat_feat, self.num_joints, side_out, side_out)

                heat_cam = utils.to_heatmap(cam_feat, self.depth, self.num_joints, side_out, side_out)

                spec_mat = mat_utils.decode(heat_mat, self.side_in)

                key_index = self.data_info.key_index

                relat_cam = utils.decode(heat_cam, self.depth_range)

                relat_cam = relat_cam - relat_cam[:, key_index:key_index + 1]

                spec_cam = relat_cam + true_cam[:, key_index:key_index + 1]

                cam_loss = self.criterion(spec_cam.view(-1, 3)[true_val.view(-1)], true_cam.view(-1, 3)[true_val.view(-1)])
                mat_loss = self.criterion(spec_mat.view(-1, 2)[true_val.view(-1)], true_mat.view(-1, 2)[true_val.view(-1)])

            cam_loss_avg += cam_loss.item() * batch
            mat_loss_avg += mat_loss.item() * batch

            total += batch

            print('| test Epoch[%d] [%d/%d]  Cam Loss: %1.4f  Mat Loss: %1.4f' % (epoch, i, n_batches, cam_loss.item(), mat_loss.item()))

            true_val = true_val.cpu().numpy().astype(np.bool)

            spec_mat = spec_mat.cpu().numpy()
            true_mat = true_mat.cpu().numpy()

            mat_stats.append(mat_utils.analyze(spec_mat, true_mat, true_val, self.side_in))

            spec_cam = spec_cam.cpu().numpy()
            true_cam = true_cam.cpu().numpy()

            spec_cam = np.einsum('Bij,BCj->BCi', back_rotation, spec_cam)
            true_cam = np.einsum('Bij,BCj->BCi', back_rotation, true_cam)

            cam_stats.append(utils.analyze(spec_cam, true_cam, true_val, self.data_info.mirror, self.thresh))

            if self.do_track:
                relat_cam = relat_cam.cpu().numpy()

                deter_cam = utils.get_deter_cam(spec_mat, relat_cam, intrinsics)

                deter_cam = np.einsum('Bij,BCj->BCi', back_rotation, deter_cam)

                det_stats.append(utils.analyze(deter_cam, true_cam, true_val, self.data_info.mirror, self.thresh))

        cam_loss_avg /= total
        mat_loss_avg /= total

        record = dict(cam_test_loss = cam_loss_avg, mat_test_loss = mat_loss_avg)

        record.update(mat_utils.parse_epoch(mat_stats))
        record.update(utils.parse_epoch(cam_stats))

        print('\n=> test Epoch[%d]  Cam Loss: %1.4f  Mat Loss: %1.4f\n' % (epoch, cam_loss_avg, mat_loss_avg))

        print('=>mat_mean: %1.3f  [oks]: %1.3f\n' % (record['mat_mean'], record['score_oks']))

        print('=>[SPEC] cam_mean: %1.3f  [pck]: %1.3f  [auc]: %1.3f\n' % (record['cam_mean'], record['score_pck'], record['score_auc']))

        if self.do_track:

            track_rec = utils.parse_epoch(det_stats)

            print('=>[DETER] cam_mean: %1.3f  [pck]: %1.3f  [auc]: %1.3f\n' % (track_rec['cam_mean'], track_rec['score_pck'], track_rec['score_auc']))

            for key in track_rec:
                record['recon_' + key] = track_rec[key]

        return record


    def cam_test(self, epoch, test_loader, cuda_device):
        n_batches = len(test_loader)

        loss_avg = 0
        total = 0

        side_out = (self.side_in - 1) / self.stride + 1

        cam_stats = []

        for i, (image, true_cam, back_rotation, true_val) in enumerate(test_loader):

            image = image.to(cuda_device)

            true_cam = true_cam.to(cuda_device)
            true_val = true_val.to(cuda_device)

            batch = image.size(0)

            with torch.no_grad():
                cam_feat = self.model(image)

                heat_cam = utils.to_heatmap(cam_feat, self.depth, self.num_joints, side_out, side_out)

                key_index = self.data_info.key_index

                relat_cam = utils.decode(heat_cam, self.depth_range)

                relat_cam = relat_cam - relat_cam[:, key_index:key_index + 1]

                spec_cam = relat_cam + true_cam[:, key_index:key_index + 1]

                loss = self.criterion(spec_cam.view(-1, 3)[true_val.view(-1)], true_cam.view(-1, 3)[true_val.view(-1)])

            loss_avg += loss.item() * batch

            total += batch

            true_val = true_val.cpu().numpy().astype(np.bool)

            spec_cam = spec_cam.cpu().numpy()
            true_cam = true_cam.cpu().numpy()

            spec_cam = np.einsum('Bij,BCj->BCi', back_rotation, spec_cam)
            true_cam = np.einsum('Bij,BCj->BCi', back_rotation, true_cam)

            cam_stats.append(utils.analyze(spec_cam, true_cam, true_val, self.data_info.mirror, self.thresh))

            print('| test Epoch[%d] [%d/%d]  Cam Loss %1.4f' % (epoch, i, n_batches, loss.item()))

        loss_avg /= total

        record = dict(test_loss = loss_avg)
        record.update(utils.parse_epoch(cam_stats))

        print('\n=> test Epoch[%d]  Cam Loss: %1.4f\n' % (epoch, loss_avg))

        print('=>[SPEC] cam_mean: %1.3f  [pck]: %1.3f  [auc]: %1.3f\n' % (record['cam_mean'], record['score_pck'], record['score_auc']))

        return record


    def test(self, epoch, test_loader):
        self.model.eval()

        if self.joint_space:
            return self.joint_test(epoch, test_loader, torch.device('cuda'))
        else:
            return self.cam_test(epoch, test_loader, torch.device('cuda'))


    def adapt_learn_rate(self, epoch):
        if epoch - 1 < self.num_epochs * 0.6:
            learn_rate = self.learn_rate
        elif epoch - 1 < self.num_epochs * 0.9:
            learn_rate = self.learn_rate * 0.2
        else:
            learn_rate = self.learn_rate * 0.04

        if self.do_track and epoch != 1:
            learn_rate /= 2

        for group in self.optimizer.param_groups:
            group['lr'] = learn_rate
