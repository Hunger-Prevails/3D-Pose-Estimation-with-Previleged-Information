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

        self.list_params = [model.parameters()]

        if args.do_complement:
            self.comp_loss_weight = args.comp_loss_weight

            self.comp_linear = nn.Linear(args.num_joints, args.num_joints)

            if args.n_cudas:
                self.comp_linear = self.comp_linear.cuda()

            self.list_params.append(self.comp_linear.parameters())

        self.n_cudas = args.n_cudas
        self.depth = args.depth
        self.num_joints = args.num_joints
        self.side_in = args.side_in
        self.stride = args.stride
        self.depth_range = args.depth_range
        self.flip_test = args.flip_test
        self.joint_space = args.joint_space

        self.learn_rate = args.learn_rate
        self.num_epochs = args.n_epochs
        self.grad_norm = args.grad_norm

        self.thresh = dict(
            solid = args.thresh_solid,
            close = args.thresh_close,
            rough = args.thresh_rough
        )
        self.optimizer = optim.Adam(self.get_params(), args.learn_rate, weight_decay = args.weight_decay)

        self.criterion = nn.__dict__[args.criterion + 'Loss'](reduction = 'mean')

        if args.n_cudas:
            self.criterion = self.criterion.cuda()


    def joint_train(self, epoch, data_loader, comp_loader, cuda_device):
        n_batches = len(data_loader)

        cam_loss_avg = 0
        mat_loss_avg = 0
        total = 0

        side_out = (self.side_in - 1) / self.stride + 1

        comp_data_iter = iter(comp_loader) if comp_loader != None else None

        for i, (image, true_cam, true_mat, valid_mask) in enumerate(data_loader):

            if self.n_cudas:
                image = image.to(cuda_device)

                true_cam = true_cam.to(cuda_device)

                true_mat = true_mat.to(cuda_device)

                valid_mask = valid_mask.unsqueeze(-1).to(cuda_device)

            batch = image.size(0)

            cam_feat, mat_feat = self.model(image)

            heat_mat = mat_utils.to_heatmap(mat_feat, self.num_joints, side_out, side_out)

            heat_cam = utils.to_heatmap(cam_feat, self.depth, self.num_joints, side_out, side_out)

            spec_mat = mat_utils.decode(heat_mat, self.side_in)

            mat_loss = self.criterion(spec_mat * valid_mask, true_mat * valid_mask)

            key_index = self.data_info.key_index

            relat_cam = utils.decode(heat_cam, self.depth_range)

            relat_cam = relat_cam - relat_cam[:, key_index:key_index + 1]

            spec_cam = relat_cam + true_cam[:, key_index:key_index + 1]

            cam_loss = self.criterion(spec_cam * mask_valid, true_cam * mask_valid)

            loss = cam_loss + mat_loss

            if comp_loader != None:
                try:
                    comp_image, comp_true_mat, comp_valid_mask = next(comp_data_iter)
                except:
                    comp_data_iter = iter(comp_loader)
                    comp_image, comp_true_mat, comp_valid_mask = next(comp_data_iter)

                if self.n_cudas:
                    comp_image = comp_image.to(cuda_device)

                    comp_true_mat = comp_true_mat.to(cuda_device)

                    comp_valid_mask = comp_valid_mask.to(cuda_device)

                comp_batch = comp_image.size(0)

                comp_cam_feat, comp_mat_feat = self.model(comp_image)

                comp_heat_mat = mat_utils.to_heatmap(comp_mat_feat, self.num_joints, side_out, side_out)

                comp_spec_mat = mat_utils.decode(comp_heat_mat, self.side_in).permute(0, 2, 1)

                comp_spec_mat = self.comp_linear(comp_spec_mat).permute(0, 2, 1)

                comp_mat_loss = self.criterion(comp_spec_mat * comp_valid_mask, comp_true_mat * comp_valid_mask)

                loss += self.comp_loss_weight * comp_mat_loss

            self.optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(self.get_params(), self.grad_norm)
            self.optimizer.step()

            cam_loss_avg += cam_loss.item() * batch
            mat_loss_avg += mat_loss.item() * batch
            total += batch

            print "| train Epoch[%d] [%d/%d]  Cam Loss: %1.4f  Mat Loss: %1.4f" % (epoch, i, n_batches, cam_loss.item(), mat_loss.item())

        cam_loss_avg /= total
        mat_loss_avg /= total

        print "\n=> train Epoch[%d]  Cam Loss: %1.4f  Mat Loss: %1.4f\n" % (epoch, cam_loss_avg, mat_loss_avg)

        return dict(cam_train_loss = cam_loss_avg, mat_train_loss = mat_loss_avg)


    def cam_train(self, epoch, data_loader, cuda_device):
        n_batches = len(data_loader)

        loss_avg = 0
        total = 0

        side_out = (self.side_in - 1) / self.stride + 1

        for i, (image, true_cam, valid_mask) in enumerate(data_loader):

            if self.n_cudas:
                image = image.to(cuda_device)

                true_cam = true_cam.to(cuda_device)

                valid_mask = valid_mask.unsqueeze(-1).to(cuda_device)

            batch = image.size(0)

            cam_feat = self.model(image)

            heat_cam = utils.to_heatmap(cam_feat, self.depth, self.num_joints, side_out, side_out)

            key_index = self.data_info.key_index

            relat_cam = utils.decode(heat_cam, self.depth_range)

            relat_cam = relat_cam - relat_cam[:, key_index:key_index + 1]

            spec_cam = relat_cam + true_cam[:, key_index:key_index + 1]

            loss = self.criterion(spec_cam * mask_valid, true_cam * mask_valid)

            self.optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            self.optimizer.step()

            loss_avg += loss.item() * batch
            total += batch
            
            print "| train Epoch[%d] [%d/%d]  Loss %1.4f" % (epoch, i, n_batches, loss.item())

        return loss_avg / total


    def train(self, epoch, data_loader, comp_loader):
        self.model.train()
        self.adapt_learn_rate(epoch)

        if self.joint_space:
            return self.joint_train(epoch, data_loader, comp_loader, torch.device('cuda'))
        else:
            return self.cam_train(epoch, data_loader, torch.device('cuda'))


    def joint_test(self, epoch, test_loader, cuda_device, do_track = False):
        n_batches = len(test_loader)

        cam_loss_avg = 0
        mat_loss_avg = 0
        total = 0

        side_out = (self.side_in - 1) / self.stride + 1

        mat_stats = []
        cam_stats = []
        det_stats = []

        for i, (image, true_cam, true_mat, back_rotation, valid_mask, intrinsics) in enumerate(test_loader):

            if self.n_cudas:
                image = image.to(cuda_device)

                true_cam = true_cam.to(cuda_device)

                true_mat = true_mat.to(cuda_device)

                mask_valid = valid_mask.unsqueeze(-1).to(cuda_device)

            batch = image.size(0)

            with torch.no_grad():
                cam_feat, mat_feat = self.model(image)

                heat_mat = mat_utils.to_heatmap(mat_feat, self.num_joints, side_out, side_out)

                heat_cam = utils.to_heatmap(cam_feat, self.depth, self.num_joints, side_out, side_out)

                if self.flip_test:

                    _cam_feat, _mat_feat = self.model(image[:, :, :, ::-1])

                    _heat_mat = mat_utils.to_heatmap(_mat_feat, self.num_joints, side_out, side_out)

                    _heat_mat = _heat_mat[:, self.data_info.mirror, :, ::-1]

                    heat_mat = 0.5 * (heat_mat + _heat_mat)

                    _heat_cam = utils.to_heatmap(_cam_feat, self.depth, self.num_joints, side_out, side_out)

                    _heat_cam = _heat_cam[:, self.data_info.mirror, :, ::-1]

                    heat_cam = 0.5 * (heat_cam + _heat_cam)

                spec_mat = mat_utils.decode(heat_mat, self.side_in)

                key_index = self.data_info.key_index

                relat_cam = utils.decode(heat_cam, self.depth_range)

                relat_cam = relat_cam - relat_cam[:, key_index:key_index + 1]

                spec_cam = relat_cam + true_cam[:, key_index:key_index + 1]

                cam_loss = self.criterion(spec_cam * mask_valid, true_cam * mask_valid)
                mat_loss = self.criterion(spec_mat * mask_valid, true_mat * mask_valid)

                loss = cam_loss + mat_loss

                cam_loss_avg += cam_loss.item() * batch
                mat_loss_avg += mat_loss.item() * batch
                total += batch

            print "| test Epoch[%d] [%d/%d]  Cam Loss: %1.4f  Mat Loss: %1.4f" % (epoch, i, n_batches, cam_loss.item(), mat_loss.item())

            valid_mask = valid_mask.numpy()

            spec_mat = spec_mat.cpu().numpy()
            true_mat = true_mat.cpu().numpy()

            mat_stats.append(mat_utils.analyze(spec_mat, true_mat, valid_mask, self.side_in))

            spec_cam = spec_cam.cpu().numpy()
            true_cam = true_cam.cpu().numpy()

            spec_cam = np.einsum('Bij,BCj->BCi', back_rotation, spec_cam)
            true_cam = np.einsum('Bij,BCj->BCi', back_rotation, true_cam)

            cam_stats.append(utils.analyze(spec_cam, true_cam, valid_mask, self.data_info.mirror, self.thresh))

            if do_track:

                relat_cam = relat_cam.cpu().numpy()

                deter_cam = utils.get_deter_cam(spec_mat, relat_cam, valid_mask, intrinsics)

                deter_cam = np.einsum('Bij,BCj->BCi', back_rotation, deter_cam)

                det_stats.append(utils.analyze(deter_cam, true_cam, valid_mask, self.data_info.mirror, self.thresh))

        cam_loss_avg /= total
        mat_loss_avg /= total

        record = dict(cam_test_loss = cam_loss_avg, mat_test_loss = mat_loss_avg)

        record.update(mat_utils.parse_epoch(mat_stats, total))
        record.update(utils.parse_epoch(cam_stats, total))

        print '\n=> test Epoch[%d]  Cam Loss: %1.4f  Mat Loss: %1.4f' % (epoch, cam_loss_avg, mat_loss_avg)

        print 'cam_mean: %1.3f  [pck]: %1.3f  [auc]: %1.3f' % (record['cam_mean'], record['score_pck'], record['score_auc'])

        print 'mat_mean: %1.3f  [oks]: %1.3f\n' % (record['mat_mean'], record['score_oks'])

        if do_track:

            track_rec = utils.parse_epoch(det_stats, total)

            print 'cam_mean: %1.3f  [pck]: %1.3f  [auc]: %1.3f' % (track_rec['cam_mean'], track_rec['score_pck'], track_rec['score_auc'])

        return record


    def cam_test(self, epoch, test_loader, cuda_device):
        n_batches = len(test_loader)

        loss_avg = 0
        total = 0

        side_out = (self.side_in - 1) / self.stride + 1

        cam_stats = []

        for i, (image, true_cam, back_rotation, valid_mask) in enumerate(test_loader):

            if self.n_cudas:
                image = image.to(cuda_device)

                true_cam = true_cam.to(cuda_device)

                mask_valid = valid_mask.unsqueeze(-1).to(cuda_device)

            batch = image.size(0)

            with torch.no_grad():
                cam_feat = self.model(image)

                heat_cam = utils.to_heatmap(cam_feat, self.depth, self.num_joints, side_out, side_out)

                if self.flip_test:
                    _cam_feat = self.model(image[:, :, :, ::-1])

                    _heat_cam = utils.to_heatmap(_cam_feat, self.depth, self.num_joints, side_out, side_out)
                    _heat_cam = _heat_cam[:, self.data_info.mirror, :, ::-1]

                    heat_cam = 0.5 * (heat_cam + _heat_cam)

                key_index = self.data_info.key_index

                relat_cam = utils.decode(heat_cam, self.depth_range)

                relat_cam = relat_cam - relat_cam[:, key_index:key_index + 1]

                spec_cam = relat_cam + true_cam[:, key_index:key_index + 1]

                loss = self.criterion(spec_cam * mask_valid, true_cam * mask_valid)

                loss_avg += loss.item() * batch
                total += batch

            valid_mask = valid_mask.numpy()

            spec_cam = spec_cam.cpu().numpy()
            true_cam = true_cam.cpu().numpy()

            spec_cam = np.einsum('Bij,BCj->BCi', back_rotation, spec_cam)
            true_cam = np.einsum('Bij,BCj->BCi', back_rotation, true_cam)

            cam_stats.append(utils.analyze(spec_cam, true_cam, valid_mask, self.data_info.mirror, self.thresh))

            print "| test Epoch[%d] [%d/%d]  Cam Loss %1.4f" % (epoch, i, n_batches, loss.item())

        loss_avg /= total

        record = dict(test_loss = loss_avg)
        record.update(utils.parse_epoch(cam_stats, total))

        print '\n=> test Epoch[%d]  Loss: %1.4f  cam_mean: %1.3f' % (epoch, loss_avg, record['cam_mean'])
        print '[pck]: %1.3f  [auc]: %1.3f' % (record['score_pck'], record['score_auc'])

        return record


    def test(self, epoch, test_loader, do_track):
        self.model.eval()

        if self.joint_space:
            return self.joint_test(epoch, test_loader, torch.device('cuda'), do_track)
        else:
            return self.cam_test(epoch, test_loader, torch.device('cuda'))


    def adapt_learn_rate(self, epoch):
        if epoch - 1 < self.num_epochs * 0.6:
            learn_rate = self.learn_rate
        elif epoch - 1 < self.num_epochs * 0.9:
            learn_rate = self.learn_rate * 0.2
        else:
            learn_rate = self.learn_rate * 0.04

        for group in self.optimizer.param_groups:
            group['lr'] = learn_rate


    def get_params(self):
        for params in self.list_params:
            for param in params:
                yield param
