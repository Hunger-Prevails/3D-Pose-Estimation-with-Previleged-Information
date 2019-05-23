import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import mat_utils
import utils

from torch.autograd import Variable

class Trainer:

    def __init__(self, args, model, joint_info):
        self.model = model
        self.joint_info = joint_info

        self.nGPU = args.nGPU
        self.depth = args.depth
        self.num_joints = args.num_joints
        self.side_eingabe = args.side_eingabe
        self.side_ausgabe = args.side_ausgabe
        self.depth_range = args.depth_range
        self.flip_test = args.flip_test
        self.joint_space = args.joint_space

        self.learn_rate = args.learn_rate
        self.num_epochs = args.n_epochs
        self.grad_norm = args.grad_norm

        self.thresholds = dict(
                            score = args.score_thresh,
                            perfect = args.perfect_thresh,
                            good = args.good_thresh,
                            jitter = args.jitter_thresh)

        self.optimizer = optim.Adam(model.parameters(), args.learn_rate, weight_decay = args.weight_decay)

        self.criterion = nn.__dict__[args.criterion + 'Loss'](reduction = 'mean')
        if args.nGPU > 0:
            self.criterion = self.criterion.cuda()


    def joint_train(self, epoch, train_loader, cuda_device):
        n_batches = len(train_loader)

        cam_loss_avg = 0
        mat_loss_avg = 0
        total = 0

        for i, (image, true_cam, true_mat, valid_mask) in enumerate(train_loader):

            if self.nGPU > 0:
                image = image.to(cuda_device)

                true_cam = true_cam.to(cuda_device)

                true_mat = true_mat.to(cuda_device)

                valid_mask = valid_mask.unsqueeze(-1).to(cuda_device)

            batch_size = image.size(0)

            cam_feat, mat_feat = self.model(image)

            heat_mat = mat_utils.to_heatmap(mat_feat, self.num_joints, self.side_ausgabe, self.side_ausgabe)

            heat_cam = utils.to_heatmap(cam_feat, self.depth, self.num_joints, self.side_ausgabe, self.side_ausgabe)

            spec_mat = mat_utils.decode(heat_mat, self.side_eingabe)

            key_index = self.joint_info.key_index

            spec_cam = utils.decode(heat_cam, self.depth_range)

            relative = spec_cam - spec_cam[:, key_index:key_index + 1]

            true_relative = true_cam - true_cam[:, key_index:key_index + 1]

            cam_loss = self.criterion(relative * valid_mask, true_relative * valid_mask)
            mat_loss = self.criterion(spec_mat * valid_mask, true_mat * valid_mask)

            loss = cam_loss + mat_loss

            self.optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            self.optimizer.step()

            cam_loss_avg += cam_loss.item() * batch_size
            mat_loss_avg += mat_loss.item() * batch_size
            total += batch_size

            print "| train Epoch[%d] [%d/%d]  Cam Loss: %1.4f  Mat Loss: %1.4f" % (epoch, i, n_batches, cam_loss.item(), mat_loss.item())

        cam_loss_avg /= total
        mat_loss_avg /= total

        print "\n=> train Epoch[%d]  Cam Loss: %1.4f  Mat Loss: %1.4f\n" % (epoch, cam_loss_avg, mat_loss_avg)

        return dict(cam_train_loss = cam_loss_avg, mat_train_loss = mat_loss_avg)


    def cam_train(self, epoch, train_loader, cuda_device):
        n_batches = len(train_loader)

        loss_avg = 0
        total = 0

        for i, (image, true_cam, valid_mask) in enumerate(train_loader):

            if self.nGPU > 0:
                image = image.to(cuda_device)

                true_cam = true_cam.to(cuda_device)

                valid_mask = valid_mask.unsqueeze(-1).to(cuda_device)

            batch_size = image.size(0)

            cam_feat = self.model(image)

            heat_cam = utils.to_heatmap(cam_feat, self.depth, self.num_joints, self.side_ausgabe, self.side_ausgabe)

            key_index = self.joint_info.key_index

            spec_cam = utils.decode(heat_cam, self.depth_range)

            relative = spec_cam - spec_cam[:, key_index:key_index + 1]

            true_relative = true_cam - true_cam[:, key_index:key_index + 1]

            loss = self.criterion(relative * valid_mask, true_relative * valid_mask)

            self.optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            self.optimizer.step()

            loss_avg += loss.item() * batch_size
            total += batch_size
            
            print "| train Epoch[%d] [%d/%d]  Loss %1.4f" % (epoch, i, n_batches, loss.item())

        return loss_avg / total


    def train(self, epoch, train_loader):
        self.model.train()
        self.adapt_learn_rate(epoch)

        if self.joint_space:
            return self.joint_train(epoch, train_loader, torch.device('cuda'))
        else:
            return self.cam_train(epoch, train_loader, torch.device('cuda'))


    def joint_test(self, epoch, test_loader, cuda_device):
        n_batches = len(test_loader)

        cam_loss_avg = 0
        mat_loss_avg = 0
        total = 0

        cam_stats = []
        mat_stats = []

        for i, (image, true_cam, true_mat, back_rotation, valid_mask) in enumerate(test_loader):

            if self.nGPU > 0:
                image = image.to(cuda_device)

                true_cam = true_cam.to(cuda_device)

                true_mat = true_mat.to(cuda_device)

                mask_valid = valid_mask.unsqueeze(-1).to(cuda_device)

            batch_size = image.size(0)

            with torch.no_grad():
                cam_feat, mat_feat = self.model(image)

                heat_mat = mat_utils.to_heatmap(mat_feat, self.num_joints, self.side_ausgabe, self.side_ausgabe)

                heat_cam = utils.to_heatmap(cam_feat, self.depth, self.num_joints, self.side_ausgabe, self.side_ausgabe)

                if self.flip_test:
                    _cam_feat, _mat_feat = self.model(image[:, :, :, ::-1])

                    _heat_mat = mat_utils.to_heatmap(_mat_feat, self.num_joints, self.side_ausgabe, self.side_ausgabe)

                    _heat_mat = _heat_mat[:, self.joint_info.mirror, :, ::-1]

                    heat_mat = 0.5 * (heat_mat + _heat_mat)

                    _heat_cam = utils.to_heatmap(_cam_feat, self.depth, self.num_joints, self.side_ausgabe, self.side_ausgabe)

                    _heat_cam = _heat_cam[:, self.joint_info.mirror, :, ::-1]

                    heat_cam = 0.5 * (heat_cam + _heat_cam)

                spec_mat = mat_utils.decode(heat_mat, self.side_eingabe)

                key_index = self.joint_info.key_index

                spec_cam = utils.decode(heat_cam, self.depth_range)

                relative = spec_cam - spec_cam[:, key_index:key_index + 1]

                true_relative = true_cam - true_cam[:, key_index:key_index + 1]

                cam_loss = self.criterion(relative * mask_valid, true_relative * mask_valid)
                mat_loss = self.criterion(spec_mat * mask_valid, true_mat * mask_valid)

                loss = cam_loss + mat_loss

                cam_loss_avg += cam_loss.item() * batch_size
                mat_loss_avg += mat_loss.item() * batch_size
                total += batch_size

            relative = relative.cpu().numpy()
            true_cam = true_cam.cpu().numpy()

            spec_cam = relative + true_cam[:, key_index:key_index + 1]

            spec_cam = np.einsum('Bij,BCj->BCi', back_rotation, spec_cam)
            true_cam = np.einsum('Bij,BCj->BCi', back_rotation, true_cam)

            spec_mat = spec_mat.cpu().numpy()
            true_mat = true_mat.cpu().numpy()

            valid_mask = valid_mask.numpy()

            cam_stats.append(utils.analyze(spec_cam, true_cam, valid_mask, self.joint_info.mirror, key_index, self.thresholds))
            mat_stats.append(mat_utils.analyze(spec_mat, true_mat, valid_mask))

            print "| test Epoch[%d] [%d/%d]  Cam Loss: %1.4f  Mat Loss: %1.4f" % (epoch, i, n_batches, cam_loss.item(), mat_loss.item())

        cam_loss_avg /= total
        mat_loss_avg /= total

        record = dict(cam_test_loss = cam_loss_avg, mat_test_loss = mat_loss_avg)

        record.update(utils.parse_epoch(cam_stats, total))
        record.update(mat_utils.parse_epoch(mat_stats, total))

        print '\n=> test Epoch[%d]  Cam Loss: %1.4f  Mat Loss: %1.4f' % (epoch, cam_loss_avg, mat_loss_avg)
        print 'cam_mean: %1.3f  pck: %1.3f  auc: %1.3f' % (record['cam_mean'], record['score_pck'], record['score_auc'])
        print 'mat_mean: %1.3f  oks: %1.3f\n' % (record['mat_mean'], record['score_oks'])

        return record


    def cam_test(self, epoch, test_loader, cuda_device):
        n_batches = len(test_loader)

        loss_avg = 0
        total = 0

        cam_stats = []

        for i, (image, true_cam, back_rotation, valid_mask) in enumerate(test_loader):

            if self.nGPU > 0:
                image = image.to(cuda_device)

                true_cam = true_cam.to(cuda_device)

                mask_valid = valid_mask.unsqueeze(-1).to(cuda_device)

            batch_size = image.size(0)

            with torch.no_grad():
                cam_feat = self.model(image)

                heat_cam = utils.to_heatmap(cam_feat, self.depth, self.num_joints, self.side_ausgabe, self.side_ausgabe)

                if self.flip_test:
                    _cam_feat = self.model(image[:, :, :, ::-1])

                    _heat_cam = utils.to_heatmap(_cam_feat, self.depth, self.num_joints, self.side_ausgabe, self.side_ausgabe)
                    _heat_cam = _heat_cam[:, self.joint_info.mirror, :, ::-1]

                    heat_cam = 0.5 * (heat_cam + _heat_cam)

                key_index = self.joint_info.key_index

                spec_cam = utils.decode(heat_cam, self.depth_range)

                relative = spec_cam - spec_cam[:, key_index:key_index + 1]
                
                true_relative = true_cam - true_cam[:, key_index:key_index + 1]

                loss = self.criterion(relative * mask_valid, true_relative * mask_valid)

                loss_avg += loss.item() * batch_size
                total += batch_size

            relative = relative.cpu().numpy()
            true_cam = true_cam.cpu().numpy()

            spec_cam = relative + true_cam[:, key_index:key_index + 1]

            spec_cam = np.einsum('Bij,BCj->BCi', back_rotation, spec_cam)
            true_cam = np.einsum('Bij,BCj->BCi', back_rotation, true_cam)

            cam_stats.append(utils.analyze(spec_cam, true_cam, valid_mask, self.joint_info.mirror, key_index, self.thresholds))

            print "| test Epoch[%d] [%d/%d]  Loss %1.4f" % (epoch, i, n_batches, loss.item())

        loss_avg /= total

        record = dict(test_loss = loss_avg)
        record.update(utils.parse_epoch(cam_stats, total))

        print '\n=> test Epoch[%d]  Loss: %1.4f  cam_mean: %1.3f' % (epoch, loss_avg, record['cam_mean'])
        print 'pck: %1.3f  auc: %1.3f' % (record['score_pck'], record['score_auc'])

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

        for group in self.optimizer.param_groups:
            group['lr'] = learn_rate
