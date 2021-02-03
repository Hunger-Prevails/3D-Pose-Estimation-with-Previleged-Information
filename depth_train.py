import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import utils

from torch.autograd import Variable


def wrap_by_name(names, params):
    param_convs = [param for name, param in zip(names, params) if 'bn' in name]
    param_bns = [param for name, param in zip(names, params) if 'bn' not in name]
    return [dict(params = param_convs), dict(params = param_bns)]


class Trainer:

    def __init__(self, args, model, data_info):

        assert args.half_acc <= args.n_cudas

        self.model = model
        self.data_info = data_info

        self.list_params = [param for name, param in model.named_parameters()]
        self.list_names = [name for name, param in model.named_parameters()]

        self.half_acc = args.half_acc
        self.depth_only = args.depth_only
        self.do_fusion = args.do_fusion
        self.do_distill = args.do_distill
        self.partial_conv = args.partial_conv

        if args.half_acc:
            self.copy_params = [param.clone().detach() for param in self.list_params]
            self.model = self.model.half()

            for param in self.copy_params:
                param.requires_grad = True
                param.grad = param.data.new_zeros(param.size())

            self.optimizer = optim.Adam(wrap_by_name(self.list_names, self.copy_params), args.learn_rate, weight_decay = args.weight_decay)
        else:
            self.optimizer = optim.Adam(wrap_by_name(self.list_names, self.list_params), args.learn_rate, weight_decay = args.weight_decay)

        self.n_cudas = args.n_cudas
        self.depth = args.depth
        self.num_joints = args.num_joints
        self.side_in = args.side_in
        self.stride = args.stride
        self.depth_range = args.depth_range

        self.learn_rate = args.learn_rate
        self.num_epochs = args.n_epochs
        self.grad_norm = args.grad_norm
        self.grad_scaling = args.grad_scaling
        self.loss_div = args.loss_div

        self.thresh = dict(
            solid = args.thresh_solid * args.loss_div,
            close = args.thresh_close * args.loss_div,
            rough = args.thresh_rough * args.loss_div
        )
        self.criterion = nn.__dict__[args.criterion + 'Loss'](reduction = 'mean')

        if args.n_cudas:
            self.criterion = self.criterion.cuda()


    def set_teacher(self, teacher):
        self.teacher = teacher.half() if self.half_acc else teacher


    def distill_train(self, epoch, data_loader, cuda_device):
        n_batches = len(data_loader)

        dist_loss_sum = 0.0
        cam_loss_sum = 0.0
        total = 0

        side_out = (self.side_in - 1) // self.stride + 1

        for i, (color_image, depth_image, true_cam, valid_mask) in enumerate(data_loader):

            if self.n_cudas:
                color_image = color_image.half().to(cuda_device) if self.half_acc else color_image.to(cuda_device)
                depth_image = depth_image.half().to(cuda_device) if self.half_acc else depth_image.to(cuda_device)

                true_cam = true_cam.to(cuda_device)
                valid_mask = valid_mask.to(cuda_device)

            batch = true_cam.size(0)

            with torch.no_grad():
                teach_cam, teach_last = self.teacher(color_image, depth_image)
                if self.half_acc:
                    teach_last = teach_last.float()

            cam_feat, last_feat = self.model(color_image)
            if self.half_acc:
                cam_feat = cam_feat.float()
                last_feat = last_feat.float()

            dist_loss = torch.linalg.norm((teach_last - last_feat).view(batch, -1), dim = -1).mean()

            heat_cam = utils.to_heatmap(cam_feat, self.depth, self.num_joints, side_out, side_out)

            key_index = self.data_info.key_index

            relat_cam = utils.decode(heat_cam, self.depth_range)

            relat_cam = relat_cam - relat_cam[:, key_index:key_index + 1]

            spec_cam = relat_cam + true_cam[:, key_index:key_index + 1]

            cam_loss = self.criterion(spec_cam.view(-1, 3)[valid_mask.view(-1)] / self.loss_div, true_cam.view(-1, 3)[valid_mask.view(-1)] / self.loss_div)

            loss = dist_loss + cam_loss

            print('| train Epoch[%d] [%d/%d]  Dist Loss %1.4f  Cam Loss %1.4f' % (epoch, i, n_batches, dist_loss.item(), cam_loss.item()))

            dist_loss_sum += dist_loss.item() * batch
            cam_loss_sum += cam_loss.item() * batch
            total += batch

            if self.half_acc:
                loss *= self.grad_scaling

                for h_param in self.list_params:

                    if h_param.grad is None:
                        continue

                    h_param.grad.detach_()
                    h_param.grad.zero_()

                loss.backward()

                self.optimizer.zero_grad()

                do_update = True

                for c_param, h_param in zip(self.copy_params, self.list_params):

                    if h_param.grad is None:
                        continue

                    if torch.any(torch.isinf(h_param.grad)):
                        do_update = False
                        print('update step skipped')
                        break

                    c_param.grad.copy_(h_param.grad)
                    c_param.grad /= self.grad_scaling

                if do_update:
                    nn.utils.clip_grad_norm_(self.copy_params, self.grad_norm)

                    self.optimizer.step()

                    for c_param, h_param in zip(self.copy_params, self.list_params):
                        h_param.data.copy_(c_param.data)

            else:
                self.optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(self.list_params, self.grad_norm)
                self.optimizer.step()

        dist_loss_sum /= total
        cam_loss_sum /= total

        print('\n=> train Epoch[%d]  Dist Loss: %1.4f\n  Cam Loss: %1.4f\n' % (epoch, dist_loss_sum, cam_loss_sum))

        return dict(dist_train_loss = dist_loss_sum, cam_train_loss = cam_loss_sum)


    def fusion_train(self, epoch, data_loader, cuda_device):
        n_batches = len(data_loader)

        loss_avg = 0
        total = 0

        side_out = (self.side_in - 1) // self.stride + 1

        for i, (color_image, depth_image, true_cam, valid_mask) in enumerate(data_loader):

            if self.n_cudas:
                color_image = color_image.half().to(cuda_device) if self.half_acc else color_image.to(cuda_device)
                depth_image = depth_image.half().to(cuda_device) if self.half_acc else depth_image.to(cuda_device)

                true_cam = true_cam.to(cuda_device)
                valid_mask = valid_mask.to(cuda_device)

            batch = true_cam.size(0)

            cam_feat = self.model(color_image, depth_image).float() if self.half_acc else self.model(color_image, depth_image)

            heat_cam = utils.to_heatmap(cam_feat, self.depth, self.num_joints, side_out, side_out)

            key_index = self.data_info.key_index

            relat_cam = utils.decode(heat_cam, self.depth_range)

            relat_cam = relat_cam - relat_cam[:, key_index:key_index + 1]

            spec_cam = relat_cam + true_cam[:, key_index:key_index + 1]

            loss = self.criterion(spec_cam.view(-1, 3)[valid_mask.view(-1)] / self.loss_div, true_cam.view(-1, 3)[valid_mask.view(-1)] / self.loss_div)

            print('| train Epoch[%d] [%d/%d]  Loss %1.4f' % (epoch, i, n_batches, loss.item()))

            loss_avg += loss.item() * batch

            total += batch

            if self.half_acc:
                loss *= self.grad_scaling

                for h_param in self.list_params:

                    if h_param.grad is None:
                        continue

                    h_param.grad.detach_()
                    h_param.grad.zero_()

                loss.backward()

                self.optimizer.zero_grad()

                do_update = True

                for c_param, h_param in zip(self.copy_params, self.list_params):

                    if h_param.grad is None:
                        continue

                    if torch.any(torch.isinf(h_param.grad)):
                        do_update = False
                        print('update step skipped')
                        break

                    c_param.grad.copy_(h_param.grad)
                    c_param.grad /= self.grad_scaling

                if do_update:
                    nn.utils.clip_grad_norm_(self.copy_params, self.grad_norm)

                    self.optimizer.step()

                    for c_param, h_param in zip(self.copy_params, self.list_params):
                        h_param.data.copy_(c_param.data)

            else:
                self.optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(self.list_params, self.grad_norm)
                self.optimizer.step()

        loss_avg /= total

        print('\n=> train Epoch[%d]  Cam Loss: %1.4f\n' % (epoch, loss_avg))

        return dict(cam_train_loss = loss_avg)


    def vanilla_train(self, epoch, data_loader, cuda_device):
        n_batches = len(data_loader)

        loss_avg = 0
        total = 0

        side_out = (self.side_in - 1) // self.stride + 1

        for i, (color_image, depth_image, true_cam, valid_mask) in enumerate(data_loader):

            in_image = depth_image if self.depth_only else color_image

            if self.n_cudas:
                in_image = in_image.half().to(cuda_device) if self.half_acc else in_image.to(cuda_device)

                true_cam = true_cam.to(cuda_device)

                valid_mask = valid_mask.to(cuda_device)

            batch = true_cam.size(0)

            cam_feat = self.model(in_image).float() if self.half_acc else self.model(in_image)

            heat_cam = utils.to_heatmap(cam_feat, self.depth, self.num_joints, side_out, side_out)

            key_index = self.data_info.key_index

            relat_cam = utils.decode(heat_cam, self.depth_range)

            relat_cam = relat_cam - relat_cam[:, key_index:key_index + 1]

            spec_cam = relat_cam + true_cam[:, key_index:key_index + 1]

            loss = self.criterion(spec_cam.view(-1, 3)[valid_mask.view(-1)] / self.loss_div, true_cam.view(-1, 3)[valid_mask.view(-1)] / self.loss_div)

            print('| train Epoch[%d] [%d/%d]  Loss %1.4f' % (epoch, i, n_batches, loss.item()), flush = True)

            loss_avg += loss.item() * batch

            total += batch

            if self.half_acc:
                loss *= self.grad_scaling

                for h_param in self.list_params:

                    if h_param.grad is None:
                        continue

                    h_param.grad.detach_()
                    h_param.grad.zero_()

                loss.backward()

                self.optimizer.zero_grad()

                do_update = True

                for c_param, h_param in zip(self.copy_params, self.list_params):

                    if h_param.grad is None:
                        continue

                    if torch.any(torch.isinf(h_param.grad)):
                        do_update = False
                        print('update step skipped')
                        break

                    c_param.grad.copy_(h_param.grad)
                    c_param.grad /= self.grad_scaling

                if do_update:
                    nn.utils.clip_grad_norm_(self.copy_params, self.grad_norm)

                    self.optimizer.step()

                    for c_param, h_param in zip(self.copy_params, self.list_params):
                        h_param.data.copy_(c_param.data)

            else:
                self.optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(self.list_params, self.grad_norm)
                self.optimizer.step()

        loss_avg /= total

        print('\n=> train Epoch[%d]  Cam Loss: %1.4f\n' % (epoch, loss_avg))

        return dict(cam_train_loss = loss_avg)


    def train(self, epoch, data_loader):
        self.model.train()
        self.adapt_learn_rate(epoch)

        if self.do_distill:
            return self.distill_train(epoch, data_loader, torch.device('cuda'))
        elif self.do_fusion:
            return self.fusion_train(epoch, data_loader, torch.device('cuda'))
        else:
            return self.vanilla_train(epoch, data_loader, torch.device('cuda'))


    def fusion_test(self, epoch, test_loader, cuda_device):
        n_batches = len(test_loader)

        loss_avg = 0
        total = 0

        side_out = (self.side_in - 1) // self.stride + 1

        cam_stats = []

        for i, (color_image, depth_image, true_cam, valid_mask, color_br) in enumerate(test_loader):

            if self.n_cudas:
                color_image = color_image.half().to(cuda_device) if self.half_acc else color_image.to(cuda_device)
                depth_image = depth_image.half().to(cuda_device) if self.half_acc else depth_image.to(cuda_device)

                true_cam = true_cam.to(cuda_device)
                valid_mask = valid_mask.to(cuda_device)

            batch = true_cam.size(0)

            with torch.no_grad():
                cam_feat = self.model(color_image, depth_image).float() if self.half_acc else self.model(color_image, depth_image)

                heat_cam = utils.to_heatmap(cam_feat, self.depth, self.num_joints, side_out, side_out)

                key_index = self.data_info.key_index

                relat_cam = utils.decode(heat_cam, self.depth_range)

                relat_cam = relat_cam - relat_cam[:, key_index:key_index + 1]

                spec_cam = relat_cam + true_cam[:, key_index:key_index + 1]

                loss = self.criterion(spec_cam.view(-1, 3)[valid_mask.view(-1)] / self.loss_div, true_cam.view(-1, 3)[valid_mask.view(-1)] / self.loss_div)

            loss_avg += loss.item() * batch

            total += batch

            valid_mask = valid_mask.cpu().numpy().astype(np.bool)

            spec_cam = spec_cam.cpu().numpy()
            true_cam = true_cam.cpu().numpy()

            spec_cam = np.einsum('Bij,BCj->BCi', color_br, spec_cam)
            true_cam = np.einsum('Bij,BCj->BCi', color_br, true_cam)

            cam_stats.append(utils.analyze(spec_cam, true_cam, valid_mask, self.data_info.mirror, self.thresh))

            print('| test Epoch[%d] [%d/%d]  Cam Loss %1.4f' % (epoch, i, n_batches, loss.item()))

        loss_avg /= total

        record = dict(test_loss = loss_avg)
        record.update(utils.parse_epoch(cam_stats))

        print('\n=> test Epoch[%d]  Cam Loss: %1.4f\n' % (epoch, loss_avg))

        print('=>[SPEC] cam_mean: %1.3f  [pck]: %1.3f  [auc]: %1.3f\n' % (record['cam_mean'], record['score_pck'], record['score_auc']))

        return record


    def vanilla_test(self, epoch, test_loader, cuda_device):
        n_batches = len(test_loader)

        loss_avg = 0
        total = 0

        side_out = (self.side_in - 1) // self.stride + 1

        cam_stats = []

        for i, (color_image, depth_image, true_cam, valid_mask, color_br) in enumerate(test_loader):

            in_image = depth_image if self.depth_only else color_image

            if self.n_cudas:
                in_image = in_image.half().to(cuda_device) if self.half_acc else in_image.to(cuda_device)

                true_cam = true_cam.to(cuda_device)

                valid_mask = valid_mask.to(cuda_device)

            batch = true_cam.size(0)

            with torch.no_grad():
                cam_feat = self.model(in_image).float() if self.half_acc else self.model(in_image)

                heat_cam = utils.to_heatmap(cam_feat, self.depth, self.num_joints, side_out, side_out)

                key_index = self.data_info.key_index

                relat_cam = utils.decode(heat_cam, self.depth_range)

                relat_cam = relat_cam - relat_cam[:, key_index:key_index + 1]

                spec_cam = relat_cam + true_cam[:, key_index:key_index + 1]

                loss = self.criterion(spec_cam.view(-1, 3)[valid_mask.view(-1)] / self.loss_div, true_cam.view(-1, 3)[valid_mask.view(-1)] / self.loss_div)

            loss_avg += loss.item() * batch

            total += batch

            valid_mask = valid_mask.cpu().numpy().astype(np.bool)

            spec_cam = spec_cam.cpu().numpy()
            true_cam = true_cam.cpu().numpy()

            spec_cam = np.einsum('Bij,BCj->BCi', color_br, spec_cam)
            true_cam = np.einsum('Bij,BCj->BCi', color_br, true_cam)

            cam_stats.append(utils.analyze(spec_cam, true_cam, valid_mask, self.data_info.mirror, self.thresh))

            print('| test Epoch[%d] [%d/%d]  Cam Loss %1.4f' % (epoch, i, n_batches, loss.item()))

        loss_avg /= total

        record = dict(test_loss = loss_avg)
        record.update(utils.parse_epoch(cam_stats))

        print('\n=> test Epoch[%d]  Cam Loss: %1.4f\n' % (epoch, loss_avg))

        print('=>[SPEC] cam_mean: %1.3f  [pck]: %1.3f  [auc]: %1.3f\n' % (record['cam_mean'], record['score_pck'], record['score_auc']))

        return record


    def test(self, epoch, test_loader):
        self.model.eval()

        if self.do_fusion:
            return self.fusion_test(epoch, test_loader, torch.device('cuda'))
        else:
            return self.vanilla_test(epoch, test_loader, torch.device('cuda'))


    def adapt_learn_rate(self, epoch):
        if epoch - 1 < self.num_epochs * 0.6:
            learn_rate = self.learn_rate
            learn_rate_bn = self.learn_rate

        elif epoch - 1 < self.num_epochs * 0.8:
            learn_rate = self.learn_rate * 0.2
            learn_rate_bn = self.learn_rate * 0.1 if self.partial_conv else learn_rate

        else:
            learn_rate = self.learn_rate * 0.04
            learn_rate_bn = self.learn_rate * 0.01 if self.partial_conv else learn_rate

        if epoch == 1:
            learn_rate /= 2
            learn_rate_bn /= 2

        self.optimizer.param_groups[0]['lr'] = learn_rate
        self.optimizer.param_groups[1]['lr'] = learn_rate_bn
