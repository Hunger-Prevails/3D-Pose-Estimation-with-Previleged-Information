import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
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
        self.unimodal = args.unimodal

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


    def train(self, epoch, train_loader):
        n_batches = len(train_loader)

        loss_avg = 0
        total = 0
        
        self.model.train()
        self.learning_rate(epoch)

        cuda_device = torch.device('cuda')

        for i, (image, true_cam, intrinsics, valid_mask) in enumerate(train_loader):
            
            if self.nGPU > 0:
                image = image.to(cuda_device)

                true_cam = true_cam.to(cuda_device)

                intrinsics = intrinsics.to(cuda_device)

                valid_mask = valid_mask.unsqueeze(-1).to(cuda_device)
                
            batch_size = image.size(0)
            
            cam_feat = self.model(image)

            heatmap = utils.to_heatmap(cam_feat, self.depth, self.num_joints, self.side_ausgabe, self.side_ausgabe)

            key_index = self.joint_info.key_index

            spec_cam = utils.decode(heatmap, self.depth_range)

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

        loss_avg /= total

        print "\n=> train Epoch[%d]  Loss: %1.4f\n" % (epoch, loss_avg)

        return dict(train_loss = loss_avg)
        

    def test(self, epoch, test_loader):
        n_batches = len(test_loader)

        loss_avg = 0
        total = 0

        self.model.eval()
        self.learning_rate(epoch)

        cuda_device = torch.device('cuda')

        scores_and_stats = []

        for i, (image, true_cam, intrinsics, back_rotation, valid_mask) in enumerate(test_loader):
            
            if self.nGPU > 0:
                image = image.to(cuda_device)

                true_cam = true_cam.to(cuda_device)

                intrinsics = intrinsics.to(cuda_device)

                mask_valid = valid_mask.unsqueeze(-1).to(cuda_device)

            batch_size = image.size(0)

            with torch.no_grad():                
                cam_feat = self.model(image)

                heatmap = utils.to_heatmap(cam_feat, self.depth, self.num_joints, self.side_ausgabe, self.side_ausgabe, self.unimodal)

                if self.flip_test:
                    ausgabe_flip = self.model(image[:, :, :, ::-1])

                    heatmap_flip = utils.to_heatmap(cam_feat, self.depth, self.num_joints, self.side_ausgabe, self.side_ausgabe, self.unimodal)
                    heatmap_flip = heatmap_flip[:, self.joint_info.mirror, :, ::-1]

                    heatmap = 0.5 * (heatmap + heatmap_flip)

                key_index = self.joint_info.key_index

                spec_cam = utils.decode(heatmap, self.depth_range)

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

            scores_and_stats.append(self.analyze(spec_cam, true_cam, valid_mask, self.joint_info.mirror, key_index))

            print "| test Epoch[%d] [%d/%d]  Loss %1.4f" % (epoch, i, n_batches, loss.item())

        loss_avg /= total

        summary = dict(test_loss = loss_avg)
        summary.update(utils.parse_epoch(scores_and_stats, total))

        print '\n=> test Epoch[%d]  Loss: %1.4f  overall_mean: %1.3f' % (epoch, loss_avg, summary['overall_mean'])
        print 'pck: %1.3f  auc: %1.3f\n' % (summary['score_pck'], summary['score_auc'])

        return summary


    def analyze(self, spec_cam, true_cam, valid_mask, mirror, key_index):
        '''
        Analyzes spec_cam against true_cam under original camera

        Args:
            spec_cam: (batch_size, num_joints, 3)
            true_cam: (batch_size, num_joints, 3)
            valid_mask: (batch_size, num_joints)
            mirror: (num_joints,)

        Returns:
            batch_size, scores and statistics

        '''
        spec_cam -= spec_cam[:, key_index:key_index + 1]
        true_cam -= true_cam[:, key_index:key_index + 1]

        cubics = np.linalg.norm(spec_cam - true_cam, axis = -1)
        reflects = np.linalg.norm(spec_cam - true_cam[:, mirror], axis = -1)
        tangents = np.linalg.norm(spec_cam[:, :, :2] - true_cam[:, :, :2], axis = -1)

        valid = np.where(valid_mask.flatten() == 1.0)[0]

        cubics = cubics.flatten()[valid]
        reflects = reflects.flatten()[valid]
        tangents = tangents.flatten()[valid]

        overall_mean = np.mean(cubics)
        score_pck = np.mean(cubics / self.thresholds['score'] <= 1.0)
        score_auc = np.mean(np.maximum(0, 1 - cubics / self.thresholds['score']))

        stats = utils.statistics(cubics, reflects, tangents, self.thresholds)

        stats.update(dict(
                        batch_size = spec_cam.shape[0],
                        score_pck = score_pck,
                        score_auc = score_auc,
                        overall_mean = overall_mean))

        return stats


    def learning_rate(self, epoch):
        if epoch - 1 < self.num_epochs * 0.6:
            learn_rate = self.learn_rate
        elif epoch - 1 < self.num_epochs * 0.9:
            learn_rate = self.learn_rate * 0.2
        else:
            learn_rate = self.learn_rate * 0.04

        for group in self.optimizer.param_groups:
            group['lr'] = learn_rate
