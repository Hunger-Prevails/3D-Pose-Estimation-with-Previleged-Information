import os
import torch
import numpy as np

class Logger:

    def __init__(self, args, state):
        self.state = state if state else dict(best_auc = 0, best_oks = 0, best_recon_auc = 0, best_epoch = 0, epoch = 0)

        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        
        self.save_path = os.path.join(args.save_path, args.model + '-' + args.suffix)
        
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        
        assert args.save_record != args.test_only or args.val_only

        self.do_track = args.do_track
        self.joint_space = args.joint_space
        self.save_record = args.save_record

        record_path = os.path.join(self.save_path, 'train_record.pth')

        self.train_record = torch.load(record_path) if args.resume and os.path.exists(record_path) else None


    def record(self, epoch, train_recs, test_recs, model):

        if torch.typename(model).find('DataParallel') != -1:
            model = model.module

        self.state['epoch'] = epoch
         
        if train_recs:
            model_file = os.path.join(self.save_path, 'model_%d.pth' % epoch);
            
            checkpoint = dict()
            checkpoint['state'] = self.state
            checkpoint['model'] = model.state_dict()
            
            torch.save(checkpoint, model_file)

        if test_recs:
            score_sum = test_recs['score_auc']
            best_sum = self.state['best_auc']

            if self.joint_space:
                score_sum += test_recs['score_oks']
                best_sum += self.state['best_oks']

            if self.do_track:
                score_sum += test_recs['recon_score_auc'] * 2
                best_sum += self.state['best_recon_auc'] * 2

            if score_sum > best_sum:
                self.state['best_epoch'] = epoch
                self.state['best_auc'] = test_recs['score_auc']

                if self.joint_space:
                    self.state['best_oks'] = test_recs['score_oks']

                if self.do_track:
                    self.state['best_recon_auc'] = test_recs['recon_score_auc']
                
                best = os.path.join(self.save_path, 'best.pth')
                torch.save({'best': epoch}, best)

        train_recs.update(test_recs)

        if self.save_record:

            if self.train_record:
                keys = [key for key in train_recs]

                records = [self.train_record[key] + [train_recs[key]] for key in train_recs]

                self.train_record = dict(zip(keys, records))
            else:
                keys = [key for key in train_recs]

                records = [[train_recs[key]] for key in train_recs]

                self.train_record = dict(zip(keys, records))

            torch.save(self.train_record, os.path.join(self.save_path, 'train_record.pth'))

            print('- train record saved to', os.path.join(self.save_path, 'train_record.pth'), '\n')


    def final_print(self):
        message = '- Best:  epoch: %3d  auc: %6.3f' % (self.state['best_epoch'], self.state['best_auc'])

        if self.joint_space:
            message += '  oks: %6.3f' % (self.state['best_oks'])

        if self.do_track:
            message += '  recon_auc: %6.3f' % (self.state['best_recon_auc'])

        print(message)

    def print_rec(self, record):
        for key, value in record.items():
            print('[' + key + ']: {:.4f}'.format(value))
