import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from opts import args
from datasets import get_train_loader
from datasets import get_test_loader
from log import Logger
from train import Trainer

from resnet import resnet18
from resnet import resnet50

def get_catalogue():
    model_creators = dict();

    model_creators['resnet18'] = resnet18;
    model_creators['resnet50'] = resnet50;

    return model_creators;

def create_model(args):

    assert not (args.resume and args.pretrained)

    state = None;

    model_creators = get_catalogue();

    assert args.model in model_creators;

    model = model_creators[args.model](args);

    if args.test_only:
        save_path = os.path.join(args.save_path, args.model + '-' + args.suffix)

        print "=> Loading checkpoint from " + os.path.join(save_path, 'best.pth')
        assert os.path.exists(save_path), "[!] Checkpoint " + save_path + " doesn't exist" 

        best = torch.load(os.path.join(save_path, 'best.pth'))
        best = best['best'];
        
        checkpoint = os.path.join(save_path, 'model_%d.pth' % best)
        checkpoint = torch.load(checkpoint)
        
        model.load_state_dict(checkpoint['model'])

    if args.resume:
        print "=> Loading checkpoint from " + args.model_path
        checkpoint = torch.load(args.model_path)
        
        model.load_state_dict(checkpoint['model'])
        state = checkpoint['state']

    if args.nGPU > 0:
        cudnn.benchmark = True
        if args.nGPU > 1:
            model = nn.DataParallel(model, device_ids=[i for i in xrange(args.nGPU)]).cuda()
        else:
            model = model.cuda()

    return model, state

def main():
    model, state = create_model(args)
    print "=> Model and criterion are ready"

    if args.test_only:
        test_loader, joint_info = get_test_loader(args, 'test')
    else:
        train_loader, joint_info = get_train_loader(args)
        val_loader, joint_info = get_test_loader(args, 'validation')
    print "=> Dataloaders are ready"

    logger = Logger(args, state)
    print "=> Logger is ready"

    trainer = Trainer(args, model, joint_info)
    print "=> Trainer is ready"

    if args.test_only:
        test_summary = trainer.test(0, test_loader)
        print "- Test Model:  pck %6.3f  auc %6.3f  overall_mean %6.3f" % (
            test_summary['score_pck'], test_summary['score_auc'], test_summary['overall_mean'])

    else:
        start_epoch = logger.state['epoch'] + 1
        print "=> Start training"
        
        for epoch in xrange(start_epoch, args.n_epochs + 1):
            train_summary = trainer.train(epoch, train_loader)
            val_summary = trainer.test(epoch, val_loader)

            logger.record(epoch, train_summary, val_summary, model) 

        logger.final_print()

if __name__ == '__main__':
    main()
