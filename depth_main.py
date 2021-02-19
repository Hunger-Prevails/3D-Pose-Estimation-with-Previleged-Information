import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import importlib

from opts import args
from depth_datasets import get_data_loader
from log import Logger
from depth_train import Trainer


def create_model(args):
    assert not (args.resume and args.pretrain)
    assert not (args.do_fusion and args.depth_only)
    assert (args.depth_host <= args.do_fusion)

    model_creator = 'fusion' if args.do_fusion else 'depth'

    if args.partial_conv:
        model_creator = 'partial_' + model_creator

    model_creator = importlib.import_module(model_creator + 'net')

    assert hasattr(model_creator, args.model)
    model = getattr(model_creator, args.model)(args, args.pretrain)
    state = None;

    if args.test_only or args.val_only:
        save_path = os.path.join(args.save_path, args.model + '-' + args.suffix)

        print('=> Loading checkpoint from ' + os.path.join(save_path, 'best.pth'))
        assert os.path.exists(save_path)

        best = torch.load(os.path.join(save_path, 'best.pth'))
        best = best['best'];
        
        checkpoint = os.path.join(save_path, 'model_%d.pth' % best)
        checkpoint = torch.load(checkpoint)['model']

        toy_keys = set(checkpoint.keys())
        model_keys = set(model.state_dict().keys())

        assert len(model_keys.difference(toy_keys)) == 0
        
        model.load_state_dict(checkpoint)

    if args.resume:
        print('=> Loading checkpoint from ' + args.model_path)
        checkpoint = torch.load(args.model_path)
        
        model.load_state_dict(checkpoint['model'])
        state = checkpoint['state']

    if args.n_cudas:
        cudnn.benchmark = True
        model = model.cuda() if args.n_cudas == 1 else nn.DataParallel(model, device_ids = range(args.n_cudas)).cuda()

    return model, state


def create_pair(args):
    assert not (args.resume and args.pretrain)
    assert not (args.do_fusion and args.depth_only)
    assert not args.depth_host
    assert not args.depth_only
    assert not args.test_only
    assert not args.valid_only

    teacher_creator = 'fusion' if args.do_fusion else 'depth'

    if args.partial_conv:
        teacher_creator = 'partial_' + teacher_creator

    teacher_creator = importlib.import_module(teacher_creator + 'net')

    assert hasattr(teacher_creator, args.model)
    teacher = getattr(teacher_creator, args.model)(args, False)

    textbook = torch.load(args.teacher_path)['model']
    teacher.load_state_dict(textbook)

    model_creator = importlib.import_module('depthnet')

    assert hasattr(model_creator, args.model)
    model = getattr(model_creator, args.model)(args, args.pretrain)
    state = None

    if args.n_cudas:
        cudnn.benchmark = True
        model = model.cuda() if args.n_cudas == 1 else nn.DataParallel(model, device_ids = range(args.n_cudas)).cuda()
        teacher = teacher.cuda() if args.n_cudas == 1 else nn.DataParallel(model, device_ids = range(args.n_cudas)).cuda()

    return model, teacher, state

def main():
    if args.do_teach:
        model, teacher, state = create_pair(args)
    else:
        model, state = create_model(args)
    print('=> Model and criterion are ready')

    if args.test_only:
        test_loader, data_info = get_data_loader(args, 'test')
    elif args.val_only:
        test_loader, data_info = get_data_loader(args, 'valid')
    else:
        test_loader, data_info = get_data_loader(args, 'valid')

        data_loader, data_info = get_data_loader(args, 'train')

    print('=> Dataloaders are ready')

    logger = Logger(args, state)
    print('=> Logger is ready')

    trainer = Trainer(args, model, data_info)
    print('=> Trainer is ready')

    if args.do_teach:
        trainer.set_teacher(teacher)

    if args.test_only or args.val_only:
        print('=> Evaluation starts')
        test_rec = trainer.test(0, test_loader)
        logger.print_rec(test_rec)

    else:
        start_epoch = logger.state['epoch'] + 1
        print('=> Train starts')
        
        for epoch in range(start_epoch, args.n_epochs + 1):
            train_rec = trainer.train(epoch, data_loader)
            test_rec = trainer.test(epoch, test_loader)

            logger.record(epoch, train_rec, test_rec, model) 

        logger.final_print()

if __name__ == '__main__':
    main()
