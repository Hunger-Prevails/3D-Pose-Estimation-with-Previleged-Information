import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import importlib

from opts import args
from utils import JointInfo
from depth_datasets import get_data_loader
from log import Logger
from depth_train import Trainer


def get_info():
    from joint_settings import h36m_short_names as short_names
    from joint_settings import h36m_parent as parent
    from joint_settings import h36m_mirror as mirror
    from joint_settings import h36m_base_joint as base_joint

    mapper = dict(zip(short_names, range(len(short_names))))

    map_mirror = [mapper[mirror[name]] for name in short_names if name in mirror]
    map_parent = [mapper[parent[name]] for name in short_names if name in parent]

    _mirror = np.arange(len(short_names))
    _parent = np.arange(len(short_names))

    _mirror[np.array([name in mirror for name in short_names])] = np.array(map_mirror)
    _parent[np.array([name in parent for name in short_names])] = np.array(map_parent)

    data_info = JointInfo(short_names, _parent, _mirror, mapper[base_joint])

    return data_info


def create_model(args):
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

    if args.resume:
        print('=> Loading checkpoint from ' + args.model_path)
        checkpoint = torch.load(args.model_path)

        model.load_state_dict(checkpoint['model'])
        state = checkpoint['state']

    if args.n_cudas:
        cudnn.benchmark = True
        model = model.cuda() if args.n_cudas == 1 else nn.DataParallel(model, device_ids = range(args.n_cudas)).cuda()
        teacher = teacher.cuda() if args.n_cudas == 1 else nn.DataParallel(model, device_ids = range(args.n_cudas)).cuda()

    return model, teacher, state

def main():
    assert not (args.resume and args.pretrain)
    assert not (args.do_fusion and args.depth_only)
    assert not (args.depth_host and args.depth_only)

    if args.do_teach:
        model, teacher, state = create_pair(args)
    else:
        model, state = create_model(args)
    print('=> Models are created and filled')

    data_info = get_info()

    if args.test_only:
        test_loader = get_data_loader(args, 'test', data_info)
    elif args.val_only:
        test_loader = get_data_loader(args, 'valid', data_info)
    else:
        test_loader = get_data_loader(args, 'valid', data_info)

        data_loader = get_data_loader(args, 'train', data_info)

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
