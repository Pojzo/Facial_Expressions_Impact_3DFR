# Source of this code: https://github.com/guochengqian/PointNeXt
# Modified by Peter Kovac

import os, logging, csv, numpy as np, wandb
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, load_checkpoint_inv, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from torch.utils.data import DataLoader
from openpoints.models import build_model_from_cfg
from openpoints.models.layers import furthest_point_sample, fps

from train import train_one_epoch, print_cls_results
import open3d as o3d

from utils import create_datasets

logging.disable(logging.CRITICAL)

def main(gpu, cfg, profile=False):
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)
        dist.barrier()
    # logger
    # print("settings up logger dist", cfg.log_path)
    # setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    if cfg.rank == 0 :
        Wandb.launch(cfg, cfg.wandb.use_wandb)
        writer = SummaryWriter(log_dir=cfg.run_dir)
    else:
        writer = None
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    if not cfg.model.get('criterion_args', False):
        cfg.model.criterion_args = cfg.criterion_args

    print("Dostal som sa sem")
    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    # return model

    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))
    # criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()
    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        logging.info('Using Distributed Data parallel ...')

    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    cfg.dataloader.num_workers = "0"

    # dest = os.path.join("../../data_faces")
    dest = r"C:/Users/pojzi/Programovanie/bakalarka/PointNeXt/data_faces"
    print(dest)

    train_dataset, test_dataset = create_datasets(dest, cfg)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory = True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory = True
    )

    num_classes = val_loader.dataset.num_classes if hasattr(
        val_loader.dataset, 'num_classes') else None
    num_points = val_loader.dataset.num_points if hasattr(
        val_loader.dataset, 'num_points') else None
    if num_classes is not None:
        print(cfg.num_classes, num_classes)
        assert cfg.num_classes == num_classes
 
    logging.info(f"number of classes of the dataset: {num_classes}, "
                 f"number of points sampled from dataset: {num_points}, "
                 f"number of points as model input: {cfg.num_points}")
    cfg.classes = cfg.get('classes', None) or val_loader.dataset.classes if hasattr(
        val_loader.dataset, 'classes') else None or np.range(num_classes)
    validate_fn = validate #eval(cfg.get('val_fn', 'validate'))


    # optionally resume from a checkpoint
    if cfg.pretrained_path is not None:
        if cfg.mode == 'resume':
            resume_checkpoint(cfg, model, optimizer, scheduler,
                              pretrained_path=cfg.pretrained_path)
            macc, oa, train_maccs, cm = validate_fn(model, train_loader, cfg)
            print_cls_results(oa, macc, train_maccs, cfg.start_epoch, cfg)
        else:
            if cfg.mode == 'test':
                # test mode
                epoch, best_val = load_checkpoint(
                    model, pretrained_path=cfg.pretrained_path)
                macc, oa, train_maccs, cm = validate_fn(model, test_loader, cfg)
                print_cls_results(oa, macc, train_maccs, epoch, cfg)
                return True
            elif cfg.mode == 'val':
                # validation mode
                epoch, best_val = load_checkpoint(model, cfg.pretrained_path)
                macc, oa, train_maccs, cm = validate_fn(model, val_loader, cfg)
                print_cls_results(oa, macc, train_maccs, epoch, cfg)
                return True
            elif cfg.mode == 'finetune':
                # finetune the whole model
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model, cfg.pretrained_path)
            elif cfg.mode == 'finetune_encoder':
                # finetune the whole model
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model.encoder, cfg.pretrained_path)
            elif cfg.mode == 'finetune_encoder_inv':
                # finetune the whole model
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint_inv(model.encoder, cfg.pretrained_path)
    else:
        logging.info('Training from scratch')
    
    # train_loader = build_dataloader_from_cfg(cfg.batch_size,
    #                                          cfg.dataset,
    #                                          cfg.dataloader,
    #                                          datatransforms_cfg=cfg.datatransforms,
    #                                          split='train',
    #                                          distributed=cfg.distributed,
    #                                          )

    logging.info(f"length of training dataset: {len(train_loader.dataset)}")

    print(next(iter(train_loader)).keys())

    val_loader = test_loader

    # ===> start traininkkg
    val_macc, val_oa, val_accs, best_val, macc_when_best, best_epoch = 0., 0., [], 0., 0., 0
    model.zero_grad()
    losses = []
    train_maccs = []
    val_maccs = []
    for epoch in range(cfg.start_epoch, cfg.start_epoch + cfg.epochs):
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, 'epoch'):
            train_loader.dataset.epoch = epoch - 1
        train_loss, train_macc, train_oa, _, _ = \
            train_one_epoch(model, train_loader,
                            optimizer, scheduler, epoch, cfg)
        
        losses.append(train_loss)
        train_maccs.append(train_macc)

        is_best = False
        if epoch % cfg.val_freq == 0:
            val_macc, val_oa, val_accs, val_cm, _= validate_fn(
                model, val_loader, cfg)
            is_best = val_oa > best_val
            if is_best:
                best_val = val_oa
                macc_when_best = val_macc
                best_epoch = epoch
                logging.info(f'Find a better ckpt @E{epoch}')
                print_cls_results(val_oa, val_macc, val_accs, epoch, cfg)
        
        val_maccs.append(val_macc)

        lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch} LR {lr:.6f} '
                     f'train_oa {train_oa:.2f}, val_oa {val_oa:.2f}, best val oa {best_val:.2f}')
        if writer is not None:
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_oa', train_macc, epoch)
            writer.add_scalar('lr', lr, epoch)
            writer.add_scalar('val_oa', val_oa, epoch)
            writer.add_scalar('mAcc_when_best', macc_when_best, epoch)
            writer.add_scalar('best_val', best_val, epoch)
            writer.add_scalar('epoch', epoch, epoch)

        if cfg.sched_on_epoch:
            scheduler.step(epoch)
        if cfg.rank == 0:
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'best_val': best_val},
                            is_best=is_best
                            )

    test_macc, test_oa, test_accs, test_cm, exp_acc = validate(model, test_loader, cfg)
    print_cls_results(test_oa, test_macc, test_accs, best_epoch, cfg)
    if writer is not None:
        writer.add_scalar('test_oa', test_oa, epoch)
        writer.add_scalar('test_macc', test_macc, epoch)

    # test the best validataion model
    best_epoch, _ = load_checkpoint(model, pretrained_path=os.path.join(
        cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
    test_macc, test_oa, test_accs, test_cm, ex_acc = validate(model, test_loader, cfg)
    if writer is not None:
        writer.add_scalar('test_oa', test_oa, best_epoch)
        writer.add_scalar('test_macc', test_macc, best_epoch)
    print_cls_results(test_oa, test_macc, test_accs, best_epoch, cfg)

    if writer is not None:
        writer.close()

    # draw_loss_curve(losses)
    # draw_accuracy_curve(accs)

    obj = {
        'train_maccs': train_maccs,
        'losses': losses,
        'val_maccs': val_maccs,
        'test_macc': test_macc,
        'test_oa': test_oa,
        'test_accs': test_accs.tolist(),
        'num_classes': cfg.num_classes,
        'num_points': cfg.num_points,
        'train_mode': cfg.train_mode,
        'exp_acc': exp_acc,
    }

    # print type of each element in the dictionary
    for key in obj:
        print(key, type(obj[key]))

    import pickle 
    with open(cfg.figures_path + "/results.pkl", 'wb') as f:
        pickle.dump(obj, f)
        print("Results saved to: ", cfg.figures_path + "/results.pkl")

    # draw_curve(train_maccs, {'title': 'Accuracy', 'num_classes': str(cfg.num_classes), 'num_points': str(cfg.num_points)}, savepath=cfg.figures_path)
    # draw_curve(losses, {'title': 'Loss', 'num_classes': str(cfg.num_classes), 'num_points': str(cfg.num_points)}, savepath=cfg.figures_path)
    
    return test_macc, test_oa, test_accs, test_cm, exp_acc

    dist.destroy_process_group()

@torch.no_grad()
def validate(model, val_loader, cfg):
    print("This was called right")
    model.eval()  # set model to eval mode
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    npoints = cfg.num_points
    print("This is the length of the val_loader", val_loader.__len__())
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())

    expression_accuracies = {}

    dataset = val_loader.dataset
    for idx, data in pbar:
        is_neutral = dataset.is_neutral(idx)

        if dataset.train_mode == "neutral-neutral" and not is_neutral:
            continue

        if dataset.train_mode == "neutral-modified" is is_neutral:
            continue

        for key in data.keys():
            if key != 'exp':
                data[key] = data[key].cuda()
        

        target = data['y']
        exp = data['exp']
        points = data['x']
        points = points[:, :npoints]
        data['pos'] = points[:, :, :3].contiguous()
        data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()
        logits = model(data)
        

        argmax = logits.argmax(dim=1)

        for output, target, exp in zip(argmax, target, exp):
            if not exp in expression_accuracies:
                expression_accuracies[exp] = {'positive': 0, 'count': 0}

            output_item = output.item()
            target_item = target.item()

            if output_item == target_item:
                expression_accuracies[exp]['positive'] += 1
            
            expression_accuracies[exp]['count'] += 1

        cm.update(logits.argmax(dim=1), target)

    tp, count = cm.tp, cm.count
    if cfg.distributed:
        dist.all_reduce(tp), dist.all_reduce(count)
    macc, overallacc, accs = cm.cal_acc(tp, count)
    return macc, overallacc, accs, cm, expression_accuracies 

def train_one_epoch(model, train_loader, optimizer, scheduler, epoch, cfg):
    loss_meter = AverageMeter()
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    npoints = cfg.num_points

    model.train()  # set model to training mode
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0
    for idx, data in pbar:
        for key in data.keys():
            if key != 'exp':
                data[key] = data[key].cuda(non_blocking=True)
        num_iter += 1
        points = data['x']
        target = data['y']
        """ bebug
        from openpoints.dataset import vis_points
        vis_points(data['pos'].cpu().numpy()[0])
        """
        num_curr_pts = points.shape[1]
        if num_curr_pts > npoints:  # point resampling strategy
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()
            if  points.size(1) < point_all:
                point_all = points.size(1)
            fps_idx = furthest_point_sample(
                points[:, :, :3].contiguous(), point_all)
            fps_idx = fps_idx[:, np.random.choice(
                point_all, npoints, False)]
            points = torch.gather(
                points, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, points.shape[-1]))

        data['pos'] = points[:, :, :3].contiguous()
        data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()
        # print(data['x'].shape, target.shape)
        logits, loss = model.get_logits_loss(data, target) if not hasattr(model, 'module') else model.module.get_logits_loss(data, target)
        loss.backward()

        # optimize
        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_norm_clip, norm_type=2)
            num_iter = 0
            optimizer.step()
            model.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

        # update confusion matrix
        cm.update(logits.argmax(dim=1), target)
        loss_meter.update(loss.item())
        if idx % cfg.print_freq == 0:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                 f"Loss {loss_meter.val:.3f} Acc {cm.overall_accuray:.2f}")
    macc, overallacc, accs = cm.all_acc()
    return loss_meter.avg, macc, overallacc, accs, cm