import os
import json
import time
import utils
import shutil
import logging
import argparse
import importlib
import os.path as osp
import torch
import torch.distributed as dist
from datetime import datetime, timedelta
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import EpochBasedRunner, build_optimizer, load_checkpoint
from mmdet.apis import set_random_seed
from mmdet.core import DistEvalHook, EvalHook
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from loaders.builder import build_dataloader


class SynchronizedDistEvalHook(DistEvalHook):
    def _do_evaluate(self, runner):
        super()._do_evaluate(runner)

        if dist.is_initialized() and dist.is_available():
            dist.barrier()


def main():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', required=True)
    parser.add_argument('--override', nargs='+', action=DictAction)
    parser.add_argument('--work_dir', default=None)
    parser.add_argument('--wandb_id', default=None)
    parser.add_argument('--wandb_name', default=None)
    parser.add_argument('--wandb_tags', default=None)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    args = parser.parse_args()

    # parse configs
    cfgs = Config.fromfile(args.config)
    if args.override is not None:
        cfgs.merge_from_dict(args.override)

    if args.wandb_id is not None:
        wandb_tags = (
            json.loads(args.wandb_tags)
            if args.wandb_tags is not None else [])
        if not isinstance(wandb_tags, list):
            raise ValueError('wandb_tags must be a JSON list of strings.')

        wandb_hook_found = False
        for hook_cfg in cfgs.log_config.hooks:
            if hook_cfg['type'] != 'WandbLoggerHook':
                continue
            wandb_hook_found = True
            init_kwargs = hook_cfg.setdefault('init_kwargs', {})
            init_kwargs['id'] = args.wandb_id
            init_kwargs['name'] = args.wandb_name
            init_kwargs['tags'] = wandb_tags
            init_kwargs['resume'] = 'never'

        if not wandb_hook_found:
            raise ValueError(
                'WandbLoggerHook not found in config. Please add it to log_config.hooks.')

    # register custom module
    importlib.import_module('models')
    importlib.import_module('loaders')

    # MMCV, please shut up
    from mmcv.utils.logging import logger_initialized
    logger_initialized['root'] = logging.Logger(__name__, logging.WARNING)
    logger_initialized['mmcv'] = logging.Logger(__name__, logging.WARNING)

    # you need GPUs
    assert torch.cuda.is_available()

    # determine local_rank and world_size
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = str(args.world_size)

    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    logging.info('Using GPU: %s' % torch.cuda.get_device_name(local_rank))
    torch.cuda.set_device(local_rank)

    timestamp = torch.tensor(time.time(), dtype=torch.float64).cuda()
    if world_size > 1:
        logging.info('Initializing DDP with %d GPUs...' % world_size)
        dist_timeout_minutes = cfgs.get('dist_timeout_minutes', 100)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            timeout=timedelta(minutes=dist_timeout_minutes))
        dist.broadcast(timestamp, 0)

    # resume or start a new run
    if args.work_dir is not None:
        work_dir = args.work_dir
    elif cfgs.resume_from is not None:
        assert os.path.isfile(cfgs.resume_from)
        work_dir = os.path.dirname(cfgs.resume_from)
    else:
        run_name = osp.splitext(osp.split(args.config)[-1])[0]
        run_name += '_' + time.strftime("%Y-%m-%d/%H-%M-%S",
                                        time.localtime(timestamp.cpu().item()))
        work_dir = os.path.join('outputs', cfgs.model.type, run_name)

    if local_rank == 0:
        if cfgs.resume_from is None:
            if os.path.exists(work_dir):  # must be an empty dir
                raise FileExistsError(work_dir)
            os.makedirs(work_dir, exist_ok=False)

        # init logging, backup code
        utils.init_logging(os.path.join(work_dir, 'train.log'), cfgs.debug)
        utils.backup_code(work_dir)
        logging.info('Logs will be saved to %s' % work_dir)
    else:
        # disable logging on other workers
        logging.root.disabled = True

    logging.info('Setting random seed: 0')
    set_random_seed(0, deterministic=True)

    logging.info('Loading training set from %s' % cfgs.dataset_root)
    train_dataset = build_dataset(cfgs.data.train)
    train_loader = build_dataloader(
        train_dataset,
        samples_per_gpu=cfgs.batch_size // world_size,
        workers_per_gpu=cfgs.data.workers_per_gpu,
        num_gpus=world_size,
        dist=world_size > 1,
        shuffle=True,
        seed=0,
    )

    logging.info('Loading validation set from %s' % cfgs.dataset_root)
    val_dataset = build_dataset(cfgs.data.val)
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfgs.data.workers_per_gpu,
        num_gpus=world_size,
        dist=world_size > 1,
        shuffle=False
    )

    logging.info('Creating model: %s' % cfgs.model.type)
    model = build_detector(cfgs.model)
    model.init_weights()

    # logging.info(f'Model:\n{model}')
    model.cuda()
    model.train()

    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logging.info('Trainable parameters: %d (%.1fM)' % (n_params, n_params / 1e6))
    logging.info('Batch size per GPU: %d' % (cfgs.batch_size // world_size))

    if world_size > 1:
        model = MMDistributedDataParallel(model, [local_rank], broadcast_buffers=False)
    else:
        model = MMDataParallel(model, [0])
    
    logging.info('Creating optimizer: %s' % cfgs.optimizer.type)
    optimizer = build_optimizer(model, cfgs.optimizer)

    runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=work_dir,
        logger=logging.root,
        max_epochs=cfgs.total_epochs,
        meta=dict(),
    )

    runner.register_lr_hook(cfgs.lr_config)
    runner.register_optimizer_hook(cfgs.optimizer_config)
    runner.register_checkpoint_hook(cfgs.checkpoint_config)
    runner.register_logger_hooks(cfgs.log_config)
    runner.register_timer_hook(dict(type='IterTimerHook'))
    runner.register_custom_hooks(dict(type='DistSamplerSeedHook'))
    if cfgs.get('custom_hooks', None) is not None:
        for hook_cfg in cfgs.custom_hooks:
            runner.register_custom_hooks(hook_cfg)

    if cfgs.eval_config['interval'] > 0:
        if world_size > 1:
            runner.register_hook(SynchronizedDistEvalHook(
                val_loader, interval=cfgs.eval_config['interval'], gpu_collect=False))
        else:
            runner.register_hook(EvalHook(val_loader, interval=cfgs.eval_config['interval']))

    if cfgs.resume_from is not None:
        logging.info('Resuming from %s' % cfgs.resume_from)
        runner.resume(cfgs.resume_from)

    elif cfgs.load_from is not None:
        logging.info('Loading checkpoint from %s' % cfgs.load_from)
        if cfgs.revise_keys is not None:
            load_checkpoint(
                model, cfgs.load_from, map_location='cpu',
                revise_keys=cfgs.revise_keys,
                strict=False
            )
        else:
            load_checkpoint(
                model, cfgs.load_from, map_location='cpu',
                strict=False
            )

    runner.run([train_loader], [('train', 1)])


if __name__ == '__main__':
    main()
