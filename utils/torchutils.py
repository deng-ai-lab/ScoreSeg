from torch.optim import lr_scheduler


def get_scheduler(optimizer, args):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args['sheduler']['lr_policy'] == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args['n_epoch'] + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args['sheduler']['lr_policy'] == 'step':
        step_size = args['n_epoch'] // args['sheduler']['n_steps']
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=args['sheduler']['gamma'])
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler
