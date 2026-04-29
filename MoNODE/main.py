import os
import argparse, time
import numpy as np
import torch
from datetime import datetime
from model.build_model import build_model
from model.model_misc import train_model
from model.misc import io_utils
from model.misc.torch_utils import seed_everything, count_params
from data.data_utils import load_data
from finetune import run_post_training_probes
from summarize_results import save_run_summary
import wandb

SOLVERS   = ["euler", "bdf", "rk4", "midpoint", "adams", "explicit_adams", "fixed_adams", "dopri5"]
TASKS     = ['rot_mnist', 'rot_mnist_ou', 'sin', 'bb', 'lv', 'mocap', 'mocap_shift', 'ecg']
MODELS     = ['node', 'sonode', 'hbnode', 'vae']
GRADIENT_ESTIMATION = ['no_adjoint', 'adjoint', 'ac_adjoint']
parser = argparse.ArgumentParser('MoNODE')
np.seterr(all='raise')

#data
parser.add_argument('--task', type=str, default='mov_mnist', choices=TASKS,
                    help="Experiment type")
parser.add_argument('--noise', type=float, default=None,
                    help="set noise level for noise robustness experiments")  
parser.add_argument('--Nobj', type=int, default=1,
                    help="param that can be used for multiple object set-up")                 
parser.add_argument('--num_workers', type=int, default=0,
                    help="number of workers")
parser.add_argument('--data_root', type=str, default='data/',
                    help="general data location")
parser.add_argument('--shuffle', type=eval, default=True,
               help='For Moving MNIST whetehr to shuffle the data')
parser.add_argument('--dataset_root', type=str, default='/projects/prjs1890/',
                    help="dataset location for ecg")
parser.add_argument('--segment_type', choices=['atrial', 'ventricular', 'whole'],
                    help="Segment type of heart beat", type=str)
parser.add_argument('--dataset', type=str, default='MedalCare-XL',
                    help='Dataset to use')
parser.add_argument('--exclude_leads_out', action='store_true', default=False,
                    help="If set, exclude leads ['II', 'III', 'aVR', 'aVL'] from output (overrides config.yml)")
parser.add_argument('--resample_freq', type=int, default=None,
                    help="Resample ECG to this frequency (Hz). None keeps the original rate.")

#de model
parser.add_argument('--model', type=str, default='node', choices=MODELS,
                    help='node model type')
parser.add_argument('--ode_latent_dim', type=int, default=10,
                    help="Latent ODE dimensionality")
parser.add_argument('--de_L', type=int, default=2,
                    help="Number of hidden layers in MLP diff func")
parser.add_argument('--de_H', type=int, default=100,
                    help="Number of hidden neurons in each layer of MLP diff func")


#invariance
parser.add_argument('--inv_fnc', type=str, default='MLP',
                    help="Invariant function")
parser.add_argument('--modulator_dim', type=int, default=0,
                    help = 'dim of the dynamics modulator variable')
parser.add_argument('--content_dim', type=int, default=0,
                    help = 'dim of the content variable')
parser.add_argument('--T_inv', type=int, default=5,
                    help="Time frames to select for RNN based Encoder for Invariance")
parser.add_argument('--cnn_filt_inv', type=int, default=16,
                    help="Nfilt invariant encoder cnn")


#ode stuff
parser.add_argument('--order', type=int, default=1,
                    help="order of ODE")
parser.add_argument('--solver', type=str, default='euler', choices=SOLVERS,
                    help="ODE solver for numerical integration")
parser.add_argument('--dt', type=float, default=0.1,
                    help="numerical solver dt")
parser.add_argument('--use_adjoint', type=str, default='no_adjoint', choices=GRADIENT_ESTIMATION, #we used False
                    help="Use adjoint method for gradient computation")

#vae 
parser.add_argument('--T_in', type=int, default=10,
                    help="Time frames to select for RNN based Encoder for intial state")
parser.add_argument('--cnn_filt_enc', type=int, default=16,
                    help="Number of filters in the cnn encoder")
parser.add_argument('--cnn_filt_de', type=int, default=16,
                    help="Number of filters in the cnn decoder")
parser.add_argument('--rnn_hidden', type=int, default=10,
                    help="Encoder RNN latent dimensionality")
parser.add_argument('--rnn_hidden_dec', type=int, default=None,
                    help="RNN decoder hidden dimensionality (VAE only). Defaults to rnn_hidden if not set.")
parser.add_argument('--inv_rnn_hidden', type=int, default=10,
                    help="RNN hidden dimensionality for the invariance encoder (INV_ENC)")
parser.add_argument('--dec_H', type=int, default=100,
                    help="Number of hidden neurons in MLP decoder") 
parser.add_argument('--dec_L', type=int, default=2,
                    help="Number of hidden layers in MLP decoder") 
parser.add_argument('--dec_act', type=str, default='relu',
                    help="MLP Decoder activation") 
parser.add_argument('--enc_H', type=int, default=50,
                    help="Encoder hidden dimensionality for GRU unit") 
parser.add_argument('--sonode_v', type=str, default='MLP', choices=['MLP','RNN'],
                    help="velocity encoder for SONODE")
parser.add_argument('--beta', type=float, default=1.0,
                    help="Beta weight on KL divergence for VAE training")

#simclr pretraining
parser.add_argument('--simclr_pretrain', action='store_true', default=False,
                    help='Run pure SimCLR pretraining (encoder-only, NT-Xent loss, no ELBO)')
parser.add_argument('--simclr_temp', type=float, default=0.5,
                    help='Temperature for NT-Xent loss')
parser.add_argument('--proj_dim', type=int, default=64,
                    help='Projection head output dimension')
parser.add_argument('--noise_sigma', type=float, default=0.05,
                    help='Gaussian noise std for SimCLR augmentation')
parser.add_argument('--crop_min_frac', type=float, default=0.7,
                    help='Minimum crop fraction for random resized crop (0.7 = 70%%)')
parser.add_argument('--crop_max_frac', type=float, default=1.0,
                    help='Maximum crop fraction for random resized crop (1.0 = 100%%)')
parser.add_argument('--timeout_max_frac', type=float, default=0.2,
                    help='Maximum time-out fraction (0.2 = zero up to 20%% of signal)')

#byol pretraining
parser.add_argument('--byol_pretrain', action='store_true', default=False,
                    help='Run BYOL pretraining (online+target encoder, no ELBO)')
parser.add_argument('--byol_tau', type=float, default=0.996,
                    help='EMA momentum for BYOL target network update')

#training 
parser.add_argument('--Nepoch', type=int, default=600,
                    help="Number of gradient steps for model training")
parser.add_argument('--Nincr', type=int, default=10,
                    help="Number of sequential increments of the sequence length")
parser.add_argument('--batch_size', type=int, default=25,
                    help="batch size")
parser.add_argument('--lr', type=float, default=0.002,
                    help="Learning rate for model training")
parser.add_argument('--sobolev_weight', type=float, default=0,
                    help="Weight of derivative loss likelihood")
parser.add_argument('--l_w', type=float, default=0,
                    help="Weight of likelihood scaled on derivative")
parser.add_argument('--seed', type=int, default=121,
                    help="Global seed for the training run")
parser.add_argument('--continue_training', type=eval, default=False,
                    help="If set to True continoues training of a previous model")
parser.add_argument('--plot_every', type=int, default=20,
                    help="How often plot the training")
parser.add_argument('--plotL', type=int, default=1,
                    help="Number of MC draws for plotting")
parser.add_argument('--forecast_tr',type=int, default=2, 
                    help="Number of forecast steps for plotting train")
parser.add_argument('--forecast_vl',type=int, default=2,
                    help="Number of forecast steps for plotting test")
parser.add_argument('--exp_id', type=int, default=0,
                    help = 'exp ID for directory')

#log 
parser.add_argument('--save', type=str, default='results/',
                    help="Directory name for saving all the model outputs")
parser.add_argument('--aladin_metadata_dir', type=str, default=None,
                    help="Path to directory containing ALADIN metadata JSON files "
                         "(train_metadata.json, valid_metadata.json, test_metadata.json)")
parser.add_argument('--early_stopping_patience', type=int, default=10,
                    help="Stop training if validation MSE does not improve for this many "
                         "consecutive validation checks. Set to 0 to disable.")
parser.add_argument('--summary_output_dir', type=str, default=None,
                    help="Directory to save the run summary JSON "
                         "({segment_type}_{model}_{dataset}.json). "
                         "Skipped if not provided.")
parser.add_argument('--summary_filename', type=str, default=None,
                    help="Override the auto-generated summary JSON filename "
                         "(e.g. my_run.json). Must include the .json extension. "
                         "Ignored if --summary_output_dir is not set.")
parser.add_argument('--finetune_dir', type=str, default=None,
                    help="Directory name for finetune results directory")
parser.add_argument('--continue_dir', type=str, default='results/',
                    help="Directory name for continue training")
parser.add_argument('--latent_dir', type=str, default=None,
                    help="Directory name for latents")


if __name__ == '__main__':
    args = parser.parse_args()
    ######### setup output directory and logger ###########
    args.save = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
        args.save+args.task+'/'+args.model+'/'+datetime.now().strftime('%d_%m_%Y-%H:%M:%S-')+str(args.exp_id), '')
    
    ############################
    io_utils.makedirs(args.save)
    io_utils.makedirs(os.path.join(args.save, 'plots'))
    io_utils.makedirs(os.path.join(args.save, 'plots', 'fit'))
    io_utils.makedirs(os.path.join(args.save, 'plots', 'latents'))
    logger = io_utils.get_logger(logpath=os.path.join(args.save, 'logs.txt'))
    logger.info('Results stored in {}'.format(args.save))
    import json as _json
    with open(os.path.join(args.save, 'args.json'), 'w') as _f:
        _json.dump(vars(args), _f, indent=2)

    ########## set global random seed ###########
    if args.seed==-1:
        args.seed = int(time.time()*np.random.random()/1000)
    seed_everything(args.seed)

    ########## dtype #########
    dtype = torch.float64
    logger.info('********** Float type is {} ********** '.format(dtype))

    ########## plotter #######
    from model.misc.plot_utils import Plotter
    save_path = os.path.join(args.save, 'plots')
    plotter   = Plotter(save_path, args.task)

    ########### device #######
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('********** Running model on {} ********** '.format(device))

    ########### data ############ ``
    trainset, validset, testset, manager, params = load_data(args, dtype)
    if args.exclude_leads_out:
        params[args.task]['exclude_leads_out'] = ['II', 'III', 'aVR', 'aVL']
    run = wandb.init(
        entity="tymechatu-university-of-amsterdam",
        name=f'{args.model}_ode-{args.ode_latent_dim}_mod-{args.modulator_dim}_batch-{args.batch_size}_lr-{args.lr}_sample-{params[args.task]["sample_type"]}',
        project=f"NODE_{params[args.task]['beat_type']}" if params[args.task]['dataset'].lower() == 'medalcare-xl' \
            else f"NODE_{params[args.task]['beat_type']}_{params[args.task]['dataset'].lower()}",
        group="new",
        config=vars(args),
    )
    logger.info('********** {} dataset with loaded ********** '.format(args.task))
    logger.info('data params: {}'.format(params[args.task]))

    ########### model ###########
    if args.task == 'ecg':
        config = {
            'inp_dim': 12 - len(params[args.task]['exclude_leads_in']),
            'w_dt': args.sobolev_weight,
            'l_w': args.l_w,
            'out_dim': 12 - len(params[args.task]['exclude_leads_out'])
        }

    inp_dim = config.get('inp_dim', None) if args.task == 'ecg' else None

    if args.simclr_pretrain:
        from model.build_model import build_simclr_model
        from model.model_misc import train_simclr
        model = build_simclr_model(args, device, dtype, inp_dim)
        logger.info('********** Built SimCLRModel (encoder-only) **********')
        logger.info('********** Number of parameters: {} **********'.format(count_params(model)))
        for arg, value in sorted(vars(args).items()):
            logger.info("Argument %s: %r", arg, value)
        logger.info(model)
        if args.Nepoch > 0:
            train_simclr(args, model, trainset, validset, logger, run)
        fname = os.path.join(args.save, 'model.pth')
        if args.task == 'ecg':
            run_post_training_probes(args, model, device, trainset, testset, params[args.task], run,
                                      validset=validset, ckpt_path=fname, finetune_dir=args.finetune_dir)

    elif args.byol_pretrain:
        from model.build_model import build_byol_model
        from model.model_misc import train_byol
        model = build_byol_model(args, device, dtype, inp_dim)
        logger.info('********** Built BYOLModel (online + target encoder) **********')
        logger.info('********** Number of parameters: {} **********'.format(count_params(model)))
        for arg, value in sorted(vars(args).items()):
            logger.info("Argument %s: %r", arg, value)
        logger.info(model)
        if args.Nepoch > 0:
            train_byol(args, model, trainset, validset, logger, run)
        fname = os.path.join(args.save, 'model.pth')
        if args.task == 'ecg':
            run_post_training_probes(args, model, device, trainset, testset, params[args.task], run,
                                      validset=validset, ckpt_path=fname, finetune_dir=args.finetune_dir)

    else:
        model = build_model(args, device, dtype, **config)
        model.to(device)
        model.to(dtype)
        print(f'Number of model parameters : {sum(p.numel() for p in model.parameters())}')
        logger.info('********** Built {} model with dynamics modulator dim {} and  content variable dim {}**********'.format(args.model, args.modulator_dim, args.content_dim))
        logger.info('********** Number of parameters: {} **********'.format(count_params(model)))
        logger.info('********** Augmented Dynamics: {} **********'.format(model.aug))
        for arg, value in sorted(vars(args).items()):
            logger.info("Argument %s: %r", arg, value)
        logger.info(model)

        if args.continue_training:
            fname = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.continue_dir, 'model.pth')
            ckpt = torch.load(fname, map_location=torch.device(device), weights_only=False)
            if 'vae.decoder.out_logsig_dt' not in ckpt["state_dict"]:
                ckpt["state_dict"]["vae.decoder.out_logsig_dt"] = ckpt["state_dict"]["vae.decoder.out_logsig"]
            model.load_state_dict(ckpt["state_dict"])
            logger.info('********** Resume training for model {} ********** '.format(fname))

        if args.Nepoch > 0:
            train_model(args, model, plotter, trainset, validset, testset, logger, params[args.task], run)
            fname = os.path.join(args.save, 'model.pth')

        if args.task == 'ecg':
            run_post_training_probes(args, model, device, trainset, testset, params[args.task], run,
                                      validset=validset, ckpt_path=fname, finetune_dir=args.finetune_dir)

    if args.summary_output_dir:
        save_run_summary(
            run_dir=args.save if not args.continue_training else args.continue_dir,
            output_dir=args.summary_output_dir,
            model='monode' if args.modulator_dim > 0 else args.model,
            dataset=params[args.task]['dataset'],
            segment_type=getattr(args, 'segment_type', None),
            original_dir=None,
            filename=args.summary_filename,
            finetune_dir=args.finetune_dir
        )

    run.finish()


