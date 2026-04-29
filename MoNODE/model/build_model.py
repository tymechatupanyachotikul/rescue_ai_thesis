from model.core.mlp import MLP
from model.core.flow import Flow
from model.core.vae import VAE, SONODE_init_velocity, EncoderRNN
from model.core.inv_enc import INV_ENC
from model.core.model import MoNODE
from model.core.hbnode import HBNODE_BASE
from model.core.simclr import SimCLRModel
from model.core.byol import BYOLModel


def build_model(args, device, dtype, **kwargs):
    """
    Builds a model object of monode.MoNODE based on training sequence

    @param args: model setup arguments
    @param device: device on which to store the model (cpu or gpu)
    @param dtype: dtype of the tensors
    @param params: dict of data properties (see config.yml)
    @return: an object of MoNODE class
    """
    #define training set-up
    aug = (args.task=='sin' or args.task=='lv' or args.task=='bb' or 'mocap' in args.task or 'ecg' in args.task) and args.modulator_dim>0
    Nobj = args.Nobj

    if args.model == 'hbnode':
        D_in  = int(args.ode_latent_dim/args.order)
        D_out = int(args.ode_latent_dim / args.order)
    else:
        D_in = args.ode_latent_dim
        D_out = int(args.ode_latent_dim / args.order)

    if aug: # augmented dynamics
        D_in += args.modulator_dim

    # latent ode 
    if args.model == 'node' or args.model == 'sonode':
        if args.model == 'node':
            de = MLP(D_in, D_out, L=args.de_L, H=args.de_H, act='softplus')
        elif args.model == 'sonode':
            de = MLP(D_in, D_out, L=args.de_L, H=args.de_H, act='elu')

        flow = Flow(diffeq=de, order=args.order, solver=args.solver, use_adjoint=args.use_adjoint)

    elif args.model == 'hbnode':
        de = MLP(D_in, D_out, L=args.de_L, H=args.de_H, act='softplus')
        odefunc = HBNODE_BASE(de, corr=0, corrf=True)

        flow = Flow(diffeq=None, solver=args.solver, use_adjoint=args.use_adjoint)
        flow.odefunc = odefunc

    elif args.model == 'vae':
        flow = None  # no ODE integration

    # encoder & decoder
    if args.model in ('node', 'hbnode', 'vae'):
        vae = VAE(task=args.task, cnn_filt_enc=args.cnn_filt_enc, cnn_filt_de=args.cnn_filt_de, ode_latent_dim=args.ode_latent_dim//args.order,
            dec_act=args.dec_act, rnn_hidden=args.rnn_hidden, dec_H=args.dec_H, dec_L=args.dec_L, enc_H=args.enc_H,
            content_dim=args.content_dim, T_in=args.T_in, order=args.order, device=device,
            use_rnn_decoder=(args.model == 'vae'),
            rnn_hidden_dec=getattr(args, 'rnn_hidden_dec', None),
            **kwargs).to(dtype)
    elif args.model == 'sonode':
        if args.sonode_v == 'MLP':
            vae = SONODE_init_velocity(dim=args.ode_latent_dim//2, nhidden=args.dec_H, Tin=args.T_in) #improved SONODE
        elif args.sonode_v == 'RNN':
            vae = EncoderRNN(args.ode_latent_dim//2, rnn_hidden=args.rnn_hidden, enc_out_dim=args.ode_latent_dim//2, H=args.enc_H, out_distr='dirac')
        else:
            raise ValueError('Invalid sonode velocity encoder passeed {}'.format(args.sonode_v))

    # time-invariant network
    if args.modulator_dim>0 or args.content_dim>0:
        if args.task == 'bb':
            inv_enc = INV_ENC(task=args.task, modulator_dim=args.modulator_dim, content_dim = args.content_dim,
                cnn_filt=args.cnn_filt_inv, rnn_hidden=args.inv_rnn_hidden, T_inv=args.T_inv, vae_enc=vae.encoder, device=device, **kwargs).to(dtype)
        else:
            inv_enc = INV_ENC(task=args.task, modulator_dim=args.modulator_dim, content_dim = args.content_dim,
                cnn_filt=args.cnn_filt_inv, rnn_hidden=args.inv_rnn_hidden, T_inv=args.T_inv, vae_enc=None, device=device, inp_dim=kwargs['inp_dim']).to(dtype)
    else:
        inv_enc = None

    #full model
    monode = MoNODE(model = args.model,
                        flow = flow,
                        vae = vae,
                        inv_enc = inv_enc,
                        order = args.order,
                        dt  = args.dt,
                        aug = aug,
                        nobj=Nobj,
                        Tin=args.T_in)

    return monode


def build_simclr_model(args, device, dtype, inp_dim):
    """Build an encoder-only SimCLRModel. inp_dim = number of ECG leads after exclusion."""
    enc_out_dim = args.ode_latent_dim // args.order
    return SimCLRModel(
        input_dim   = inp_dim,
        enc_out_dim = enc_out_dim,
        rnn_hidden  = args.rnn_hidden,
        enc_H       = args.enc_H,
        proj_dim    = args.proj_dim,
        device      = device,
        dtype       = dtype,
    )


def build_byol_model(args, device, dtype, inp_dim):
    """Build a BYOLModel (online + target encoder-projector). inp_dim = ECG leads after exclusion."""
    enc_out_dim = args.ode_latent_dim // args.order
    return BYOLModel(
        input_dim   = inp_dim,
        enc_out_dim = enc_out_dim,
        rnn_hidden  = args.rnn_hidden,
        enc_H       = args.enc_H,
        proj_dim    = args.proj_dim,
        device      = device,
        dtype       = dtype,
    )