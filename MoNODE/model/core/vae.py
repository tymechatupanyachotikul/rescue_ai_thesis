import torch
import torch.nn as nn
from torch.distributions import Normal
from torchsummary import summary
from model.misc.torch_utils import Flatten, UnFlatten
from model.core.gru_encoder import GRUEncoder
from model.core.mlp import MLP
import numpy as np

EPSILON = 1e-5


class SONODE_init_velocity(nn.Module):
    
    def __init__(self, dim, nhidden, Tin):
        super(SONODE_init_velocity, self).__init__()
        self.elu = nn.ELU(inplace=False)
        self.fc1 = nn.Linear(dim*Tin, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, dim)
        
    def forward(self, x0):
        N, T, q = x0.shape
        xin = x0.reshape(N, T*q)
        out = self.fc1(xin)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out 


def build_rot_mnist_cnn_enc(n_in_channels, n_filt):
    cnn =   nn.Sequential(
            nn.Conv2d(n_in_channels, n_filt, kernel_size=5, stride=2, padding=(2,2)), # 14,14
            nn.BatchNorm2d(n_filt),
            nn.ReLU(),
            nn.Conv2d(n_filt, n_filt*2, kernel_size=5, stride=2, padding=(2,2)), # 7,7
            nn.BatchNorm2d(n_filt*2),
            nn.ReLU(),
            nn.Conv2d(n_filt*2, n_filt*4, kernel_size=5, stride=2, padding=(2,2)),
            nn.BatchNorm2d(n_filt*4), # this is new
            nn.ReLU(),
            Flatten()
        )
    out_features = n_filt*4 * 4*4 # encoder output is [4*n_filt,4,4]
    return cnn, out_features

def build_rot_mnist_cnn_dec(n_filt, n_in):
    out_features = n_filt*4 * 4*4 # encoder output is [4*n_filt,4,4]
    cnn = nn.Sequential(
        nn.Linear(n_in, out_features),
        UnFlatten(4),
        nn.ConvTranspose2d(out_features//16, n_filt*8, kernel_size=3, stride=1, padding=(0,0)),
        nn.BatchNorm2d(n_filt*8),
        nn.ReLU(),
        nn.ConvTranspose2d(n_filt*8, n_filt*4, kernel_size=5, stride=2, padding=(1,1)),
        nn.BatchNorm2d(n_filt*4),
        nn.ReLU(),
        nn.ConvTranspose2d(n_filt*4, n_filt*2, kernel_size=5, stride=2, padding=(1,1), output_padding=(1,1)),
        nn.BatchNorm2d(n_filt*2),
        nn.ReLU(),
        nn.ConvTranspose2d(n_filt*2, 1, kernel_size=5, stride=1, padding=(2,2)),
        nn.Sigmoid(),
    )
    return cnn

def build_mov_mnist_cnn_enc(n_in_channels, n_filt):
	cnn = nn.Sequential(
            nn.Conv2d(n_in_channels, n_filt, kernel_size=5, stride=2, padding=(2,2)), # 32,32
            nn.BatchNorm2d(n_filt),
            nn.ReLU(),
            nn.Conv2d(n_filt, n_filt*2, kernel_size=5, stride=2, padding=(2,2)), # 16,16
            nn.BatchNorm2d(n_filt*2),
            nn.ReLU(),
            nn.Conv2d(n_filt*2, n_filt*4, kernel_size=5, stride=2, padding=(2,2)), # 8,8
            nn.BatchNorm2d(n_filt*4),
            nn.ReLU(),
            nn.Conv2d(n_filt*4, n_filt*8, kernel_size=5, stride=2, padding=(2,2)), # 4,4
            nn.BatchNorm2d(n_filt*8),
            nn.ReLU(),
            Flatten()
        )
	out_features = n_filt*8 * 4*4 
	return cnn, out_features

def build_mov_mnist_cnn_dec(n_filt, n_in):
    out_features = n_filt*8 * 4*4 # encoder output is [4*n_filt,4,4]
    cnn = nn.Sequential(
            nn.Linear(n_in, out_features),
            UnFlatten(4),
            nn.ConvTranspose2d(out_features//16, n_filt*8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n_filt*8),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filt*8, n_filt*4, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(n_filt*4),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filt*4, n_filt*2, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(n_filt*2),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filt*2, n_filt, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(n_filt),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filt, 1, kernel_size=6, stride=1, padding=2),
            nn.Sigmoid(),
        )
    return cnn


class VAE(nn.Module):

    def __init__(self, task, cnn_filt_enc=8, cnn_filt_de=8, dec_H=100, rnn_hidden=10, dec_act='relu',
                 ode_latent_dim=8, content_dim=0, T_in=10, device='cpu', order=1, enc_H=50, inp_dim=None, w_dt=0, l_w=0, out_dim=None, use_rnn_decoder=False):
        super(VAE, self).__init__()

        ### build encoder
        self.prior = Normal(torch.zeros(ode_latent_dim).to(device), torch.ones(ode_latent_dim).to(device))
        self.ode_latent_dim = ode_latent_dim
        self.order = order

        lhood_distribution = 'bernoulli'
        if task in ['rot_mnist', 'rot_mnist_ou']:
            self.encoder = PositionEncoderCNN(task=task, out_distr='normal', enc_out_dim=ode_latent_dim, n_filt=cnn_filt_enc, T_in=T_in).to(device)
            self.decoder = Decoder(task, ode_latent_dim+content_dim, n_filt=cnn_filt_de, distribution=lhood_distribution).to(device)
            
        elif task=='bb':
            self.encoder = EncoderCNNMLP(out_distr='normal', enc_out_dim=ode_latent_dim, n_filt=cnn_filt_enc, T_in=T_in).to(device)
            self.decoder = Decoder(task, ode_latent_dim+content_dim, n_filt=cnn_filt_de, distribution=lhood_distribution).to(device)
            if order == 2:
                self.encoder_v = EncoderCNNMLP(out_distr='normal', enc_out_dim=ode_latent_dim, n_filt=cnn_filt_enc, T_in=T_in).to(device)
                self.prior = Normal(torch.zeros(ode_latent_dim*order).to(device), torch.ones(ode_latent_dim*order).to(device))

        elif task in ['sin', 'lv', 'mocap', 'mocap_shift', 'ecg']:
            lhood_distribution = 'normal'

            if inp_dim:
                if out_dim is not None:
                    out_dim = out_dim
                else:
                    out_dim = inp_dim
                data_dim = inp_dim
            elif task=='sin':
                data_dim = 1
            elif task=='lv':
                data_dim = 2
            elif 'mocap' in task:
                data_dim = 50
                
            if rnn_hidden==-1:
                self.encoder = IdentityEncoder()
                self.decoder = IdentityDecoder(data_dim)
                if order==2:
                    self.encoder_v = IdentityEncoder()
            else:
                self.encoder = EncoderRNN(data_dim, rnn_hidden=rnn_hidden, enc_out_dim=ode_latent_dim, out_distr='normal', H=enc_H).to(device)
                if use_rnn_decoder:
                    self.decoder = RNNDecoder(ode_latent_dim+content_dim, rnn_hidden=rnn_hidden, dec_H=dec_H, data_dim=out_dim, act=dec_act).to(device)
                else:
                    self.decoder = Decoder(task, ode_latent_dim+content_dim, H=dec_H, distribution=lhood_distribution, dec_out_dim=out_dim, act=dec_act, w_dt=w_dt, l_w=l_w).to(device)
                if order==2:
                    self.encoder_v = EncoderRNN(data_dim, rnn_hidden=rnn_hidden, enc_out_dim=ode_latent_dim, out_distr='normal', H=enc_H).to(device)
                    self.prior = Normal(torch.zeros(ode_latent_dim*order).to(device), torch.ones(ode_latent_dim*order).to(device))

        
    def reset_parameters(self):
        modules = [self.encoder, self.decoder]
        if self.order==2:
            modules += [self.encoder_v]
        for module in modules:
            for layer in module.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def print_summary(self):
        """Print the summary of both the models: encoder and decoder"""
        summary(self.encoder, (1, *(28,28)))
        summary(self.decoder, (1, self.ode_latent_dim))
        if self.order==2:
            summary(self.encoder_v, (1,*(28,28)))

    def save(self, encoder_path=None, decoder_path=None):
        """Save the VAE model. Both encoder and decoder and saved in different files."""
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)

    def test(self, x):
        """Test the VAE model on data x. First x is encoded using encoder model, a sample is produced from then latent
        distribution and then it is passed through the decoder model."""
        self.encoder.eval()
        self.decoder.eval()
        enc_m, enc_log_var = self.encoder(x)
        z = self.encoder.sample(enc_m, enc_log_var)
        y = self.decoder(z)
        return y


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sp = nn.Softplus()
    
    @property
    def device(self):
        return self.sp.device

    def sample(self, mu, std, L=1):
        ''' mu,std  - [N,q]
            returns - [L,N,q] if L>1 else [N,q]'''
        if std is None:
            return mu
        eps = torch.randn([L,*std.shape]).to(mu.device).to(mu.dtype).squeeze(0) # [N,q] or [L,N,q]
        return mu + std*eps

    def q_dist(self, mu_s, std_s, mu_v=None, std_v=None):
        if mu_v is not None:
            means = torch.cat((mu_s,mu_v), dim=-1)
            stds  = torch.cat((std_s,std_v), dim=-1)
        else:
            means = mu_s
            stds  = std_s
            
        return Normal(means, stds) #N,q

    @property
    def device(self):
        return self.sp.device
    
class EncoderRCNN(AbstractEncoder):
    def __init__(self, out_distr='normal', enc_out_dim=16, n_filt=8, n_in_channels=1, rnn_output_size=50, H=50):
        super(EncoderRCNN, self).__init__()
        self.enc_out_dim = enc_out_dim
        self.out_distr  = out_distr
        self.cnn, self.in_features = build_rot_mnist_cnn_enc(n_in_channels, n_filt)
        self.fc1 = nn.Linear(self.in_features, enc_out_dim)
        self.gru = GRUEncoder(enc_out_dim*2, enc_out_dim, rnn_output_size=rnn_output_size, H=H)
        
    def forward(self, X):
        ''' X - [N,T,nc,w,w] '''
        [N,T,nc,w,w] = X.shape
        h = self.cnn(X.reshape(N*T,nc,w,w)).reshape(N,T,-1)
        z = self.fc1(h)
        outputs = self.gru(z)
        if self.out_distr=='normal':
            z0_mu, z0_log_sig = outputs[:,:self.enc_out_dim,], outputs[:,self.enc_out_dim:]
            z0_log_sig = self.sp(z0_log_sig)
            return z0_mu, z0_log_sig
        return outputs

class EncoderCNNMLP(AbstractEncoder):
    def __init__(self, out_distr='normal', enc_out_dim=16, n_filt=8, T_in=10, H=50):
        super(EncoderCNNMLP, self).__init__()
        self.T_in = T_in
        self.H    = H
        self.enc_out_dim = enc_out_dim
        self.out_distr  = out_distr
        self.cnn, self.in_features = build_rot_mnist_cnn_enc(1, n_filt)
        self.fc1 = nn.Sequential(nn.Linear(self.in_features, H), nn.ReLU()) 
        self.fc2 = nn.Linear(H*T_in, 2*enc_out_dim)
    
    def backbone(self,X):
        return self.fc1(self.cnn(X))
        
    def forward(self, X):
        ''' X - [N,T,nc,w,w] '''
        [N,T,nc,w,w] = X.shape
        h = self.backbone(X.reshape(N*T,nc,w,w)).reshape(N,T,-1).reshape(N,-1)
        outputs = self.fc2(h)
        if self.out_distr=='normal':
            z0_mu, z0_log_sig = outputs[:,:self.enc_out_dim,], outputs[:,self.enc_out_dim:]
            z0_log_sig = self.sp(z0_log_sig)
            return z0_mu, z0_log_sig
        return outputs


class EncoderCNN(AbstractEncoder):
    def __init__(self, task, out_distr='normal', enc_out_dim=16, n_filt=8, n_in_channels=1):
        super(EncoderCNN, self).__init__()
        self.enc_out_dim = enc_out_dim
        self.out_distr  = out_distr
        if task=='rot_mnist' or task=='rot_mnist_ou':
            self.cnn, in_features = build_rot_mnist_cnn_enc(n_in_channels, n_filt)
        elif task=='mov_mnist':
            self.cnn, in_features = build_mov_mnist_cnn_enc(n_in_channels, n_filt)
        else:
            raise ValueError(f'Unknown task {task}')
        self.fc1 = nn.Linear(in_features, enc_out_dim)
        if out_distr=='normal':
            self.fc2 = nn.Linear(in_features, enc_out_dim)
        
    def forward(self, X):
        ''' X - [N,T,nc,w,w] '''
        h = self.cnn(X)
        z0_mu = self.fc1(h)
        if self.out_distr=='normal':
            z0_log_sig = self.fc2(h) # N,q & N,q
            z0_log_sig = self.sp(z0_log_sig)
            return z0_mu, z0_log_sig
        else:
            return z0_mu

class PositionEncoderCNN(EncoderCNN):
    def __init__(self, task, out_distr='normal', enc_out_dim=16, n_filt=8, n_in_channels=1, T_in=1):
        super().__init__(task, out_distr=out_distr, enc_out_dim=enc_out_dim, n_filt=n_filt, n_in_channels=n_in_channels*T_in)
    def forward(self,X):
        [N,T,nc,d,d] = X.shape
        X_in = X.reshape(N, T*nc, d, d)
        return super().forward(X_in)

class VelocityEncoderCNN(EncoderCNN):
    def __init__(self, num_frames, task, out_distr='normal', enc_out_dim=16, n_filt=8, n_in_channels=1):
        super().__init__(task, out_distr=out_distr, enc_out_dim=enc_out_dim, n_filt=n_filt, n_in_channels=n_in_channels)
        self.num_frames = num_frames
    def forward(self,X):
        [N,T,nc,d,d] = X.shape
        X_in = X[:,:self.num_frames].reshape(N, self.num_frames*nc, d, d)
        return super().forward(X_in)

class EncoderRNN(AbstractEncoder):
    def __init__(self, input_dim, rnn_hidden=10, enc_out_dim=16, out_distr='normal', H=50):
        super(EncoderRNN, self).__init__()
        self.enc_out_dim = enc_out_dim
        self.out_distr   = out_distr
        enc_out_dim      = enc_out_dim + enc_out_dim*(out_distr=='normal')
        self.gru = GRUEncoder(enc_out_dim, input_dim, rnn_output_size=rnn_hidden, H=H)
        
    def forward(self, x, mask=None):
        lengths = None
        if mask is not None:
            lengths = mask.sum(dim=1).to(dtype=torch.int64, device=x.device)

        outputs = self.gru(x, lengths=lengths)
        if self.out_distr=='normal':
            z0_mu, z0_log_sig = outputs[:,:self.enc_out_dim,], outputs[:,self.enc_out_dim:]
            z0_log_sig = self.sp(z0_log_sig)
            return z0_mu, z0_log_sig
        return outputs

class IdentityEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def sample(self, mu, std, L=1):
        return torch.stack([mu]*L) if L>1 else mu
    def q_dist(self, mu_s, std_s, mu_v=None, std_v=None):
        return Normal(mu_s, torch.ones_like(mu_s)) #N,q
    def __call__(self,x):
        return x[:,0], None
    def __repr__(self) -> str:
        return 'Identity encoder'
    
class IdentityDecoder(nn.Module):
    def __init__(self,data_dim):
        super().__init__()
        self.data_dim = data_dim
    def __call__(self,z,dims):
        return z[...,:self.data_dim]
    def log_prob(self,X,Xhat,L=1):
        XL = X.repeat([L]+[1]*X.ndim) # L,N,T,nc,d,d or L,N,T,d
        assert XL.numel()==Xhat.numel()
        Xhat = Xhat.reshape(XL.shape)
        return torch.distributions.Normal(XL,torch.ones_like(XL)).log_prob(Xhat)
    def __repr__(self) -> str:
        return 'Identity decoder'


class RNNDecoder(nn.Module):
    """VAE-RNN decoder: z0 seeds the GRU hidden state; GRU runs for T steps
    with z0 repeated as input; MLP projects each hidden state to the signal.
    Implements the same forward/log_prob interface as Decoder."""

    def __init__(self, latent_dim, rnn_hidden, dec_H, data_dim, act='relu'):
        super().__init__()
        self.rnn_hidden = rnn_hidden
        self.data_dim = data_dim
        self.dec_out_dim = data_dim  # match Decoder.dec_out_dim interface
        self.w_dt = 0               # match Decoder interface (unused for VAE)
        self.distribution = 'normal'

        # Project z0 → initial GRU hidden state (mirrors encoder's rnn_hidden_to_latent)
        self.z_to_h = nn.Sequential(
            nn.Linear(latent_dim, dec_H),
            nn.ReLU(True),
            nn.Linear(dec_H, rnn_hidden),
        )
        # GRU: z0 repeated T times as input, learns temporal structure via hidden state
        self.gru = nn.GRU(latent_dim, rnn_hidden, batch_first=True)
        # MLP output head: same depth/style as existing Decoder MLP
        self.mlp = MLP(rnn_hidden, data_dim, L=2, H=dec_H, act=act)
        # Learned output std (matches Decoder.out_logsig interface for elbo log_prob)
        self.sp = nn.Softplus()
        self.out_logsig = nn.Parameter(torch.zeros(data_dim))
        self.out_logsig_dt = nn.Parameter(torch.zeros(data_dim))  # unused; kept for compat

    def forward(self, stL, _dims):
        """
        stL  : (L, N, T, q)  — z0 expanded across T (from model.py VAE branch)
        _dims: unused; kept to match Decoder.forward signature
        Returns: (L, N, T, D)
        """
        L, _, T, _ = stL.shape
        z0 = stL[:, :, 0, :]                              # (L, N, q)
        h0 = self.z_to_h(z0)                 # (L, N, rnn_hidden)
        
        outputs = []
        for l in range(L):
            inp = z0[l].unsqueeze(1).expand(-1, T, -1)    # (N, T, q)
            out, _ = self.gru(inp, h0[l].unsqueeze(0))    # (N, T, rnn_hidden)
            xrec = self.mlp(out)                           # (N, T, D)
            outputs.append(xrec)
        return torch.stack(outputs)                        # (L, N, T, D)

    @property
    def device(self):
        return next(self.parameters()).device

    def log_prob(self, X, Xhat, L=1):
        """Gaussian log-likelihood — matches Decoder.log_prob interface."""
        XL = X.repeat([L] + [1] * X.ndim)   # L, N, T, D
        Xhat = Xhat.reshape(XL.shape)
        std = self.sp(self.out_logsig)
        return torch.distributions.Normal(XL, std).log_prob(Xhat)


class Decoder(nn.Module):
    def __init__(self, task, dec_inp_dim, n_filt=8, H=100, distribution='bernoulli', dec_out_dim=None, act='relu', w_dt=0, l_w=0):
        super(Decoder, self).__init__()
        self.distribution = distribution
        self.dec_out_dim = dec_out_dim
        if task=='rot_mnist' or task=='rot_mnist_ou':
            self.net = build_rot_mnist_cnn_dec(n_filt, dec_inp_dim)
        elif task=='mov_mnist':
            self.net = build_mov_mnist_cnn_dec(n_filt, dec_inp_dim)
        elif task=='bb':
            self.net = build_rot_mnist_cnn_dec(n_filt, dec_inp_dim)
        elif task=='sin' or task=='spiral' or task=='lv' or 'mocap' in task or 'ecg' in task:
            self.net = MLP(dec_inp_dim, dec_out_dim, L=2, H=H, act=act)
            self.w_dt = w_dt
            self.l_w = l_w
            self.out_logsig = torch.nn.Parameter(torch.zeros(dec_out_dim)*0.0)
            self.out_logsig_dt = torch.nn.Parameter(torch.zeros(dec_out_dim)*0.0)
            self.sp = nn.Softplus()
        else:
            raise ValueError('Unknown task {task}')

    def forward(self, z, dims):
        inp  = z.contiguous().view([np.prod(list(z.shape[:-1])),z.shape[-1]])  # L*N*T,q  
        Xrec = self.net(inp)
        return Xrec.view(dims) # L,N,T,...
    
    @property
    def device(self):
        return next(self.parameters()).device

    def log_prob(self, X, Xhat, L=1):
        '''
        x - input [N,T,nc,d,d]   or [N,T,d]
        z - preds [L,N,T,nc,d,d] or [L,N,T,d]
        '''
        XL = X.repeat([L]+[1]*X.ndim) # L,N,T,nc,d,d or L,N,T,d
        assert XL.numel()==Xhat.numel()
        Xhat = Xhat.reshape(XL.shape)
        if self.distribution == 'bernoulli':
            try:
                log_p = torch.log(Xhat)*XL + torch.log(1-Xhat)*(1-XL) # L,N,T,nc,d,d
            except:
                log_p = torch.log(EPSILON+Xhat)*XL + torch.log(EPSILON+1-Xhat)*(1-XL) # L,N,T,nc,d,d
        elif self.distribution == 'normal':
            std = self.sp(self.out_logsig)
            #std_dt = self.sp(self.out_logsig_dt)
            log_p = torch.distributions.Normal(XL,std).log_prob(Xhat)
            #XL_dt   = torch.diff(XL, dim=2, prepend=XL[:, :, :1])
            #w = 1.0 + self.l_w * torch.abs(XL_dt)
            #log_p = log_p * w

            # if self.w_dt > 0:
            #     Xhat_dt = torch.diff(Xhat, dim=2, prepend=Xhat[:, :, :1])

            #     non_static = (XL_dt.abs() > 1e-5).float()

            #     std_dt = torch.clamp(std_dt, min=0.01, max=5.0)
            #     log_p_dt = torch.distributions.Normal(XL_dt, std_dt).log_prob(Xhat_dt) 
            #     log_p_dt = log_p_dt * non_static * self.w_dt

            #     if (log_p_dt > 0).any():
            #         print(f"Positive log_p_dt detected!")
            #         print(f"Max log_p_dt: {log_p_dt.max().item()}")
            #         print(f"std_dt range: [{std_dt.min().item()}, {std_dt.max().item()}]")
            #         print(f"Diff range: [{(XL_dt - Xhat_dt).abs().min().item()}, {(XL_dt - Xhat_dt).abs().max().item()}]")
                
        else:
            raise ValueError('Currently only bernoulli dist implemented')

        return log_p


# ---------------------------------------------------------------------------
# Sanity check: run with  python -m model.core.vae  from MoNODE/
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=== VAE-RNN sanity check ===")

    N, T, D_in, D_out = 4, 200, 12, 8
    latent_dim = 32
    rnn_hidden = 64
    dec_H = 128

    vae = VAE(
        task='ecg',
        ode_latent_dim=latent_dim,
        rnn_hidden=rnn_hidden,
        dec_H=dec_H,
        enc_H=64,
        inp_dim=D_in,
        out_dim=D_out,
        use_rnn_decoder=True,
        device='cpu',
    )

    n_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    x = torch.randn(N, T, D_in)
    mask = torch.ones(N, T, dtype=torch.bool)
    mu, log_sig = vae.encoder(x, mask=mask)
    assert mu.shape == (N, latent_dim),      f"mu shape {mu.shape}"
    assert log_sig.shape == (N, latent_dim), f"log_sig shape {log_sig.shape}"
    print(f"  Encoder  OK — mu: {tuple(mu.shape)}, log_sig: {tuple(log_sig.shape)}")

    L = 1
    z0 = vae.encoder.sample(mu, log_sig, L=L)
    z0 = z0.unsqueeze(0) if z0.ndim == 2 else z0
    stL = z0.unsqueeze(2).expand(-1, -1, T, -1)   # (L, N, T, latent_dim)
    Xrec = vae.decoder(stL, [L, N, T, D_out])
    assert Xrec.shape == (L, N, T, D_out), f"Xrec shape {Xrec.shape}"
    print(f"  Decoder  OK — Xrec: {tuple(Xrec.shape)}")

    x_out = x[:, :, :D_out]
    lp = vae.decoder.log_prob(x_out, Xrec, L=L)
    assert lp.shape == (L, N, T, D_out), f"log_prob shape {lp.shape}"
    print(f"  log_prob OK — shape: {tuple(lp.shape)}")

    # KL: log_sig here is std (softplus output), so log_var = 2*log(std)
    kl = -0.5 * (1 + 2 * torch.log(log_sig) - mu.pow(2) - log_sig.pow(2)).mean()
    assert not torch.isnan(kl), "KL is NaN"
    print(f"  KL loss  OK — value: {kl.item():.4f}")

    print("=== All checks passed ===")