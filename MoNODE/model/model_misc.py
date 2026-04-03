import os
import numpy as np
from datetime import datetime
from collections import defaultdict

import torch
from torch.distributions import kl_divergence as kl
from torch.nn.utils.rnn import pad_sequence

from model.misc import log_utils
from model.misc.plot_utils import plot_results
import wandb


# ─────────────────────────────────────────────────────────────────────────────
# Loss helpers
# ─────────────────────────────────────────────────────────────────────────────

def elbo(model, X, Xrec, s0_mu, s0_logv, v0_mu, v0_logv, L, mask=None):
    """Compute ELBO components: reconstruction log-likelihood and KL divergence.

    Args:
        model    : MoNODE model (used to access encoder.q_dist and decoder.log_prob)
        X        : ground-truth input  (N, T, D)
        Xrec     : model reconstruction (L, N, T, D)
        s0_mu    : posterior mean of initial state z0  (N, q)
        s0_logv  : posterior std  of initial state z0  (N, q)  [softplus-activated]
        v0_mu    : posterior mean of initial velocity (N, q) or None for 1st-order
        v0_logv  : posterior std  of initial velocity (N, q) or None
        L        : number of Monte-Carlo samples
        mask     : boolean validity mask (N, T); True = valid timestep

    Returns:
        lhood  : mean reconstruction log-likelihood, scalar
        kl_z0  : mean KL(q(z0) || p(z0)), scalar
    """
    # KL divergence between approximate posterior and standard normal prior
    q = model.vae.encoder.q_dist(s0_mu, s0_logv, v0_mu, v0_logv)
    kl_z0 = kl(q, model.vae.prior).sum(-1)  # (N,)

    # Reconstruction log-likelihood under the decoder distribution
    lhood = model.vae.decoder.log_prob(X, Xrec, L)  # (L, N, T, D)

    # Reduce: sum over spatial/feature dims, mean over L samples
    idx = list(np.arange(X.ndim + 1))  # [0, 1, 2, ...]
    lhood = lhood.sum(idx[2:]).mean(0)  # (N,)

    return lhood.mean(), kl_z0.mean()


def contrastive_loss(C):
    """Contrastive loss encouraging time-invariant embeddings to be consistent
    across timesteps for the same sample.

    Args:
        C : invariant embeddings (N, T, q) or (L, N, T, q)

    Returns:
        scalar loss (negative cosine similarity between time-aligned pairs)
    """
    C = C.mean(0) if C.ndim == 4 else C          # (N, T, q)
    C = C / C.pow(2).sum(-1, keepdim=True).sqrt() # L2-normalise
    N_, T_, q_ = C.shape
    C = C.reshape(N_ * T_, q_)                    # (N*T, q)
    Z = (C.unsqueeze(0) * C.unsqueeze(1)).sum(-1) # (N*T, N*T) cosine sim matrix

    # Positive pairs: timesteps from the same sequence
    idx = torch.meshgrid(torch.arange(T_), torch.arange(T_), indexing='ij')
    idxset0 = torch.cat([idx[0].reshape(-1) + n * T_ for n in range(N_)])
    idxset1 = torch.cat([idx[1].reshape(-1) + n * T_ for n in range(N_)])
    pos = Z[idxset0, idxset1].sum()
    return -pos


def compute_masked_mse(se, mask=None, dims=None):
    """Compute mean squared error with optional mask for variable-length sequences.

    Args:
        se   : squared error tensor (L, N, T, D) or similar
        mask : boolean mask (N, T); True = valid. Broadcasts over L and D.
        dims : if given, reduce only over these dimensions instead of all

    Returns:
        scalar MSE if dims is None, else a tensor reduced over the specified dims
    """
    if mask is None:
        return torch.mean(se)

    # Broadcast mask to match se shape: (1, N, T, 1, ...)
    mask = mask.unsqueeze(0).unsqueeze(-1).float()
    for _ in range(se.ndim - mask.ndim):
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(se).to(se.device)

    masked = se * mask
    if dims is None:
        return masked.sum() / (mask.sum() + 1e-8)
    return masked.sum(dim=dims) / (mask.sum(dim=dims) + 1e-8)


def compute_sobolov(X, Xrec, weight, mask=None):
    """Sobolev regularisation: penalises mismatch in temporal derivatives.

    Encourages the model to reproduce not just signal values but also their
    rate of change, which helps learn smooth dynamics.

    Args:
        X      : ground truth  (N, T, D)
        Xrec   : reconstruction (L, N, T, D)
        weight : scalar loss weight (w_dt from decoder)
        mask   : boolean mask (N, T)

    Returns:
        scalar Sobolev penalty
    """
    Xhat_dt = torch.diff(Xrec, dim=2, prepend=Xrec[:, :, :1])       # (L, N, T, D)
    X_dt = torch.diff(X, dim=1, prepend=X[:, :1, :])                 # (N, T, D)
    X_dt = X_dt.unsqueeze(0).expand_as(Xhat_dt)                      # (L, N, T, D)

    if mask is not None:
        mask_exp = mask.unsqueeze(0).unsqueeze(-1).expand_as(Xhat_dt).float().to(Xhat_dt.device)
        penalty = (torch.abs(Xhat_dt - X_dt) * mask_exp).sum(dim=(2, 3)).mean()
    else:
        penalty = torch.abs(Xhat_dt - X_dt).sum(dim=(2, 3)).mean()

    return penalty * weight


def freeze_pars(par_list):
    """Set requires_grad=False for all parameters in par_list."""
    for par in par_list:
        try:
            par.requires_grad = False
        except Exception:
            raise ValueError('Expected a Parameter but got something else.')


def compute_mse_stats(mse_list):
    """Aggregate a list of per-batch MSE tensors into mean and std arrays.

    Pads tensors to the same length (handles variable sequence lengths), then
    computes nanmean / nanstd across the batch dimension.

    Args:
        mse_list : list of 1-D tensors, one per batch

    Returns:
        mean_list : list of floats
        std_list  : list of floats
    """
    padded = pad_sequence(mse_list, batch_first=True, padding_value=float('nan'))
    padded_np = padded.detach().cpu().numpy()
    return (
        np.nanmean(padded_np, axis=0).astype(float).tolist(),
        np.nanstd(padded_np, axis=0).astype(float).tolist(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Forward-pass wrappers
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss(model, data, y, L, num_observations, mask=None, out_channels=None, calculate_y=True, beta=1.0):
    """Run one forward pass and compute the training loss for any model type.

    Args:
        model            : MoNODE model
        data             : input batch  (N, T, D)
        y                : list of (class_label, patient_id) tuples, one per sample
        L                : number of MC trajectory samples
        num_observations : total training set size (used to scale ELBO terms)
        mask             : boolean validity mask (N, T)
        out_channels     : indices of output leads (used for ECG lead subsetting)
        calculate_y      : whether to compute per-class / per-patient MSE
        beta             : KL weight for VAE training (beta-VAE)

    Returns:
        12-tuple: (loss, nll, kl_z0, Xrec, ztL, mse, c, m,
                   loss_per_class, loss_per_patient, sobolev_loss, mse_per_lead)
    """
    T = data.shape[1]
    Xrec, ztL, (s0_mu, s0_logv), (v0_mu, v0_logv), _, c, m = model(data, L, T_custom=T, mask=mask)

    if out_channels is not None:
        data = data[:, :, out_channels]

    # ── Per-class and per-patient MSE ────────────────────────────────────────
    loss_per_class = {}
    loss_per_patient = {}
    if calculate_y:
        classes  = [cls      for cls, _  in y]
        run_ids  = [run_id   for _,  run_id in y]

        for cls in set(classes):
            idx = [i for i, c_ in enumerate(classes) if c_ == cls]
            loss_per_class[cls] = compute_masked_mse(
                (Xrec[:, idx] - data[idx]) ** 2,
                mask=mask[idx] if mask is not None else None,
            ).cpu().detach().numpy()

        for run_id in set(run_ids):
            idx = [i for i, r in enumerate(run_ids) if r == run_id]
            loss_per_patient[run_id] = compute_masked_mse(
                (Xrec[:, idx] - data[idx]) ** 2,
                mask=mask[idx] if mask is not None else None,
            ).cpu().detach().numpy()

    # ── Loss by model type ───────────────────────────────────────────────────
    se           = (Xrec - data) ** 2                              # (L, N, T, D)
    mse          = compute_masked_mse(se, mask=mask)               # scalar
    mse_per_lead = compute_masked_mse(se, mask=mask, dims=(0,1,2)) # (D,)

    if model.model == 'sonode':
        return mse, 0.0, 0.0, Xrec, ztL, mse, c, m, loss_per_class, loss_per_patient, 0.0, mse_per_lead

    elif model.model == 'vae':
        lhood, kl_z0 = elbo(model, data, Xrec, s0_mu, s0_logv, None, None, L, mask=mask)
        lhood  = lhood  * num_observations
        kl_z0  = kl_z0  * num_observations
        loss   = -lhood + beta * kl_z0
        return loss, -lhood, kl_z0, Xrec, ztL, mse, c, m, loss_per_class, loss_per_patient, 0.0, mse_per_lead

    elif model.model in ('node', 'hbnode'):
        lhood, kl_z0 = elbo(model, data, Xrec, s0_mu, s0_logv, v0_mu, v0_logv, L, mask=mask)
        lhood         = lhood  * num_observations
        kl_z0         = kl_z0  * num_observations
        sobolev_loss  = compute_sobolov(data, Xrec, weight=model.vae.decoder.w_dt, mask=mask) * num_observations
        loss          = -lhood + kl_z0 + sobolev_loss
        return loss, -lhood, kl_z0, Xrec, ztL, mse, c, m, loss_per_class, loss_per_patient, sobolev_loss, mse_per_lead

    raise ValueError(f"Unknown model type: {model.model}")


def compute_mse(model, data, y_data, T_train, L=1, mask=None, task=None, out_channels=None, has_label=True):
    """Evaluate reconstruction quality at increasing prediction horizons.

    Iterates over the sequence in steps of T_train, computing MSE for each
    horizon. Also computes per-timestep and per-lead MSE for the full horizon,
    and per-class / per-patient MSE when labels are available.

    Args:
        model       : MoNODE model
        data        : input batch  (N, T, D)
        y_data      : list of (class_label, patient_id) tuples
        T_train     : step size for curriculum evaluation
        L           : MC samples (typically 1 at eval time)
        mask        : boolean validity mask (N, T)
        task        : task name (used only for rot_mnist sliding-window logic)
        out_channels: output lead indices for ECG subsetting
        has_label   : whether y_data contains meaningful class/patient labels

    Returns:
        dict_mse        : {horizon_str: scalar MSE}
        dict_misc       : {'mse_t': (L,N,D) tensor, 'mse_l': (L,N,T) tensor}
        loss_per_class  : {class: scalar MSE}
        loss_per_patient: {patient_id: scalar MSE}
    """
    T = data.shape[1]
    Xrec, _, _, _, _, _, _ = model(data, L, T, mask=mask)

    if out_channels is not None:
        data = data[:, :, out_channels]

    dict_mse       = {}
    dict_misc      = {}
    loss_per_class  = {}
    loss_per_patient = {}

    T_start, T_max = 0, 0
    while T_max < T:
        if task == 'rot_mnist':
            # Sliding window: each window is one T_train segment
            T_max += T_train
            mse = compute_masked_mse(
                (Xrec[:, :, T_start:T_max] - data[:, T_start:T_max]) ** 2, mask=mask
            )
            dict_mse[str(T_max)] = mse
            T_start += T_train
            T_max   += T_train
        else:
            # Expanding horizon: evaluate from 0 to T_max
            T_max += T_train
            se = (Xrec[:, :, :T_max] - data[:, :T_max]) ** 2
            sliced_mask = mask[:, :T_max] if mask is not None else None
            dict_mse[str(T_max)] = compute_masked_mse(se, mask=sliced_mask)

            if T_max >= T:
                # Full-horizon breakdown: per-timestep and per-lead
                dict_misc['mse_t'] = compute_masked_mse(se, mask=sliced_mask, dims=(1, 3)).squeeze(0)
                dict_misc['mse_l'] = compute_masked_mse(se, mask=sliced_mask, dims=(1, 2)).squeeze(0)

                if has_label:
                    classes     = [cls       for cls, _    in y_data]
                    patient_ids = [patient   for _,   patient in y_data]

                    for cls in set(classes):
                        idx = [i for i, (c_, _) in enumerate(y_data) if c_ == cls]
                        loss_per_class[cls] = compute_masked_mse(
                            (Xrec[:, idx] - data[idx]) ** 2,
                            mask=mask[idx] if mask is not None else None,
                        ).item()

                    for pid in set(patient_ids):
                        idx = [i for i, p in enumerate(patient_ids) if p == pid]
                        loss_per_patient[pid] = compute_masked_mse(
                            (Xrec[:, idx] - data[idx]) ** 2,
                            mask=mask[idx] if mask is not None else None,
                        ).item()

    return dict_mse, dict_misc, loss_per_class, loss_per_patient


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_split(model, dataloader, T_train, out_channels, has_label, task=None):
    """Run evaluation on an entire dataloader and aggregate metrics across batches.

    Args:
        model       : MoNODE model (in eval / no_grad context expected by caller)
        dataloader  : torch DataLoader for validation or test set
        T_train     : sequence length step for horizon evaluation
        out_channels: output lead indices for ECG subsetting
        has_label   : whether to compute per-class / per-patient MSE
        task        : task name (passed through to compute_mse)

    Returns:
        dict_mses       : {horizon_str: [batch_scalars]}
        mse_t_batches   : list of per-timestep MSE tensors (one per batch)
        mse_l_batches   : list of per-lead MSE tensors (one per batch)
        loss_per_class  : defaultdict(list) of per-class MSE values
        loss_per_patient: defaultdict(list) of per-patient MSE values
    """
    dict_mses        = {}
    mse_t_batches    = []
    mse_l_batches    = []
    loss_per_class   = defaultdict(list)
    loss_per_patient = defaultdict(list)

    for batch, y, mask in dataloader:
        batch = batch.to(model.device)
        d_mse, d_misc, cls_mse, pat_mse = compute_mse(
            model, batch, y, T_train,
            mask=mask, task=task, out_channels=out_channels, has_label=has_label,
        )
        for horizon, val in d_mse.items():
            dict_mses.setdefault(horizon, []).append(val.item())

        mse_t_batches.append(d_misc['mse_t'])
        mse_l_batches.append(d_misc['mse_l'])

        for cls, v in cls_mse.items():
            loss_per_class[cls].append(v)
        for pid, v in pat_mse.items():
            loss_per_patient[pid].append(v)

    return dict_mses, mse_t_batches, mse_l_batches, loss_per_class, loss_per_patient


def _log_split_metrics(run, prefix, dict_mses, mse_t_batches, mse_l_batches,
                        loss_per_class, loss_per_patient, custom_channel, has_label, logger=None):
    """Log evaluation metrics for one data split (val or test) to wandb.

    Args:
        run             : wandb run object
        prefix          : 'val' or 'test'
        dict_mses       : {horizon_str: [batch_scalars]}
        mse_t_batches   : list of per-timestep MSE tensors
        mse_l_batches   : list of per-lead MSE tensors
        loss_per_class  : defaultdict(list) of per-class MSE
        loss_per_patient: defaultdict(list) of per-patient MSE
        custom_channel  : lead name labels or None
        has_label       : whether class/patient metrics exist
        logger          : optional Python logger for console output

    Returns:
        mse_rec : MSE at the first (reconstruction) horizon — used for early stopping
    """
    horizons   = list(dict_mses.keys())
    mse_rec    = np.mean(dict_mses[horizons[0]])
    mse_for    = np.mean(dict_mses[horizons[-1]])

    if logger:
        for h, vals in dict_mses.items():
            logger.info(f'  T={h} {prefix}_mse {np.mean(vals):.3f} (±{np.std(vals):.3f})')

    # Per-timestep MSE band
    mse_t, sse_t = compute_mse_stats(mse_t_batches)
    t = np.arange(len(mse_t))
    table = wandb.Table(
        data=[[ti, m, m - s, m + s] for ti, m, s in zip(t, mse_t, sse_t)],
        columns=["timestep", "mean", "lower", "upper"],
    )
    run.log({
        f"{prefix}/mse_t": wandb.plot_table(
            vega_spec_name="tymechatu-university-of-amsterdam/std_band_custom",
            data_table=table,
            fields={"x": "timestep", "y": "mean", "lower_bound": "lower", "upper_bound": "upper"},
        )
    })

    # Per-lead MSE bar chart
    mse_l, sse_l = compute_mse_stats(mse_l_batches)
    channel = custom_channel if custom_channel else list(np.arange(len(mse_l)))
    table = wandb.Table(
        data=[[ch, m, m - s, m + s] for ch, m, s in zip(channel, mse_l, sse_l)],
        columns=["lead", "mean", "lower", "upper"],
    )
    run.log({
        f"{prefix}/mse_by_lead": wandb.plot.bar(table, "lead", "mean", title="MSE per Lead")
    })

    # Scalar MSE for this split
    run.log({f"{prefix}/mse": mse_rec})

    if has_label:
        table = wandb.Table(
            data=[[cls, np.mean(vals)] for cls, vals in loss_per_class.items()],
            columns=["class", "mse"],
        )
        run.log({f"{prefix}/mse_per_class": wandb.plot.bar(table, "class", "mse", title="MSE per class")})

        table = wandb.Table(
            data=[[pid, np.mean(vals)] for pid, vals in loss_per_patient.items()],
            columns=["patient_id", "mse"],
        )
        run.log({f"{prefix}/mse_per_patient": wandb.plot.bar(table, "patient_id", "mse", title="MSE per patient")})

    return mse_rec, mse_for


def log_gradients(model, run):
    """Log L2 gradient norms for encoder, ODE, and decoder to wandb.

    Handles the VAE case where model.flow is None (no ODE gradient to log).

    Args:
        model : MoNODE model (after loss.backward())
        run   : wandb run object
    """
    def _grad_norm(params):
        return sum(
            p.grad.data.norm(2).item()
            for p in params if p.grad is not None
        )

    norm_enc = _grad_norm(model.vae.encoder.parameters())
    norm_dec = _grad_norm(model.vae.decoder.parameters())
    norm_ode = _grad_norm(model.flow.odefunc.parameters()) if model.flow is not None else 0.0

    run.log({
        "grads/gru_norm":        norm_enc,
        "grads/ode_norm":        norm_ode,
        "grads/dec_norm":        norm_dec,
        "grads/ratio_gru_to_dec": norm_enc / (norm_dec + 1e-8),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_model(args, model, plotter, trainset, validset, testset, logger, params, run, freeze_dyn=False):
    """Full training loop with curriculum learning, validation, and test evaluation.

    Training is structured in three phases:
      1. Inner batch loop  — forward pass, loss, backward, optimizer step.
      2. Epoch end         — validation evaluation and wandb logging.
      3. Best-model update — test evaluation whenever validation improves.

    Args:
        args       : argparse namespace (hyperparameters and task config)
        model      : MoNODE model
        plotter    : Plotter instance for periodic reconstruction visualisation
        trainset   : DataLoader for the training set
        validset   : DataLoader for the validation set
        testset    : DataLoader for the test set
        logger     : Python logger
        params     : dataset config dict (from config.yml)
        run        : wandb run object
        freeze_dyn : if True, freeze the ODE dynamics (flow) parameters
    """
    # ── Metric meters ────────────────────────────────────────────────────────
    loss_meter   = log_utils.CachedRunningAverageMeter(0.97)
    tr_mse_meter = log_utils.CachedRunningAverageMeter(0.97)
    vl_mse_rec   = log_utils.CachedRunningAverageMeter(0.97)
    vl_mse_for   = log_utils.CachedRunningAverageMeter(0.97)
    time_meter   = log_utils.CachedRunningAverageMeter(0.97)

    if args.model in ('node', 'hbnode'):
        nll_meter   = log_utils.CachedRunningAverageMeter(0.97)
        kl_z0_meter = log_utils.CachedRunningAverageMeter(0.97)

    logger.info('********** Started Training **********')

    if freeze_dyn:
        freeze_pars(model.flow.parameters())

    # ── Optimiser ────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── Curriculum: gradually increase sequence length T_ over training ──────
    T_train   = params['train']['T']
    if args.task == 'rot_mnist':
        T_train = params['train']['T'] - 1  # one window per pass
    ep_inc_c  = args.Nepoch // args.Nincr   # epochs between T increases
    ep_inc_v  = T_train // args.Nincr       # timesteps added per increment
    T_        = ep_inc_v                    # current training horizon

    # ── ECG-specific lead configuration ──────────────────────────────────────
    has_label       = params['dataset'].lower() == 'medalcare-xl'
    custom_channel  = None
    out_ecg_lead_idx = None

    if args.task == 'ecg':
        lead_idx = {
            'I': 0, 'II': 1, 'III': 2, 'aVR': 3, 'aVL': 4, 'aVF': 5,
            'V1': 6, 'V2': 7, 'V3': 8, 'V4': 9, 'V5': 10, 'V6': 11,
        }
        custom_channel   = [l for l in lead_idx if l not in params['exclude_leads_out']]
        out_ecg_lead_idx = [lead_idx[l] for l in lead_idx if l not in params['exclude_leads_out']]

    print(
        f'Training set size: {len(trainset.dataset.file_paths)}, '
        f'Validation set size: {len(validset.dataset.file_paths)}, '
        f'Test set size: {len(testset.dataset.file_paths)}'
    )

    # ── Epoch loop ───────────────────────────────────────────────────────────
    start_time     = datetime.now()
    global_itr     = 0
    best_valid_loss = 1e9
    dict_test_mses = {}  # populated when best model is found

    for ep in range(args.Nepoch):
        # Curriculum: increase horizon every ep_inc_c epochs
        if ep != 0 and ep % ep_inc_c == 0:
            T_ += ep_inc_v

        # VAE has no stochastic latent trajectory — use L=1 always
        L = 1 if args.model == 'sonode' else (1 if ep < args.Nepoch // 2 else 2)

        # Per-epoch accumulators for wandb logging
        loss_per_class   = defaultdict(list)
        loss_per_patient = defaultdict(list)
        mse_per_lead_accum = []

        # ── Inner batch loop ─────────────────────────────────────────────────
        for itr, (local_batch, local_y, local_mask) in enumerate(trainset):
            tr_minibatch = local_batch.to(model.device)
            if itr == 0:
                print(f"Train batch shape: {tr_minibatch.shape}")

            # Curriculum sub-sampling for non-image tasks
            if args.task in ('sin', 'spiral', 'lv') or 'mocap' in args.task:
                N, T = tr_minibatch.shape[:2]
                N_   = int(N * (T // T_))
                t0s  = torch.randint(0, max(T - T_, 1), [N_]) if T_ < T else torch.zeros([N_], dtype=torch.int)
                tr_minibatch = tr_minibatch.repeat([N_, 1, 1])
                tr_minibatch = torch.stack([tr_minibatch[n, t0:t0 + T_] for n, t0 in enumerate(t0s)])

            loss, nlhood, kl_z0, _, _, tr_mse, _, _, _cls_mse, _pat_mse, sobolev_loss, _mse_per_lead = compute_loss(
                model, tr_minibatch, local_y, L,
                mask=local_mask,
                num_observations=len(trainset.dataset.file_paths),
                out_channels=out_ecg_lead_idx,
                calculate_y=has_label,
                beta=getattr(args, 'beta', 1.0),
            )

            optimizer.zero_grad()
            loss.backward()
            log_gradients(model, run)
            optimizer.step()

            # Update running meters
            loss_meter.update(loss.item(), global_itr)
            tr_mse_meter.update(tr_mse.item(), global_itr)
            if args.model in ('node', 'hbnode'):
                nll_meter.update(float(nlhood), global_itr)
                kl_z0_meter.update(float(kl_z0), global_itr)
            global_itr += 1

            if has_label:
                for cls, v in _cls_mse.items():
                    loss_per_class[cls].append(v)
                for pid, v in _pat_mse.items():
                    loss_per_patient[pid].append(v)

            mse_per_lead_accum.append(_mse_per_lead.detach().cpu().numpy())

            run.log({
                'train/tr_loss':            loss.item(),
                'train/tr_loss_per_sample': loss.item() / len(trainset.dataset.file_paths),
                'train/mse':                tr_mse.item(),
                'train/nll':                float(nlhood),
                'train/kl_z0':              float(kl_z0),
                'train/nll_dt':             float(sobolev_loss),
                'train/nll_dt_ratio':       float(sobolev_loss) / (float(nlhood) + 1e-8),
            })

        # ── End-of-epoch: log per-class / per-lead training metrics ──────────
        if has_label:
            run.log({
                "train/mse_per_class": wandb.plot.bar(
                    wandb.Table(
                        data=[[cls, np.mean(v)] for cls, v in loss_per_class.items()],
                        columns=["class", "mse"],
                    ),
                    "class", "mse", title="Train MSE per class",
                ),
                "train/mse_per_patient": wandb.plot.bar(
                    wandb.Table(
                        data=[[pid, np.mean(v)] for pid, v in loss_per_patient.items()],
                        columns=["patient_id", "mse"],
                    ),
                    "patient_id", "mse", title="Train MSE per patient",
                ),
            })

        if mse_per_lead_accum:
            mean_lead = np.stack(mse_per_lead_accum).mean(axis=0).squeeze()  # (D,)
            lead_labels = custom_channel if custom_channel else list(range(len(mean_lead)))
            run.log({
                "train/mse_per_lead": wandb.plot.bar(
                    wandb.Table(
                        data=[[str(l), v] for l, v in zip(lead_labels, mean_lead.tolist())],
                        columns=["lead", "mse"],
                    ),
                    "lead", "mse", title="Train MSE per lead",
                )
            })

        # ── Validation ───────────────────────────────────────────────────────
        with torch.no_grad():
            val_results = _evaluate_split(
                model, validset, T_train, out_ecg_lead_idx, has_label
            )
            valid_mse_rec, valid_mse_for = _log_split_metrics(
                run, 'val', *val_results, custom_channel, has_label, logger=None
            )

            run.log({'val/tr_loss': loss_meter.val, 'val/T_rec': valid_mse_rec})
            vl_mse_rec.update(valid_mse_rec, ep)
            vl_mse_for.update(valid_mse_for, ep)
            time_meter.update((datetime.now() - start_time).seconds, ep)

            logger.info(
                f'Epoch:{ep:4d}/{args.Nepoch:4d} | '
                f'tr_loss:{loss_meter.val:8.2f}({loss_meter.avg:8.2f}) | '
                f'valid_mse_rec:{valid_mse_rec:5.3f} | valid_mse_for:{valid_mse_for:5.3f}'
            )

            # ── Test evaluation on best model ─────────────────────────────
            if valid_mse_rec < best_valid_loss:
                best_valid_loss = valid_mse_rec

                torch.save(
                    {'args': args, 'state_dict': model.state_dict()},
                    os.path.join(args.save, 'model.pth'),
                )

                logger.info('********** New best model — evaluating on test set **********')
                logger.info(f'Epoch:{ep:4d}/{args.Nepoch:4d}')

                test_results = _evaluate_split(
                    model, testset, T_train, out_ecg_lead_idx, has_label, task=args.task
                )
                dict_test_mses = test_results[0]
                _log_split_metrics(
                    run, 'test', *test_results, custom_channel, has_label, logger=logger
                )

        # ── Periodic visualisation ────────────────────────────────────────────
        if ep % args.plot_every == 0 or (ep + 1) == args.Nepoch:
            plot_tr_batch    = trainset.dataset.get_class_samples(k=1) if has_label else {0: tr_minibatch}
            plot_valid_batch = validset.dataset.get_class_samples(k=1) if has_label else {0: local_batch.to(model.device)}

            train_plot_dict = {}
            valid_plot_dict = {}

            for cls, batch in plot_tr_batch.items():
                Xrec, ztL, _, _, C, _, _ = model(batch.to(model.device), L=args.plotL)
                train_plot_dict[cls] = {
                    'Xrec': Xrec.cpu().detach(), 'ztL': ztL.cpu().detach(),
                    'C': C.cpu().detach() if C is not None else None,
                    'batch': batch.cpu().detach(),
                }

            for cls, batch in plot_valid_batch.items():
                Xrec, ztL, _, _, C, _, _ = model(batch.to(model.device), L=args.plotL)
                valid_plot_dict[cls] = {
                    'Xrec': Xrec.cpu().detach(), 'ztL': ztL.cpu().detach(),
                    'C': C.cpu().detach() if C is not None else None,
                    'batch': batch.cpu().detach(),
                }

            plot_config = {}
            if args.task == 'ecg':
                plot_config = {
                    'exclude_leads': params['exclude_leads_out'],
                    'f': params['f'],
                    'run': run,
                }

            for cls in train_plot_dict:
                try:
                    if args.model in ('node', 'hbnode'):
                        plot_results(
                            plotter,
                            train_plot_dict[cls]['Xrec'], train_plot_dict[cls]['batch'],
                            valid_plot_dict[cls]['Xrec'], valid_plot_dict[cls]['batch'],
                            {"plot": {'Loss(-elbo)': loss_meter, 'Nll': nll_meter, 'KL-z0': kl_z0_meter, "train-MSE": tr_mse_meter},
                             "valid-MSE-rec": vl_mse_rec, "valid-MSE-for": vl_mse_for,
                             "iteration": ep, "time": time_meter},
                            train_plot_dict[cls]['ztL'], valid_plot_dict[cls]['ztL'],
                            train_plot_dict[cls]['C'],   valid_plot_dict[cls]['C'],
                            tr_fname=f'tr_{cls}', val_fname=f'val_{cls}', **plot_config,
                        )
                    elif args.model in ('sonode', 'vae'):
                        plot_results(
                            plotter,
                            train_plot_dict[cls]['Xrec'], train_plot_dict[cls]['batch'],
                            valid_plot_dict[cls]['Xrec'], valid_plot_dict[cls]['batch'],
                            {"plot": {"Loss": loss_meter, "valid-MSE-rec": vl_mse_rec, "valid-MSE-for": vl_mse_for},
                             "time": time_meter, "iteration": ep},
                            **plot_config,
                        )
                except Exception as e:
                    logger.error(f"Plotting error for class {cls}: {e}")

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info(
        f'Epoch:{ep:4d}/{args.Nepoch:4d} | time:{datetime.now() - start_time} | '
        f'train_loss:{loss_meter.val:8.2f} | train_mse:{tr_mse_meter.avg:5.3f} | '
        f'valid_mse_rec:{vl_mse_rec.val:5.3f} | valid_mse_for:{vl_mse_for.val:5.3f} | '
        f'best_valid_mse:{best_valid_loss:5.3f}'
    )
    for h, vals in dict_test_mses.items():
        logger.info(f'T={h} test_mse {np.mean(vals):.3f} (±{np.std(vals):.3f})')
