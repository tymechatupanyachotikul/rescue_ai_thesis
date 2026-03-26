import os, numpy as np
from datetime import datetime
import torch
from torch.distributions import kl_divergence as kl
from torch.nn.utils.rnn import pad_sequence

from model.misc import log_utils 
from model.misc.plot_utils import plot_results
import wandb
from collections import defaultdict

def elbo(model, X, Xrec, s0_mu, s0_logv, v0_mu, v0_logv,L, mask=None):
    ''' Input:
            qz_m        - latent means [N,2q]
            qz_logv     - latent logvars [N,2q]
            X           - input images [L,N,T,nc,d,d]
            Xrec        - reconstructions [L,N,T,nc,d,d]
        Returns:
            likelihood
            kl terms
    '''
    # KL reg
    q = model.vae.encoder.q_dist(s0_mu, s0_logv, v0_mu, v0_logv)
    kl_z0 = kl(q, model.vae.prior).sum(-1) #N

    #Reconstruction log-likelihood
    lhood, lhood_v = model.vae.decoder.log_prob(X,Xrec,L) #L,N,T,d,nc,nc

    if mask is not None:
        mask_exp = mask.unsqueeze(0).to(lhood.device)                             # [1,N,T]
        for _ in range(lhood.ndim - mask_exp.ndim):
            mask_exp = mask_exp.unsqueeze(-1)                      # [1,N,T,1,...]
        mask_exp = mask_exp.expand_as(lhood).float()               # [L,N,T,...]

        lhood   = lhood   * mask_exp
        if lhood_v is not None:
            mask_exp_v = mask.unsqueeze(0).to(lhood_v.device)
            for _ in range(lhood_v.ndim - mask_exp_v.ndim):
                mask_exp_v = mask_exp_v.unsqueeze(-1)
            lhood_v = lhood_v * mask_exp_v.expand_as(lhood_v).float()

        idx      = list(np.arange(X.ndim + 1))                    # [0,1,2,...]
        lhood    = lhood.sum(idx[2:])                              # [L,N]  summed over T,...
        lhood    = lhood.mean(0)                                   # [N]    mean over L

        lhood_v = lhood_v.sum(idx[2:]).mean(0) if lhood_v is not None else torch.zeros_like(lhood)    # [N]

    else:
        idx   = list(np.arange(X.ndim+1)) # 0,1,2,...
        lhood = lhood.sum(idx[2:]).mean(0) #N
        lhood_v = lhood_v.sum(idx[2:]).mean(0) if lhood_v is not None else torch.zeros_like(lhood)

    return lhood.mean(), lhood_v.mean(), kl_z0.mean() 


def contrastive_loss(C):
    ''' 
    C - invariant embeddings [N,T,q] or [L,N,T,q] 
    '''
    C = C.mean(0) if C.ndim==4 else C
    C = C / C.pow(2).sum(-1,keepdim=True).sqrt() # N,Tinv,q
    N_,T_,q_ = C.shape
    C = C.reshape(N_*T_,q_) # NT,q
    Z   = (C.unsqueeze(0) * C.unsqueeze(1)).sum(-1) # NT, NT
    idx = torch.meshgrid(torch.arange(T_),torch.arange(T_))
    idxset0 = torch.cat([idx[0].reshape(-1)+ n*T_ for n in range(N_)])
    idxset1 = torch.cat([idx[1].reshape(-1)+ n*T_ for n in range(N_)])
    pos = Z[idxset0,idxset1].sum()
    return -pos

def compute_masked_mse(se, mask=None,dims=None):

    if mask is None:
        mse = torch.mean(se)
    else:
        mask = mask.unsqueeze(0).unsqueeze(-1).float() 
        for _ in range(se.ndim - mask.ndim):
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(se).to(se.device)
        diff = se * mask

        if dims is None:
            return diff.sum() / (mask.sum() + 1e-8)
        else:
            return diff.sum(dim=dims) / (mask.sum(dim=dims) + 1e-8)

    return mse 

def compute_mse(model, data, y_data, T_train, L=1, mask=None, task=None):

    T_start = 0
    T_max = 0
    T = data.shape[1]
    #run model    
    Xrec, ztL, (s0_mu, s0_logv), (v0_mu, v0_logv), C, c, m = model(data, L, T, mask=mask)
    
    dict_mse = {}
    dict_misc = {}
    while T_max < T:
        if task == 'rot_mnist':
            T_max += T_train
            mse = compute_masked_mse((Xrec[:,:,T_start:T_max]-data[:,T_start:T_max])**2, mask=mask)
            dict_mse[str(T_max)] = mse
            T_start += T_train 
            T_max += T_train
        else:
            T_max += T_train
            se = (Xrec[:,:,:T_max]-data[:,:T_max])**2
            mse = compute_masked_mse(se, mask=mask[:, :T_max])
            dict_mse[str(T_max)] = mse
            if T_max >= T:
                mse_T = compute_masked_mse(se,  mask=mask[:, :T_max], dims=(1, 3))

                mse_l = compute_masked_mse(se, mask=mask[:, :T_max], dims=(1, 2))
                #dict_misc['mse_dt'] = ((torch.diff(Xrec[:,:,:T_max], dim=2)-torch.diff(data[:,:T_max], dim=1))**2).mean(dim=(-1, 1))[0]
                dict_misc['mse_t'] = mse_T.squeeze(0)
                dict_misc['mse_l'] = mse_l.squeeze(0)

                loss_per_class = {}
                loss_per_patient = {}

                classes = [] 
                patient_ids = [] 

                for (_cls, patient_id) in y_data:
                    classes.append(_cls)
                    patient_ids.append(patient_id)

                for cls in list(set(classes)):
                    idx = [i for i, val in enumerate(y_data) if val[0] == cls]
                    loss_per_class[cls] = torch.mean((Xrec[:, idx, :, :] - data[idx, :, :])**2).item()
                
                for patient_id in list(set(patient_ids)):
                    idx = [i for i, val in enumerate(patient_ids) if val[1] == patient_id]
                    loss_per_patient[patient_id] = torch.mean((Xrec[:, idx, :, :] - data[idx, :, :])**2).item()


    return dict_mse, dict_misc, loss_per_class, loss_per_patient


def compute_loss(model, data, y, L, num_observations, mask=None):
    """
    Compute loss for optimization
    @param model: mo/node  
    @param data: true observation sequence 
    @param L: number of MC samples
    @return: loss, nll, regularizing_kl, inducing_kl
    """
    T = data.shape[1]

    #run model    
    Xrec, ztL, (s0_mu, s0_logv), (v0_mu, v0_logv), C, c, m = model(data, L, T_custom=T, mask=mask)

    loss_per_class = {} 
    loss_per_patient = {}
    
    classes = []
    run_ids = []

    for (cls, run_id) in y:
        classes.append(cls)
        run_ids.append(run_id)

    for cls in list(set(classes)):
        idx = [i for i, val in enumerate(classes) if val == cls]
        loss_per_class[cls] = compute_masked_mse((Xrec[:, idx, :, :] - data[idx, :, :])**2,  mask=mask[idx]).cpu().detach().numpy()

    for run_id in list(set(run_ids)):
        idx = [i for i, val in enumerate(run_ids) if val == run_id]
        loss_per_patient[run_id] = compute_masked_mse((Xrec[:, idx, :, :] - data[idx, :, :])**2,  mask=mask[idx]).cpu().detach().numpy()

    #compute loss
    if model.model =='sonode':
        mse = compute_masked_mse((Xrec-data)**2, mask=mask)

        loss = mse 
        return loss, 0.0, 0.0, Xrec, ztL, mse, c, m
    
    elif model.model =='node' or model.model == 'hbnode':
        lhood, lhood_dt, kl_z0 = elbo(model, data, Xrec, s0_mu, s0_logv, v0_mu, v0_logv,L, mask=mask)
        
        lhood = (lhood + lhood_dt) * num_observations
        kl_z0 = kl_z0 * num_observations
        loss  = - lhood + kl_z0
        mse   = compute_masked_mse((Xrec-data)**2, mask=mask)
        return loss, -lhood, kl_z0, Xrec, ztL, mse, c, m, loss_per_class, loss_per_patient, -lhood_dt * num_observations
    

def freeze_pars(par_list):
    for par in par_list:
        try:
            par.requires_grad = False
        except:
            print('something wrong!')
            raise ValueError('This is not a parameter!?')

def compute_mse_stats(mse_list):
    padded_tensor = pad_sequence(mse_list, batch_first=True, padding_value=float('nan'))
    padded_np = padded_tensor.detach().cpu().numpy()
    
    mean_list = np.nanmean(padded_np, axis=0).astype(float).tolist()
    std_list = np.nanstd(padded_np, axis=0).astype(float).tolist()
    
    return mean_list, std_list

def train_model(args, model, plotter, trainset, validset, testset, logger, params, run, freeze_dyn=False):

    loss_meter  = log_utils.CachedRunningAverageMeter(0.97)
    tr_mse_meter   = log_utils.CachedRunningAverageMeter(0.97)
    vl_mse_rec  = log_utils.CachedRunningAverageMeter(0.97)
    vl_mse_for = log_utils.CachedRunningAverageMeter(0.97)
    time_meter = log_utils.CachedRunningAverageMeter(0.97)

    if args.model == 'node' or args.model=='hbnode':
        nll_meter   = log_utils.CachedRunningAverageMeter(0.97)
        kl_z0_meter = log_utils.CachedRunningAverageMeter(0.97)
        

    logger.info('********** Started Training **********')
    if freeze_dyn:
        freeze_pars(model.flow.parameters())

    ############## build the optimizer ############
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ########## Training loop ###########
    start_time=datetime.now()
    global_itr = 0
    best_valid_loss = None
    test_mse = 0.0
    #increase the data set length in N increments sequentally 
    T_train = params['train']['T']
    if args.task == 'rot_mnist':
        T_train = params['train']['T'] - 1 #compute mse for one loop
    ep_inc_c = args.Nepoch // args.Nincr
    ep_inc_v = T_train // args.Nincr
    T_ = ep_inc_v
    best_valid_loss = 1e9

    custom_channel = None
    if args.task == 'ecg':
        exclude_leads = params['exclude_leads']
        lead_idx = {
            'I': 0, 
            'II': 1,
            'III': 2, 
            'aVR': 3, 
            'aVL': 4,
            'aVF': 5,
            'V1': 6,
            'V2': 7,
            'V3': 8, 
            'V4': 9, 
            'V5': 10, 
            'V6': 11
        }

        custom_channel = [lead for lead, idx in lead_idx.items() if lead not in exclude_leads]

    for ep in range(args.Nepoch):
        #no latent space to sample from
        if args.model == 'sonode':
            L=1
        else:
            L = 1 if ep<args.Nepoch//2 else 2

        if (ep != 0) and (ep % ep_inc_c == 0):
            T_ += ep_inc_v
        loss_per_class = defaultdict(list)
        loss_per_patient = defaultdict(list)

        for itr, (local_batch, local_y, local_mask) in enumerate(trainset):
            tr_minibatch = local_batch.to(model.device) # N,T,...
            if args.task=='sin' or args.task=='spiral' or args.task=='lv' or 'mocap' in args.task: #slowly increase sequence length
                [N,T] = tr_minibatch.shape[:2]

                N_  = int(N*(T//T_))
                if T_ < T:
                    t0s = torch.randint(0,T-T_,[N_])  #select a random initial point from the sequence
                else:
                    t0s = torch.zeros([N_]).to(int)
                tr_minibatch = tr_minibatch.repeat([N_,1,1])
                tr_minibatch = torch.stack([tr_minibatch[n,t0:t0+T_] for n,t0 in enumerate(t0s)]) # N*ns,T//2,d
                
            loss, nlhood, kl_z0, Xrec_tr, ztL_tr, tr_mse, _, _, _loss_per_class, _loss_per_patient, nlhood_dt= compute_loss(model, tr_minibatch, local_y, L, mask=local_mask, num_observations = len(trainset.dataset.file_paths))

            optimizer.zero_grad()
            loss.backward() 
            log_gradients(model, run)

            optimizer.step()

            #store values 
            loss_meter.update(loss.item(), global_itr)
            tr_mse_meter.update(tr_mse.item(), global_itr)
            if args.model == 'node' or args.model=='hbnode':
                nll_meter.update(nlhood.item(), global_itr)
                kl_z0_meter.update(kl_z0.item(), global_itr)
            global_itr +=1

            for _cls, cls_mse in _loss_per_class.items():
                loss_per_class[_cls].append(cls_mse)

            for run_id, patient_mse in _loss_per_patient.items():
                loss_per_patient[run_id].append(patient_mse)

            time_val = datetime.now()-start_time
            run.log({
                'train/tr_loss': loss.item(),
                'train/tr_loss_per_sample': loss.item() / len(trainset.dataset.file_paths),
                'train/mse': tr_mse.item(),
                'train/nll': nlhood.item(),
                'train/kl_z0': kl_z0.item(),
                'train/nll_dt': nlhood_dt.item(), 
                'train/nll_dt_ratio': nlhood_dt.item()/nlhood.item()
            })
        
        table = wandb.Table(data=[[_cls, np.mean(cls_loss)] for _cls, cls_loss in loss_per_class.items()],
                                columns=["class", "mse"])

        run.log({
            "train/mse_per_class": wandb.plot.bar(
                table, 
                "class", 
                "mse", 
                title="MSE per class"
            )
        })

        table = wandb.Table(data=[[patient_id, np.mean(patient_loss)] for patient_id, patient_loss in loss_per_patient.items()],
                                columns=["patient_id", "mse"])

        run.log({
            "train/mse_per_patient": wandb.plot.bar(
                table, 
                "patient_id", 
                "mse", 
                title="MSE per patient"
            )
        })

        with torch.no_grad():
            
            dict_valid_mses = {}
            dict_valid_misc = {
                'mse_t': [],
                'mse_l': [],
                #'mse_dt': [],
            }
            val_loss_per_class = defaultdict(list)
            val_loss_per_patient = defaultdict(list)

            for valid_batch, valid_y, valid_mask in validset:
                valid_batch = valid_batch.to(model.device)
                dict_mse, dict_misc, _loss_per_class, _loss_per_patient = compute_mse(model, valid_batch, valid_y, T_train, mask=valid_mask)
                for key,val in dict_mse.items():
                    if key not in dict_valid_mses:
                        dict_valid_mses[key] = []
                    dict_valid_mses[key].append(val.item())
                for cls, cls_mse in _loss_per_class.items():
                    val_loss_per_class[cls].append(cls_mse)
                for patient_id, patient_mse in _loss_per_patient.items():
                    val_loss_per_patient[patient_id].append(patient_mse)

                dict_valid_misc['mse_t'].append(dict_misc['mse_t'])
                dict_valid_misc['mse_l'].append(dict_misc['mse_l'])
                #dict_valid_misc['mse_dt'].append(dict_misc['mse_dt'])
                

            T_rec = list(dict_valid_mses.keys())[0]
            T_for  = list(dict_valid_mses.keys())[-1]
            valid_mse_rec = np.mean(dict_valid_mses[T_rec])
            valid_mse_for = np.mean(dict_valid_mses[T_for])

            mse_t, sse_t = compute_mse_stats(dict_valid_misc['mse_t'])
            mse_l, sse_l = compute_mse_stats(dict_valid_misc['mse_l'])
            #mse_dt, sse_dt = compute_mse_stats(dict_valid_misc['mse_dt'])

            t = np.arange(len(mse_t))
            table = wandb.Table(data=[[ti, m, m-s, m+s] for ti, m, s in zip(t, mse_t, sse_t)],
                                columns=["timestep", "mean", "lower", "upper"])

            run.log({
                "val/mse_t": wandb.plot_table(
                    vega_spec_name="tymechatu-university-of-amsterdam/std_band_custom", 
                    data_table=table,
                    fields={
                        "x": "timestep",
                        "y": "mean",
                        "lower": "lower",
                        "upper": "upper"
                    }
                )
            })

            # t = np.arange(len(mse_dt))
            # table = wandb.Table(data=[[ti, m, m-s, m+s] for ti, m, s in zip(t, mse_dt, sse_dt)],
            #                     columns=["timestep", "mean", "lower", "upper"])

            # run.log({
            #     "val/mse_dt": wandb.plot_table(
            #         vega_spec_name="tymechatu-university-of-amsterdam/std_band_custom", 
            #         data_table=table,
            #         fields={
            #             "x": "timestep",
            #             "y": "mean",
            #             "lower_bound": "lower",
            #             "upper_bound": "upper"
            #         }
            #     )
            # })
            
            channel = np.arange(len(mse_t)) if not custom_channel else custom_channel
            table = wandb.Table(data=[[ti, m, m-s, m+s] for ti, m, s in zip(channel, mse_l, sse_l)],
                                columns=["lead", "mean", "lower", "upper"])

            run.log({
                "val/mse_by_lead": wandb.plot.bar(
                    table, 
                    "lead", 
                    "mean", 
                    title="MSE per Lead"
                )
            })
            
            table = wandb.Table(data=[[patient_id, np.mean(patient_loss)] for patient_id, patient_loss in val_loss_per_patient.items()],
                                columns=["patient_id", "mse"])

            run.log({
                "val/mse_per_patient": wandb.plot.bar(
                    table, 
                    "patient_id", 
                    "mse", 
                    title="MSE per patient"
                )
            })

            table = wandb.Table(data=[[_cls, np.mean(cls_loss)] for _cls, cls_loss in val_loss_per_class.items()],
                                columns=["class", "mse"])

            run.log({
                "val/mse_per_class": wandb.plot.bar(
                    table, 
                    "class", 
                    "mse", 
                    title="MSE per class"
                )
            })

            logger.info('Epoch:{:4d}/{:4d} | tr_loss:{:8.2f}({:8.2f}) | valid_mse T={} :{:5.3f} | valid_mse T={} :{:5.3f} '.\
                    format(ep, args.Nepoch, loss_meter.val, loss_meter.avg, T_rec, valid_mse_rec, T_for, valid_mse_for)) 
            run.log({
                'val/tr_loss': loss_meter.val,
                'val/T_rec': valid_mse_rec
            })
                
            # update valid loggers
            vl_mse_rec.update(valid_mse_rec,ep)
            vl_mse_for.update(valid_mse_for, ep) 
            time_meter.update(time_val.seconds, ep)
    
            #compare validation error seen so far
            if best_valid_loss > valid_mse_rec: #we want as smaller mse
                best_valid_loss = valid_mse_rec

                torch.save({
                    'args': args,
                    'state_dict': model.state_dict(),
                }, os.path.join(args.save, 'model.pth'))
                            
                #compute test error for this model 
                dict_test_mses = {}
                dict_test_misc = {
                    'mse_t': [],
                    'mse_l': [],
                    #'mse_dt': []
                }
                test_loss_per_class = defaultdict(list)
                test_loss_per_patient = defaultdict(list)

                # test_mse = {}
                for test_batch, test_y, test_mask in testset:
                    test_batch = test_batch.to(model.device)
                    dict_mse, dict_misc, _loss_per_class, _loss_per_patient = compute_mse(model, test_batch, test_y, T_train, L=1, task=args.task, mask=test_mask)
                    for key,val in dict_mse.items():
                        if key not in dict_test_mses:
                            dict_test_mses[key] = []
                        dict_test_mses[key].append(val.item())
                    for cls, cls_mse in _loss_per_class.items():
                        test_loss_per_class[cls].append(cls_mse)

                    for patient_id, patient_mse in _loss_per_patient.items():
                        test_loss_per_patient[patient_id].append(patient_mse)

                    dict_test_misc['mse_t'].append(dict_misc['mse_t'])
                    dict_test_misc['mse_l'].append(dict_misc['mse_l'])
                    #dict_test_misc['mse_dt'].append(dict_misc['mse_dt'])

                logger.info('********** Current Best Model based on validation error ***********')
                logger.info('Epoch:{:4d}/{:4d}'.format(ep, args.Nepoch))
                for key, val in dict_test_mses.items():
                    logger.info('T={} test_mse {:5.3f}({:5.3f})'.format(key, np.mean(dict_test_mses[key]), np.std(dict_test_mses[key])))
                    run.log({
                        'test/mse': np.mean(dict_test_mses[key])
                    })

                mse_t, sse_t = compute_mse_stats(dict_test_misc['mse_t'])
                mse_l, sse_l = compute_mse_stats(dict_test_misc['mse_l'])
                #mse_dt, sse_dt = compute_mse_stats(dict_test_misc['mse_dt'])

                t = np.arange(len(mse_t))
                table = wandb.Table(data=[[ti, m, m-s, m+s] for ti, m, s in zip(t, mse_t, sse_t)],
                                    columns=["timestep", "mean", "lower", "upper"])

                run.log({
                "test/mse_t": wandb.plot_table(
                        vega_spec_name="tymechatu-university-of-amsterdam/std_band_custom",
                        data_table=table,
                        fields={
                            "x": "timestep",
                            "y": "mean",
                            "lower": "lower",
                            "upper": "upper"
                        }
                    )
                })

                # t = np.arange(len(mse_dt))
                # table = wandb.Table(data=[[ti, m, m-s, m+s] for ti, m, s in zip(t, mse_dt, sse_dt)],
                #                     columns=["timestep", "mean", "lower", "upper"])

                # run.log({
                #     "test/mse_dt": wandb.plot_table(
                #         vega_spec_name="tymechatu-university-of-amsterdam/std_band_custom", 
                #         data_table=table,
                #         fields={
                #             "x": "timestep",
                #             "y": "mean",
                #             "lower_bound": "lower",
                #             "upper_bound": "upper"
                #         }
                #     )
                # })

                channel = np.arange(len(mse_t)) if not custom_channel else custom_channel
                table = wandb.Table(data=[[ti, m, m-s, m+s] for ti, m, s in zip(channel, mse_l, sse_l)],
                                    columns=["lead", "mean", "lower", "upper"])

                run.log({
                    "test/mse_by_lead": wandb.plot.bar(
                        table, 
                        "lead", 
                        "mean", 
                        title="MSE per Lead"
                    )
                })
                table = wandb.Table(data=[[_cls, np.mean(cls_loss)] for _cls, cls_loss in test_loss_per_class.items()],
                                columns=["class", "mse"])

                run.log({
                    "test/mse_per_class": wandb.plot.bar(
                        table, 
                        "class", 
                        "mse", 
                        title="MSE per class"
                    )
                })

                table = wandb.Table(data=[[patient_id, np.mean(patient_loss)] for patient_id, patient_loss in test_loss_per_patient.items()],
                                columns=["patient_id", "mse"])

                run.log({
                    "test/mse_per_patient": wandb.plot.bar(
                        table, 
                        "patient_id", 
                        "mse", 
                        title="MSE per patient"
                    )
                })

            if ep % args.plot_every==0 or (ep+1) == args.Nepoch:
                plot_tr_batch = trainset.dataset.get_class_samples(k=1)
                plot_valid_batch = validset.dataset.get_class_samples(k=1)

                train_plot_dict = {}
                valid_plot_dict = {}

                for _cls, tr_batch in plot_tr_batch.items():
                    Xrec_tr, ztL_tr, _, _, C_tr, _, _ = model(tr_batch.to(model.device), L=args.plotL)
                    train_plot_dict[_cls] = {
                        'Xrec': Xrec_tr.cpu().detach(),
                        'ztL': ztL_tr.cpu().detach(),
                        'C': C_tr.cpu().detach() if C_tr is not None else None,
                        'batch': tr_batch.cpu().detach()
                    }
                
                for _cls, valid_batch in plot_valid_batch.items():
                    Xrec_vl, ztL_vl, _, _, C_vl, _, _ = model(valid_batch.to(model.device),  L=args.plotL)
                    valid_plot_dict[_cls] = {
                        'Xrec': Xrec_vl.cpu().detach(),
                        'ztL': ztL_vl.cpu().detach(),
                        'C': C_vl.cpu().detach() if C_vl is not None else None,
                        'batch': valid_batch.cpu().detach()
                    }

                plot_config = {}
                if args.task == 'ecg':
                    plot_config = {
                        'exclude_leads': params['exclude_leads'],
                        'f': params['f'],
                        'run': run,
                    }
                for _cls in train_plot_dict.keys():
                    if args.model == 'node' or args.model == 'hbnode':
                        
                        plot_results(plotter, \
                                    train_plot_dict[_cls]['Xrec'], train_plot_dict[_cls]['batch'], valid_plot_dict[_cls]['Xrec'], valid_plot_dict[_cls]['batch'], \
                                    {"plot":{'Loss(-elbo)': loss_meter, 'Nll' : nll_meter, 'KL-z0': kl_z0_meter, "train-MSE": tr_mse_meter}, "valid-MSE-rec": vl_mse_rec, "valid-MSE-for": vl_mse_for, "iteration": ep, "time": time_meter}, \
                                    train_plot_dict[_cls]['ztL'],  valid_plot_dict[_cls]['ztL'],  train_plot_dict[_cls]['C'], valid_plot_dict[_cls]['C'], tr_fname=f'tr_{_cls}', val_fname=f'val_{_cls}', **plot_config)
                    elif args.model == 'sonode':
                        plot_results(plotter, \
                                    train_plot_dict[_cls]['Xrec'], train_plot_dict[_cls]['batch'], valid_plot_dict[_cls]['Xrec'], valid_plot_dict[_cls]['batch'],\
                                    {"plot":{"Loss" : loss_meter, "valid-MSE-rec": vl_mse_rec, "valid-MSE-for":vl_mse_for}, "time" : time_meter, "iteration": ep}, **plot_config)


    logger.info('Epoch:{:4d}/{:4d} | time: {} | train_elbo: {:8.2f} | train_mse: {:5.3f} | valid_mse_rec: {:5.3f}) | valid_mse_for: {:5.3f})  | best_valid_mse: {:5.3f})'.\
                format(ep, args.Nepoch, datetime.now()-start_time, loss_meter.val, tr_mse_meter.avg, vl_mse_rec.val, vl_mse_for.val, best_valid_loss))

    for key, val in dict_test_mses.items():
        logger.info('T={} test_mse {:5.3f}({:5.3f})'.format(key, np.mean(dict_test_mses[key]), np.std(dict_test_mses[key])))

    
def log_gradients(model, run):

    total_norm_gru = 0
    for p in model.vae.encoder.parameters():
        if p.grad is not None:
            total_norm_gru += p.grad.data.norm(2).item()

    total_norm_ode = 0
    for p in model.flow.odefunc.parameters():
        if p.grad is not None:
            total_norm_ode += p.grad.data.norm(2).item()

    total_norm_dec = 0
    for p in model.vae.decoder.parameters():
        if p.grad is not None:
            total_norm_dec += p.grad.data.norm(2).item()

    run.log({
        "grads/gru_norm": total_norm_gru,
        "grads/ode_norm": total_norm_ode,
        "grads/dec_norm": total_norm_dec,
        "grads/ratio_gru_to_dec": total_norm_gru / (total_norm_dec + 1e-8) 
    })


