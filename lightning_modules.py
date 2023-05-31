import math
from argparse import Namespace
from typing import Optional
from time import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import wandb
from torch_scatter import scatter_add, scatter_mean
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_one

from constants import dataset_params, FLOAT_TYPE, INT_TYPE
from equivariant_diffusion.dynamics import EGNNDynamics
from equivariant_diffusion.en_diffusion import EnVariationalDiffusion
from equivariant_diffusion.conditional_model import ConditionalDDPM, \
    SimpleConditionalDDPM
from dataset import ProcessedLigandPocketDataset
import utils
from analysis.visualization import save_xyz_file, visualize, visualize_chain
from analysis.metrics import check_stability, BasicMolecularMetrics, \
    CategoricalDistribution
from analysis.molecule_builder import build_molecule, process_molecule


class LigandPocketDDPM(pl.LightningModule):
    def __init__(
            self,
            outdir,
            dataset,
            datadir,
            batch_size,
            lr,
            egnn_params: Namespace,
            diffusion_params,
            num_workers,
            augment_noise,
            augment_rotation,
            clip_grad,
            eval_epochs,
            eval_params,
            visualize_sample_epoch,
            visualize_chain_epoch,
            auxiliary_loss,
            loss_params,
            mode,
            node_histogram,
            pocket_representation='CA',
    ):
        super(LigandPocketDDPM, self).__init__()
        self.save_hyperparameters()

        ddpm_models = {'joint': EnVariationalDiffusion,
                       'pocket_conditioning': ConditionalDDPM,
                       'pocket_conditioning_simple': SimpleConditionalDDPM}
        assert mode in ddpm_models
        self.mode = mode
        assert pocket_representation in {'CA', 'full-atom'}
        self.pocket_representation = pocket_representation

        self.dataset_name = dataset
        self.datadir = datadir
        self.outdir = outdir
        self.batch_size = batch_size
        self.eval_batch_size = eval_params.eval_batch_size \
            if 'eval_batch_size' in eval_params else batch_size
        self.lr = lr
        self.loss_type = diffusion_params.diffusion_loss_type
        self.eval_epochs = eval_epochs
        self.visualize_sample_epoch = visualize_sample_epoch
        self.visualize_chain_epoch = visualize_chain_epoch
        self.eval_params = eval_params
        self.num_workers = num_workers
        self.augment_noise = augment_noise
        self.augment_rotation = augment_rotation
        self.dataset_info = dataset_params[dataset]
        self.T = diffusion_params.diffusion_steps
        self.clip_grad = clip_grad
        if clip_grad:
            self.gradnorm_queue = utils.Queue()
            # Add large value that will be flushed.
            self.gradnorm_queue.add(3000)

        smiles_list = None if eval_params.smiles_file is None \
            else np.load(eval_params.smiles_file)
        self.ligand_metrics = BasicMolecularMetrics(self.dataset_info,
                                                    smiles_list)
        self.ligand_type_distribution = CategoricalDistribution(
            self.dataset_info['atom_hist'], self.dataset_info['atom_encoder'])
        if self.pocket_representation == 'CA':
            self.pocket_type_distribution = CategoricalDistribution(
                self.dataset_info['aa_hist'], self.dataset_info['aa_encoder'])
        else:
            # TODO: full-atom case
            self.pocket_type_distribution = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.lig_type_encoder = self.dataset_info['atom_encoder']
        self.lig_type_decoder = self.dataset_info['atom_decoder']
        self.pocket_type_encoder = self.dataset_info['aa_encoder'] \
            if self.pocket_representation == 'CA' \
            else self.dataset_info['atom_encoder']
        self.pocket_type_decoder = self.dataset_info['aa_decoder'] \
            if self.pocket_representation == 'CA' \
            else self.dataset_info['atom_decoder']

        self.atom_nf = len(self.lig_type_decoder)
        self.aa_nf = len(self.pocket_type_decoder)
        self.x_dims = 3

        net_dynamics = EGNNDynamics(
            atom_nf=self.atom_nf,
            residue_nf=self.aa_nf,
            n_dims=self.x_dims,
            joint_nf=egnn_params.joint_nf,
            device=egnn_params.device if torch.cuda.is_available() else 'cpu',
            hidden_nf=egnn_params.hidden_nf,
            act_fn=torch.nn.SiLU(),
            n_layers=egnn_params.n_layers,
            attention=egnn_params.attention,
            tanh=egnn_params.tanh,
            norm_constant=egnn_params.norm_constant,
            inv_sublayers=egnn_params.inv_sublayers,
            sin_embedding=egnn_params.sin_embedding,
            normalization_factor=egnn_params.normalization_factor,
            aggregation_method=egnn_params.aggregation_method,
            edge_cutoff=egnn_params.__dict__.get('edge_cutoff'),
            update_pocket_coords=(self.mode == 'joint')
        )

        self.ddpm = ddpm_models[self.mode](
                dynamics=net_dynamics,
                atom_nf=self.atom_nf,
                residue_nf=self.aa_nf,
                n_dims=self.x_dims,
                timesteps=diffusion_params.diffusion_steps,
                noise_schedule=diffusion_params.diffusion_noise_schedule,
                noise_precision=diffusion_params.diffusion_noise_precision,
                loss_type=diffusion_params.diffusion_loss_type,
                norm_values=diffusion_params.normalize_factors,
                size_histogram=node_histogram,
            )

        self.auxiliary_loss = auxiliary_loss
        if self.auxiliary_loss:
            self.clamp_lj = loss_params.clamp_lj
            self.auxiliary_weight_schedule = WeightSchedule(
                T=diffusion_params.diffusion_steps,
                max_weight=loss_params.max_weight, mode=loss_params.schedule)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.ddpm.parameters(), lr=self.lr,
                                 amsgrad=True, weight_decay=1e-12)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            self.train_dataset = ProcessedLigandPocketDataset(
                Path(self.datadir, 'train.npz'))
            self.val_dataset = ProcessedLigandPocketDataset(
                Path(self.datadir, 'val.npz'))
        elif stage == 'test':
            self.test_dataset = ProcessedLigandPocketDataset(
                Path(self.datadir, 'test.npz'))
        else:
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=self.train_dataset.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.val_dataset.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.test_dataset.collate_fn)

    def get_ligand_and_pocket(self, data):
        ligand = {
            'x': data['lig_coords'].to(self.device, FLOAT_TYPE),
            'one_hot': data['lig_one_hot'].to(self.device, FLOAT_TYPE),
            'size': data['num_lig_atoms'].to(self.device, INT_TYPE),
            'mask': data['lig_mask'].to(self.device, INT_TYPE)
        }

        pocket = {
            'x': data['pocket_c_alpha'].to(self.device, FLOAT_TYPE),
            'one_hot': data['pocket_one_hot'].to(self.device, FLOAT_TYPE),
            'size': data['num_pocket_nodes'].to(self.device, INT_TYPE),
            'mask': data['pocket_mask'].to(self.device, INT_TYPE)
        }
        return ligand, pocket

    def forward(self, data):
        ligand, pocket = self.get_ligand_and_pocket(data)

        # Note: \mathcal{L} terms in the paper represent log-likelihoods while
        # our loss terms are a negative(!) log-likelihoods
        delta_log_px, error_t_lig, error_t_pocket, SNR_weight, \
        loss_0_x_ligand, loss_0_x_pocket, loss_0_h, neg_log_const_0, \
        kl_prior, log_pN, t_int, xh_lig_hat, info = \
            self.ddpm(ligand, pocket, return_info=True)

        if self.loss_type == 'l2' and self.training:
            # normalize loss_t
            denom_lig = (self.x_dims + self.ddpm.atom_nf) * ligand['size']
            error_t_lig = error_t_lig / denom_lig
            denom_pocket = (self.x_dims + self.ddpm.residue_nf) * pocket['size']
            error_t_pocket = error_t_pocket / denom_pocket
            loss_t = 0.5 * (error_t_lig + error_t_pocket)

            # normalize loss_0
            loss_0_x_ligand = loss_0_x_ligand / (self.x_dims * ligand['size'])
            loss_0_x_pocket = loss_0_x_pocket / (self.x_dims * pocket['size'])
            loss_0 = loss_0_x_ligand + loss_0_x_pocket + loss_0_h

        # VLB objective or evaluation step
        else:
            # Note: SNR_weight should be negative
            loss_t = -self.T * 0.5 * SNR_weight * (error_t_lig + error_t_pocket)
            loss_0 = loss_0_x_ligand + loss_0_x_pocket + loss_0_h
            loss_0 = loss_0 + neg_log_const_0

        nll = loss_t + loss_0 + kl_prior

        # Correct for normalization on x.
        if not (self.loss_type == 'l2' and self.training):
            nll = nll - delta_log_px

            # Transform conditional nll into joint nll
            # Note:
            # loss = -log p(x,h|N) and log p(x,h,N) = log p(x,h|N) + log p(N)
            # Therefore, log p(x,h|N) = -loss + log p(N)
            # => loss_new = -log p(x,h,N) = loss - log p(N)
            nll = nll - log_pN

        # Add auxiliary loss term
        if self.auxiliary_loss and self.loss_type == 'l2' and self.training:
            x_lig_hat = xh_lig_hat[:, :self.x_dims]
            h_lig_hat = xh_lig_hat[:, self.x_dims:]
            weighted_lj_potential = \
                self.auxiliary_weight_schedule(t_int.long()) * \
                self.lj_potential(x_lig_hat, h_lig_hat, ligand['mask'])
            nll = nll + weighted_lj_potential
            info['weighted_lj'] = weighted_lj_potential.mean(0)

        info['error_t_lig'] = error_t_lig.mean(0)
        info['error_t_pocket'] = error_t_pocket.mean(0)
        info['SNR_weight'] = SNR_weight.mean(0)
        info['loss_0'] = loss_0.mean(0)
        info['kl_prior'] = kl_prior.mean(0)
        info['delta_log_px'] = delta_log_px.mean(0)
        info['neg_log_const_0'] = neg_log_const_0.mean(0)
        info['log_pN'] = log_pN.mean(0)
        return nll, info

    def lj_potential(self, atom_x, atom_one_hot, batch_mask):
        adj = batch_mask[:, None] == batch_mask[None, :]
        adj = adj ^ torch.diag(torch.diag(adj))  # remove self-edges
        edges = torch.where(adj)

        # Compute pair-wise potentials
        r = torch.sum((atom_x[edges[0]] - atom_x[edges[1]])**2, dim=1).sqrt()

        # Get optimal radii
        lennard_jones_radii = torch.tensor(
            self.dataset_info['lennard_jones_rm'], device=r.device)
        # unit conversion pm -> A
        lennard_jones_radii = lennard_jones_radii / 100.0
        # normalization
        lennard_jones_radii = lennard_jones_radii / self.ddpm.norm_values[0]
        atom_type_idx = atom_one_hot.argmax(1)
        rm = lennard_jones_radii[atom_type_idx[edges[0]],
                                 atom_type_idx[edges[1]]]
        sigma = 2 ** (-1 / 6) * rm
        out = 4 * ((sigma / r) ** 12 - (sigma / r) ** 6)

        if self.clamp_lj is not None:
            out = torch.clamp(out, min=None, max=self.clamp_lj)

        # Compute potential per atom
        out = scatter_add(out, edges[0], dim=0, dim_size=len(atom_x))

        # Sum potentials of all atoms
        return scatter_add(out, batch_mask, dim=0)

    def log_metrics(self, metrics_dict, split, batch_size=None, **kwargs):
        for m, value in metrics_dict.items():
            self.log(f'{m}/{split}', value, batch_size=batch_size, **kwargs)

    def training_step(self, data, *args):
        if self.augment_noise > 0:
            raise NotImplementedError
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian(x.size(), x.device)
            x = x + eps * args.augment_noise

        if self.augment_rotation:
            raise NotImplementedError
            x = utils.random_rotation(x).detach()

        nll, info = self.forward(data)
        loss = nll.mean(0)

        info['loss'] = loss
        self.log_metrics(info, 'train', batch_size=len(data['num_lig_atoms']))

        return info

    def _shared_eval(self, data, prefix, *args):
        nll, info = self.forward(data)
        loss = nll.mean(0)

        info['loss'] = loss

        # some additional info
        gamma_0 = self.ddpm.gamma(torch.zeros(1, device=self.device))
        gamma_1 = self.ddpm.gamma(torch.ones(1, device=self.device))
        log_SNR_max = -gamma_0
        log_SNR_min = -gamma_1
        info['log_SNR_max'] = log_SNR_max
        info['log_SNR_min'] = log_SNR_min

        self.log_metrics(info, prefix, batch_size=len(data['num_lig_atoms']),
                         sync_dist=True)

        return info

    def validation_step(self, data, *args):
        self._shared_eval(data, 'val', *args)

    def test_step(self, data, *args):
        self._shared_eval(data, 'test', *args)

    def validation_epoch_end(self, validation_step_outputs):

        # Perform validation on single GPU
        # TODO: sample on multiple devices if available
        if not self.trainer.is_global_zero:
            return

        suffix = '' if self.mode == 'joint' else '_given_pocket'

        if (self.current_epoch + 1) % self.eval_epochs == 0:
            tic = time()

            sampling_results = getattr(self, 'sample_and_analyze' + suffix)(
                self.eval_params.n_eval_samples, self.val_dataset,
                batch_size=self.eval_batch_size)
            self.log_metrics(sampling_results, 'val')

            print(f'Evaluation took {time() - tic:.2f} seconds')

        if (self.current_epoch + 1) % self.visualize_sample_epoch == 0:
            tic = time()
            getattr(self, 'sample_and_save' + suffix)(
                self.eval_params.n_visualize_samples)
            print(f'Sample visualization took {time() - tic:.2f} seconds')

        if (self.current_epoch + 1) % self.visualize_chain_epoch == 0:
            tic = time()
            getattr(self, 'sample_chain_and_save' + suffix)(
                self.eval_params.keep_frames)
            print(f'Chain visualization took {time() - tic:.2f} seconds')

    @torch.no_grad()
    def sample_and_analyze(self, n_samples, dataset=None, batch_size=None):
        print(f'Analyzing molecule stability at epoch {self.current_epoch}...')

        batch_size = self.batch_size if batch_size is None else batch_size
        batch_size = min(batch_size, n_samples)

        # each item in molecules is a tuple (position, atom_type_encoded)
        molecules = []
        atom_types = []
        aa_types = []
        for i in range(math.ceil(n_samples / batch_size)):

            n_samples_batch = min(batch_size, n_samples - len(molecules))

            num_nodes_lig, num_nodes_pocket = \
                self.ddpm.size_distribution.sample(n_samples_batch)

            xh_lig, xh_pocket, lig_mask, _ = self.ddpm.sample(
                n_samples_batch, num_nodes_lig, num_nodes_pocket,
                device=self.device)

            x = xh_lig[:, :self.x_dims].detach().cpu()
            atom_type = xh_lig[:, self.x_dims:].argmax(1).detach().cpu()

            molecules.extend(list(
                zip(utils.batch_to_list(x, lig_mask),
                    utils.batch_to_list(atom_type, lig_mask))
            ))

            atom_types.extend(atom_type.tolist())
            aa_types.extend(
                xh_pocket[:, self.x_dims:].argmax(1).detach().cpu().tolist())

        return self.analyze_sample(molecules, atom_types, aa_types)

    def analyze_sample(self, molecules, atom_types, aa_types):
        # Distribution of node types
        kl_div_atom = self.ligand_type_distribution.kl_divergence(atom_types) \
            if self.ligand_type_distribution is not None else -1
        kl_div_aa = self.pocket_type_distribution.kl_divergence(aa_types) \
            if self.pocket_type_distribution is not None else -1

        # Stability
        molecule_stable = 0
        nr_stable_bonds = 0
        n_atoms = 0
        for pos, atom_type in molecules:
            validity_results = check_stability(pos, atom_type,
                                               self.dataset_info)
            molecule_stable += int(validity_results[0])
            nr_stable_bonds += int(validity_results[1])
            n_atoms += int(validity_results[2])

        fraction_mol_stable = molecule_stable / float(len(molecules))
        fraction_atm_stable = nr_stable_bonds / float(n_atoms)

        # Other basic metrics
        validity, connectivity, uniqueness, novelty = \
            self.ligand_metrics.evaluate(molecules)[0]

        return {
            'kl_div_atom_types': kl_div_atom,
            'kl_div_residue_types': kl_div_aa,
            'mol_stable': fraction_mol_stable,
            'atm_stable': fraction_atm_stable,
            'Validity': validity,
            'Connectivity': connectivity,
            'Uniqueness': uniqueness,
            'Novelty': novelty
        }

    @torch.no_grad()
    def sample_and_analyze_given_pocket(self, n_samples, dataset=None,
                                        batch_size=None):
        print(f'Analyzing molecule stability given pockets at epoch '
              f'{self.current_epoch}...')

        batch_size = self.batch_size if batch_size is None else batch_size
        batch_size = min(batch_size, n_samples)

        # each item in molecules is a tuple (position, atom_type_encoded)
        molecules = []
        atom_types = []
        aa_types = []
        for i in range(math.ceil(n_samples / batch_size)):

            n_samples_batch = min(batch_size, n_samples - len(molecules))

            # Create a batch
            batch = dataset.collate_fn(
                [dataset[(i * batch_size + j) % len(dataset)]
                 for j in range(n_samples_batch)]
            )

            ligand, pocket = self.get_ligand_and_pocket(batch)

            num_nodes_lig = self.ddpm.size_distribution.sample_conditional(
                n1=None, n2=pocket['size'])

            xh_lig, xh_pocket, lig_mask, _ = self.ddpm.sample_given_pocket(
                pocket, num_nodes_lig)

            x = xh_lig[:, :self.x_dims].detach().cpu()
            atom_type = xh_lig[:, self.x_dims:].argmax(1).detach().cpu()

            molecules.extend(list(
                zip(utils.batch_to_list(x, lig_mask),
                    utils.batch_to_list(atom_type, lig_mask))
            ))

            atom_types.extend(atom_type.tolist())
            aa_types.extend(
                xh_pocket[:, self.x_dims:].argmax(1).detach().cpu().tolist())

        return self.analyze_sample(molecules, atom_types, aa_types)

    def sample_and_save(self, n_samples):
        num_nodes_lig, num_nodes_pocket = \
            self.ddpm.size_distribution.sample(n_samples)

        xh_lig, xh_pocket, lig_mask, pocket_mask = \
            self.ddpm.sample(n_samples, num_nodes_lig, num_nodes_pocket,
                             device=self.device)

        if self.pocket_representation == 'CA':
            # convert residues into atom representation for visualization
            x_pocket, one_hot_pocket = utils.residues_to_atoms(
                xh_pocket[:, :self.x_dims], self.dataset_info)
        else:
            x_pocket, one_hot_pocket = \
                xh_pocket[:, :self.x_dims], xh_pocket[:, self.x_dims:]
        x = torch.cat((xh_lig[:, :self.x_dims], x_pocket), dim=0)
        one_hot = torch.cat((xh_lig[:, self.x_dims:], one_hot_pocket), dim=0)

        outdir = Path(self.outdir, f'epoch_{self.current_epoch}')
        save_xyz_file(str(outdir) + '/', one_hot, x, self.dataset_info,
                      name='molecule',
                      batch_mask=torch.cat((lig_mask, pocket_mask)))
        # visualize(str(outdir), dataset_info=self.dataset_info, wandb=wandb)
        visualize(str(outdir), dataset_info=self.dataset_info, wandb=None)

    def sample_and_save_given_pocket(self, n_samples):
        batch = self.val_dataset.collate_fn(
            [self.val_dataset[i] for i in torch.randint(len(self.val_dataset),
                                                        size=(n_samples,))]
        )
        ligand, pocket = self.get_ligand_and_pocket(batch)

        num_nodes_lig = self.ddpm.size_distribution.sample_conditional(
            n1=None, n2=pocket['size'])

        xh_lig, xh_pocket, lig_mask, pocket_mask = \
            self.ddpm.sample_given_pocket(pocket, num_nodes_lig)

        if self.pocket_representation == 'CA':
            # convert residues into atom representation for visualization
            x_pocket, one_hot_pocket = utils.residues_to_atoms(
                xh_pocket[:, :self.x_dims], self.dataset_info)
        else:
            x_pocket, one_hot_pocket = \
                xh_pocket[:, :self.x_dims], xh_pocket[:, self.x_dims:]
        x = torch.cat((xh_lig[:, :self.x_dims], x_pocket), dim=0)
        one_hot = torch.cat((xh_lig[:, self.x_dims:], one_hot_pocket), dim=0)

        outdir = Path(self.outdir, f'epoch_{self.current_epoch}')
        save_xyz_file(str(outdir) + '/', one_hot, x, self.dataset_info,
                      name='molecule',
                      batch_mask=torch.cat((lig_mask, pocket_mask)))
        # visualize(str(outdir), dataset_info=self.dataset_info, wandb=wandb)
        visualize(str(outdir), dataset_info=self.dataset_info, wandb=None)

    def sample_chain_and_save(self, keep_frames):
        n_samples = 1
        n_tries = 1

        num_nodes_lig, num_nodes_pocket = \
            self.ddpm.size_distribution.sample(n_samples)

        one_hot_lig, x_lig, one_hot_pocket, x_pocket = [None] * 4
        for i in range(n_tries):
            chain_lig, chain_pocket, _, _ = self.ddpm.sample(
                n_samples, num_nodes_lig, num_nodes_pocket,
                return_frames=keep_frames, device=self.device)

            chain_lig = utils.reverse_tensor(chain_lig)
            chain_pocket = utils.reverse_tensor(chain_pocket)

            # Repeat last frame to see final sample better.
            chain_lig = torch.cat([chain_lig, chain_lig[-1:].repeat(10, 1, 1)],
                                  dim=0)
            chain_pocket = torch.cat(
                [chain_pocket, chain_pocket[-1:].repeat(10, 1, 1)], dim=0)

            # Check stability of the generated ligand
            x_final = chain_lig[-1, :, :self.x_dims].cpu().detach().numpy()
            one_hot_final = chain_lig[-1, :, self.x_dims:]
            atom_type_final = torch.argmax(
                one_hot_final, dim=1).cpu().detach().numpy()

            mol_stable = check_stability(x_final, atom_type_final,
                                         self.dataset_info)[0]

            # Prepare entire chain.
            x_lig = chain_lig[:, :, :self.x_dims]
            one_hot_lig = chain_lig[:, :, self.x_dims:]
            one_hot_lig = F.one_hot(
                torch.argmax(one_hot_lig, dim=2),
                num_classes=len(self.lig_type_decoder))
            x_pocket = chain_pocket[:, :, :self.x_dims]
            one_hot_pocket = chain_pocket[:, :, self.x_dims:]
            one_hot_pocket = F.one_hot(
                torch.argmax(one_hot_pocket, dim=2),
                num_classes=len(self.pocket_type_decoder))

            if mol_stable:
                print('Found stable molecule to visualize :)')
                break
            elif i == n_tries - 1:
                print('Did not find stable molecule, showing last sample.')

        if self.pocket_representation == 'CA':
            # convert residues into atom representation for visualization
            x_pocket, one_hot_pocket = utils.residues_to_atoms(
                x_pocket, self.dataset_info)

        x = torch.cat((x_lig, x_pocket), dim=1)
        one_hot = torch.cat((one_hot_lig, one_hot_pocket), dim=1)

        # flatten (treat frame (chain dimension) as batch for visualization)
        x_flat = x.view(-1, x.size(-1))
        one_hot_flat = one_hot.view(-1, one_hot.size(-1))
        mask_flat = torch.arange(x.size(0)).repeat_interleave(x.size(1))

        outdir = Path(self.outdir, f'epoch_{self.current_epoch}', 'chain')
        save_xyz_file(str(outdir), one_hot_flat, x_flat, self.dataset_info,
                      name='/chain', batch_mask=mask_flat)
        visualize_chain(str(outdir), self.dataset_info, wandb=wandb)

    def sample_chain_and_save_given_pocket(self, keep_frames):
        n_samples = 1
        n_tries = 1

        batch = self.val_dataset.collate_fn([
            self.val_dataset[torch.randint(len(self.val_dataset), size=(1,))]
        ])
        ligand, pocket = self.get_ligand_and_pocket(batch)

        num_nodes_lig = self.ddpm.size_distribution.sample_conditional(
            n1=None, n2=pocket['size'])

        one_hot_lig, x_lig, one_hot_pocket, x_pocket = [None] * 4
        for i in range(n_tries):
            chain_lig, chain_pocket, _, _ = self.ddpm.sample_given_pocket(
                pocket, num_nodes_lig, return_frames=keep_frames)

            chain_lig = utils.reverse_tensor(chain_lig)
            chain_pocket = utils.reverse_tensor(chain_pocket)

            # Repeat last frame to see final sample better.
            chain_lig = torch.cat([chain_lig, chain_lig[-1:].repeat(10, 1, 1)],
                                  dim=0)
            chain_pocket = torch.cat(
                [chain_pocket, chain_pocket[-1:].repeat(10, 1, 1)], dim=0)

            # Check stability of the generated ligand
            x_final = chain_lig[-1, :, :self.x_dims].cpu().detach().numpy()
            one_hot_final = chain_lig[-1, :, self.x_dims:]
            atom_type_final = torch.argmax(
                one_hot_final, dim=1).cpu().detach().numpy()

            mol_stable = check_stability(x_final, atom_type_final,
                                         self.dataset_info)[0]

            # Prepare entire chain.
            x_lig = chain_lig[:, :, :self.x_dims]
            one_hot_lig = chain_lig[:, :, self.x_dims:]
            one_hot_lig = F.one_hot(
                torch.argmax(one_hot_lig, dim=2),
                num_classes=len(self.lig_type_decoder))
            x_pocket = chain_pocket[:, :, :3]
            one_hot_pocket = chain_pocket[:, :, 3:]
            one_hot_pocket = F.one_hot(
                torch.argmax(one_hot_pocket, dim=2),
                num_classes=len(self.pocket_type_decoder))

            if mol_stable:
                print('Found stable molecule to visualize :)')
                break
            elif i == n_tries - 1:
                print('Did not find stable molecule, showing last sample.')

        if self.pocket_representation == 'CA':
            # convert residues into atom representation for visualization
            x_pocket, one_hot_pocket = utils.residues_to_atoms(
                x_pocket, self.dataset_info)

        x = torch.cat((x_lig, x_pocket), dim=1)
        one_hot = torch.cat((one_hot_lig, one_hot_pocket), dim=1)

        # flatten (treat frame (chain dimension) as batch for visualization)
        x_flat = x.view(-1, x.size(-1))
        one_hot_flat = one_hot.view(-1, one_hot.size(-1))
        mask_flat = torch.arange(x.size(0)).repeat_interleave(x.size(1))

        outdir = Path(self.outdir, f'epoch_{self.current_epoch}', 'chain')
        save_xyz_file(str(outdir), one_hot_flat, x_flat, self.dataset_info,
                      name='/chain', batch_mask=mask_flat)
        visualize_chain(str(outdir), self.dataset_info, wandb=wandb)

    def generate_ligands(self, pdb_file, n_samples, pocket_ids=None,
                         ref_ligand=None, num_nodes_lig=None, sanitize=False,
                         largest_frag=False, relax_iter=0, timesteps=None,
                         **kwargs):
        """
        Generate ligands given a pocket
        Args:
            pdb_file: PDB filename
            n_samples: number of samples
            pocket_ids: list of pocket residues in <chain>:<resi> format
            ref_ligand: alternative way of defining the pocket based on a
                reference ligand given in <chain>:<resi> format
            num_nodes_lig: number of ligand nodes for each sample (list of
                integers), sampled randomly if 'None'
            sanitize: whether to sanitize molecules or not
            largest_frag: only return the largest fragment
            relax_iter: number of force field optimization steps
            timesteps: number of denoising steps, use training value if None
            kwargs: additional inpainting parameters
        Returns:
            list of molecules
        """

        assert (pocket_ids is None) ^ (ref_ligand is None)

        # Load PDB
        pdb_struct = PDBParser(QUIET=True).get_structure('', pdb_file)[0]
        if pocket_ids is not None:
            # define pocket with list of residues
            residues = [
                pdb_struct[x.split(':')[0]][(' ', int(x.split(':')[1]), ' ')]
                for x in pocket_ids]

        else:
            # define pocket with reference ligand
            residues = utils.get_pocket_from_ligand(pdb_struct, ref_ligand)

        if self.pocket_representation == 'CA':
            pocket_coord = torch.tensor(np.array(
                [res['CA'].get_coord() for res in residues]),
                device=self.device, dtype=FLOAT_TYPE)
            pocket_types = torch.tensor(
                [self.pocket_type_encoder[three_to_one(res.get_resname())]
                 for res in residues], device=self.device)
        else:
            pocket_atoms = [a for res in residues for a in res.get_atoms()
                            if (a.element.capitalize() in self.pocket_type_encoder or a.element != 'H')]
            pocket_coord = torch.tensor(np.array(
                [a.get_coord() for a in pocket_atoms]),
                device=self.device, dtype=FLOAT_TYPE)
            pocket_types = torch.tensor(
                [self.pocket_type_encoder[a.element.capitalize()]
                 for a in pocket_atoms], device=self.device)

        pocket_one_hot = F.one_hot(
            pocket_types, num_classes=len(self.pocket_type_encoder)
        )

        pocket_size = torch.tensor([len(pocket_coord)] * n_samples,
                                   device=self.device, dtype=INT_TYPE)
        pocket_mask = torch.repeat_interleave(
            torch.arange(n_samples, device=self.device, dtype=INT_TYPE),
            len(pocket_coord)
        )

        pocket = {
            'x': pocket_coord.repeat(n_samples, 1),
            'one_hot': pocket_one_hot.repeat(n_samples, 1),
            'size': pocket_size,
            'mask': pocket_mask
        }

        # Pocket's center of mass
        pocket_com_before = scatter_mean(pocket['x'], pocket['mask'], dim=0)

        # Create dummy ligands
        if num_nodes_lig is None:
            num_nodes_lig = self.ddpm.size_distribution.sample_conditional(
                n1=None, n2=pocket['size'])

        # Use inpainting
        if type(self.ddpm) == EnVariationalDiffusion:
            lig_mask = utils.num_nodes_to_batch_mask(
                len(num_nodes_lig), num_nodes_lig, self.device)

            ligand = {
                'x': torch.zeros((len(lig_mask), self.x_dims),
                                 device=self.device, dtype=FLOAT_TYPE),
                'one_hot': torch.zeros((len(lig_mask), self.atom_nf),
                                       device=self.device, dtype=FLOAT_TYPE),
                'size': num_nodes_lig,
                'mask': lig_mask
            }

            # Fix all pocket nodes but sample
            lig_mask_fixed = torch.zeros(len(lig_mask), device=self.device)
            pocket_mask_fixed = torch.ones(len(pocket['mask']),
                                           device=self.device)

            xh_lig, xh_pocket, lig_mask, pocket_mask = self.ddpm.inpaint(
                ligand, pocket, lig_mask_fixed, pocket_mask_fixed,
                timesteps=timesteps, **kwargs)

        # Use conditional generation
        elif type(self.ddpm) == ConditionalDDPM:
            xh_lig, xh_pocket, lig_mask, pocket_mask = \
                self.ddpm.sample_given_pocket(pocket, num_nodes_lig,
                                              timesteps=timesteps)

        else:
            raise NotImplementedError

        # Move generated molecule back to the original pocket position
        pocket_com_after = scatter_mean(
            xh_pocket[:, :self.x_dims], pocket_mask, dim=0)

        xh_pocket[:, :self.x_dims] += \
            (pocket_com_before - pocket_com_after)[pocket_mask]
        xh_lig[:, :self.x_dims] += \
            (pocket_com_before - pocket_com_after)[lig_mask]

        # Build mol objects
        lig_mask = lig_mask.cpu()
        x = xh_lig[:, :self.x_dims].detach().cpu()
        atom_type = xh_lig[:, self.x_dims:].argmax(1).detach().cpu()

        molecules = []
        for mol_pc in zip(utils.batch_to_list(x, lig_mask),
                          utils.batch_to_list(atom_type, lig_mask)):

            mol = build_molecule(*mol_pc, self.dataset_info, add_coords=True)
            mol = process_molecule(mol,
                                   add_hydrogens=False,
                                   sanitize=sanitize,
                                   relax_iter=relax_iter,
                                   largest_frag=largest_frag)
            if mol is not None:
                molecules.append(mol)

        return molecules

    def configure_gradient_clipping(self, optimizer, optimizer_idx,
                                    gradient_clip_val, gradient_clip_algorithm):

        if not self.clip_grad:
            return

        # Allow gradient norm to be 150% + 2 * stdev of the recent history.
        max_grad_norm = 1.5 * self.gradnorm_queue.mean() + \
                        2 * self.gradnorm_queue.std()

        # Get current grad_norm
        params = [p for g in optimizer.param_groups for p in g['params']]
        grad_norm = utils.get_grad_norm(params)

        # Lightning will handle the gradient clipping
        self.clip_gradients(optimizer, gradient_clip_val=max_grad_norm,
                            gradient_clip_algorithm='norm')

        if float(grad_norm) > max_grad_norm:
            self.gradnorm_queue.add(float(max_grad_norm))
        else:
            self.gradnorm_queue.add(float(grad_norm))

        if float(grad_norm) > max_grad_norm:
            print(f'Clipped gradient with value {grad_norm:.1f} '
                  f'while allowed {max_grad_norm:.1f}')


class WeightSchedule:
    def __init__(self, T, max_weight, mode='linear'):
        if mode == 'linear':
            self.weights = torch.linspace(max_weight, 0, T + 1)
        elif mode == 'constant':
            self.weights = max_weight * torch.ones(T + 1)
        else:
            raise NotImplementedError(f'{mode} weight schedule is not '
                                      f'available.')

    def __call__(self, t_array):
        """ all values in t_array are assumed to be integers in [0, T] """
        return self.weights[t_array].to(t_array.device)
