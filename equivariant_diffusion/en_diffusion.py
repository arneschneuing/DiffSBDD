import math
from typing import Dict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean

import utils


class EnVariationalDiffusion(nn.Module):
    """
    The E(n) Diffusion Module.
    """

    def __init__(
            self,
            dynamics: nn.Module, atom_nf: int, residue_nf: int,
            n_dims: int, size_histogram: Dict,
            timesteps: int = 1000, parametrization='eps',
            noise_schedule='learned', noise_precision=1e-4,
            loss_type='vlb', norm_values=(1., 1.), norm_biases=(None, 0.),
            virtual_node_idx=None):
        super().__init__()

        assert loss_type in {'vlb', 'l2'}
        self.loss_type = loss_type
        if noise_schedule == 'learned':
            assert loss_type == 'vlb', 'A noise schedule can only be learned' \
                                       ' with a vlb objective.'

        # Only supported parametrization.
        assert parametrization == 'eps'

        if noise_schedule == 'learned':
            self.gamma = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(noise_schedule,
                                                 timesteps=timesteps,
                                                 precision=noise_precision)

        # The network that will predict the denoising.
        self.dynamics = dynamics

        self.atom_nf = atom_nf
        self.residue_nf = residue_nf
        self.n_dims = n_dims
        self.num_classes = self.atom_nf

        self.T = timesteps
        self.parametrization = parametrization

        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.register_buffer('buffer', torch.zeros(1))

        #  distribution of nodes
        self.size_distribution = DistributionNodes(size_histogram)

        # indicate if virtual nodes are present
        self.vnode_idx = virtual_node_idx

        if noise_schedule != 'learned':
            self.check_issues_norm_values()

    def check_issues_norm_values(self, num_stdevs=8):
        zeros = torch.zeros((1, 1))
        gamma_0 = self.gamma(zeros)
        sigma_0 = self.sigma(gamma_0, target_tensor=zeros).item()

        # Checked if 1 / norm_value is still larger than 10 * standard
        # deviation.
        norm_value = self.norm_values[1]

        if sigma_0 * num_stdevs > 1. / norm_value:
            raise ValueError(
                f'Value for normalization value {norm_value} probably too '
                f'large with sigma_0 {sigma_0:.5f} and '
                f'1 / norm_value = {1. / norm_value}')

    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor,
                                  gamma_s: torch.Tensor,
                                  target_tensor: torch.Tensor):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.
        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = self.inflate_batch_array(
            -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t)), target_tensor
        )

        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(
            alpha_t_given_s, target_tensor)

        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    def kl_prior_with_pocket(self, xh_lig, xh_pocket, mask_lig, mask_pocket,
                             num_nodes):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice
        negligible in the loss. However, you compute it so that you see it when
        you've made a mistake in your noise schedule.
        """
        batch_size = len(num_nodes)

        # Compute the last alpha value, alpha_T.
        ones = torch.ones((batch_size, 1), device=xh_lig.device)
        gamma_T = self.gamma(ones)
        alpha_T = self.alpha(gamma_T, xh_lig)

        # Compute means.
        mu_T_lig = alpha_T[mask_lig] * xh_lig
        mu_T_lig_x, mu_T_lig_h = mu_T_lig[:, :self.n_dims], \
                                 mu_T_lig[:, self.n_dims:]

        # Compute standard deviations (only batch axis for x-part, inflated for h-part).
        sigma_T_x = self.sigma(gamma_T, mu_T_lig_x).squeeze()
        sigma_T_h = self.sigma(gamma_T, mu_T_lig_h).squeeze()

        # Compute means.
        mu_T_pocket = alpha_T[mask_pocket] * xh_pocket
        mu_T_pocket_x, mu_T_pocket_h = mu_T_pocket[:, :self.n_dims], \
                                       mu_T_pocket[:, self.n_dims:]

        # Compute KL for h-part.
        zeros_lig = torch.zeros_like(mu_T_lig_h)
        zeros_pocket = torch.zeros_like(mu_T_pocket_h)
        ones = torch.ones_like(sigma_T_h)
        mu_norm2 = self.sum_except_batch((mu_T_lig_h - zeros_lig) ** 2, mask_lig) + \
                   self.sum_except_batch((mu_T_pocket_h - zeros_pocket) ** 2, mask_pocket)
        kl_distance_h = self.gaussian_KL(mu_norm2, sigma_T_h, ones, d=1)

        # Compute KL for x-part.
        zeros_lig = torch.zeros_like(mu_T_lig_x)
        zeros_pocket = torch.zeros_like(mu_T_pocket_x)
        ones = torch.ones_like(sigma_T_x)
        mu_norm2 = self.sum_except_batch((mu_T_lig_x - zeros_lig) ** 2, mask_lig) + \
                   self.sum_except_batch((mu_T_pocket_x - zeros_pocket) ** 2, mask_pocket)
        subspace_d = self.subspace_dimensionality(num_nodes)
        kl_distance_x = self.gaussian_KL(mu_norm2, sigma_T_x, ones, subspace_d)

        return kl_distance_x + kl_distance_h

    def compute_x_pred(self, net_out, zt, gamma_t, batch_mask):
        """Commputes x_pred, i.e. the most likely prediction of x."""
        if self.parametrization == 'x':
            x_pred = net_out
        elif self.parametrization == 'eps':
            sigma_t = self.sigma(gamma_t, target_tensor=net_out)
            alpha_t = self.alpha(gamma_t, target_tensor=net_out)
            eps_t = net_out
            x_pred = 1. / alpha_t[batch_mask] * (zt - sigma_t[batch_mask] * eps_t)
        else:
            raise ValueError(self.parametrization)

        return x_pred

    def log_constants_p_x_given_z0(self, n_nodes, device):
        """Computes p(x|z0)."""

        batch_size = len(n_nodes)
        degrees_of_freedom_x = self.subspace_dimensionality(n_nodes)

        zeros = torch.zeros((batch_size, 1), device=device)
        gamma_0 = self.gamma(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_x * (- log_sigma_x - 0.5 * np.log(2 * np.pi))

    def log_pxh_given_z0_without_constants(
            self, ligand, z_0_lig, eps_lig, net_out_lig,
            pocket, z_0_pocket, eps_pocket, net_out_pocket,
            gamma_0, epsilon=1e-10):

        # Discrete properties are predicted directly from z_t.
        z_h_lig = z_0_lig[:, self.n_dims:]
        z_h_pocket = z_0_pocket[:, self.n_dims:]

        # Take only part over x.
        eps_lig_x = eps_lig[:, :self.n_dims]
        net_lig_x = net_out_lig[:, :self.n_dims]
        eps_pocket_x = eps_pocket[:, :self.n_dims]
        net_pocket_x = net_out_pocket[:, :self.n_dims]

        # Compute sigma_0 and rescale to the integer scale of the data.
        sigma_0 = self.sigma(gamma_0, target_tensor=z_0_lig)
        sigma_0_cat = sigma_0 * self.norm_values[1]

        # Computes the error for the distribution
        # N(x | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0),
        # the weighting in the epsilon parametrization is exactly '1'.
        log_p_x_given_z0_without_constants_ligand = -0.5 * (
            self.sum_except_batch((eps_lig_x - net_lig_x) ** 2, ligand['mask'])
        )

        log_p_x_given_z0_without_constants_pocket = -0.5 * (
            self.sum_except_batch((eps_pocket_x - net_pocket_x) ** 2,
                                  pocket['mask'])
        )

        # Compute delta indicator masks.
        # un-normalize
        ligand_onehot = ligand['one_hot'] * self.norm_values[1] + self.norm_biases[1]
        pocket_onehot = pocket['one_hot'] * self.norm_values[1] + self.norm_biases[1]

        estimated_ligand_onehot = z_h_lig * self.norm_values[1] + self.norm_biases[1]
        estimated_pocket_onehot = z_h_pocket * self.norm_values[1] + self.norm_biases[1]

        # Centered h_cat around 1, since onehot encoded.
        centered_ligand_onehot = estimated_ligand_onehot - 1
        centered_pocket_onehot = estimated_pocket_onehot - 1

        # Compute integrals from 0.5 to 1.5 of the normal distribution
        # N(mean=z_h_cat, stdev=sigma_0_cat)
        log_ph_cat_proportional_ligand = torch.log(
            self.cdf_standard_gaussian((centered_ligand_onehot + 0.5) / sigma_0_cat[ligand['mask']])
            - self.cdf_standard_gaussian((centered_ligand_onehot - 0.5) / sigma_0_cat[ligand['mask']])
            + epsilon
        )
        log_ph_cat_proportional_pocket = torch.log(
            self.cdf_standard_gaussian((centered_pocket_onehot + 0.5) / sigma_0_cat[pocket['mask']])
            - self.cdf_standard_gaussian((centered_pocket_onehot - 0.5) / sigma_0_cat[pocket['mask']])
            + epsilon
        )

        # Normalize the distribution over the categories.
        log_Z = torch.logsumexp(log_ph_cat_proportional_ligand, dim=1,
                                keepdim=True)
        log_probabilities_ligand = log_ph_cat_proportional_ligand - log_Z

        log_Z = torch.logsumexp(log_ph_cat_proportional_pocket, dim=1,
                                keepdim=True)
        log_probabilities_pocket = log_ph_cat_proportional_pocket - log_Z

        # Select the log_prob of the current category using the onehot
        # representation.
        log_ph_given_z0_ligand = self.sum_except_batch(
            log_probabilities_ligand * ligand_onehot, ligand['mask'])
        log_ph_given_z0_pocket = self.sum_except_batch(
            log_probabilities_pocket * pocket_onehot, pocket['mask'])

        # Combine log probabilities of ligand and pocket for h.
        log_ph_given_z0 = log_ph_given_z0_ligand + log_ph_given_z0_pocket

        return log_p_x_given_z0_without_constants_ligand, \
               log_p_x_given_z0_without_constants_pocket, log_ph_given_z0

    def sample_p_xh_given_z0(self, z0_lig, z0_pocket, lig_mask, pocket_mask,
                             batch_size, fix_noise=False):
        """Samples x ~ p(x|z0)."""
        t_zeros = torch.zeros(size=(batch_size, 1), device=z0_lig.device)
        gamma_0 = self.gamma(t_zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0)
        net_out_lig, net_out_pocket = self.dynamics(
            z0_lig, z0_pocket, t_zeros, lig_mask, pocket_mask)

        # Compute mu for p(zs | zt).
        mu_x_lig = self.compute_x_pred(net_out_lig, z0_lig, gamma_0, lig_mask)
        mu_x_pocket = self.compute_x_pred(net_out_pocket, z0_pocket, gamma_0,
                                          pocket_mask)
        xh_lig, xh_pocket = self.sample_normal(mu_x_lig, mu_x_pocket, sigma_x,
                                               lig_mask, pocket_mask, fix_noise)

        x_lig, h_lig = self.unnormalize(
            xh_lig[:, :self.n_dims], z0_lig[:, self.n_dims:])
        x_pocket, h_pocket = self.unnormalize(
            xh_pocket[:, :self.n_dims], z0_pocket[:, self.n_dims:])

        h_lig = F.one_hot(torch.argmax(h_lig, dim=1), self.atom_nf)
        h_pocket = F.one_hot(torch.argmax(h_pocket, dim=1), self.residue_nf)

        return x_lig, h_lig, x_pocket, h_pocket

    def sample_normal(self, mu_lig, mu_pocket, sigma, lig_mask, pocket_mask,
                      fix_noise=False):
        """Samples from a Normal distribution."""
        if fix_noise:
            # bs = 1 if fix_noise else mu.size(0)
            raise NotImplementedError("fix_noise option isn't implemented yet")
        eps_lig, eps_pocket = self.sample_combined_position_feature_noise(
            lig_mask, pocket_mask)

        return mu_lig + sigma[lig_mask] * eps_lig, \
               mu_pocket + sigma[pocket_mask] * eps_pocket

    def noised_representation(self, xh_lig, xh_pocket, lig_mask, pocket_mask,
                              gamma_t):
        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, xh_lig)
        sigma_t = self.sigma(gamma_t, xh_lig)

        # Sample zt ~ Normal(alpha_t x, sigma_t)
        eps_lig, eps_pocket = self.sample_combined_position_feature_noise(
            lig_mask, pocket_mask)

        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        z_t_lig = alpha_t[lig_mask] * xh_lig + sigma_t[lig_mask] * eps_lig
        z_t_pocket = alpha_t[pocket_mask] * xh_pocket + \
                     sigma_t[pocket_mask] * eps_pocket

        return z_t_lig, z_t_pocket, eps_lig, eps_pocket

    def log_pN(self, N_lig, N_pocket):
        """
        Prior on the sample size for computing
        log p(x,h,N) = log p(x,h|N) + log p(N), where log p(x,h|N) is the
        model's output
        Args:
            N: array of sample sizes
        Returns:
            log p(N)
        """
        log_pN = self.size_distribution.log_prob(N_lig, N_pocket)
        return log_pN

    def delta_log_px(self, num_nodes):
        return -self.subspace_dimensionality(num_nodes) * \
               np.log(self.norm_values[0])

    def forward(self, ligand, pocket, return_info=False):
        """
        Computes the loss and NLL terms
        """
        # Normalize data, take into account volume change in x.
        ligand, pocket = self.normalize(ligand, pocket)

        # Likelihood change due to normalization
        delta_log_px = self.delta_log_px(ligand['size'] + pocket['size'])

        # Sample a timestep t for each example in batch
        # At evaluation time, loss_0 will be computed separately to decrease
        # variance in the estimator (costs two forward passes)
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(
            lowest_t, self.T + 1, size=(ligand['size'].size(0), 1),
            device=ligand['x'].device).float()
        s_int = t_int - 1  # previous timestep

        # Masks: important to compute log p(x | z0).
        t_is_zero = (t_int == 0).float()
        t_is_not_zero = 1 - t_is_zero

        # Normalize t to [0, 1]. Note that the negative
        # step of s will never be used, since then p(x | z0) is computed.
        s = s_int / self.T
        t = t_int / self.T

        # Compute gamma_s and gamma_t via the network.
        gamma_s = self.inflate_batch_array(self.gamma(s), ligand['x'])
        gamma_t = self.inflate_batch_array(self.gamma(t), ligand['x'])

        # Concatenate x, and h[categorical].
        xh_lig = torch.cat([ligand['x'], ligand['one_hot']], dim=1)
        xh_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)

        # Find noised representation
        z_t_lig, z_t_pocket, eps_t_lig, eps_t_pocket = \
            self.noised_representation(xh_lig, xh_pocket, ligand['mask'],
                                       pocket['mask'], gamma_t)

        # Neural net prediction.
        net_out_lig, net_out_pocket = self.dynamics(
            z_t_lig, z_t_pocket, t, ligand['mask'], pocket['mask'])

        # For LJ loss term
        xh_lig_hat = self.xh_given_zt_and_epsilon(z_t_lig, net_out_lig, gamma_t,
                                                  ligand['mask'])

        # Compute the L2 error.
        error_t_lig = self.sum_except_batch((eps_t_lig - net_out_lig) ** 2,
                                            ligand['mask'])

        error_t_pocket = self.sum_except_batch(
            (eps_t_pocket - net_out_pocket) ** 2, pocket['mask'])

        # Compute weighting with SNR: (1 - SNR(s-t)) for epsilon parametrization
        SNR_weight = (1 - self.SNR(gamma_s - gamma_t)).squeeze(1)
        assert error_t_lig.size() == SNR_weight.size()

        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)].
        neg_log_constants = -self.log_constants_p_x_given_z0(
            n_nodes=ligand['size'] + pocket['size'], device=error_t_lig.device)

        # The KL between q(zT | x) and p(zT) = Normal(0, 1).
        # Should be close to zero.
        kl_prior = self.kl_prior_with_pocket(
            xh_lig, xh_pocket, ligand['mask'], pocket['mask'],
            ligand['size'] + pocket['size'])

        if self.training:
            # Computes the L_0 term (even if gamma_t is not actually gamma_0)
            # and this will later be selected via masking.
            log_p_x_given_z0_without_constants_ligand, \
            log_p_x_given_z0_without_constants_pocket, log_ph_given_z0 = \
                self.log_pxh_given_z0_without_constants(
                    ligand, z_t_lig, eps_t_lig, net_out_lig,
                    pocket, z_t_pocket, eps_t_pocket, net_out_pocket, gamma_t)

            loss_0_x_ligand = -log_p_x_given_z0_without_constants_ligand * \
                              t_is_zero.squeeze()
            loss_0_x_pocket = -log_p_x_given_z0_without_constants_pocket * \
                              t_is_zero.squeeze()
            loss_0_h = -log_ph_given_z0 * t_is_zero.squeeze()

            # apply t_is_zero mask
            error_t_lig = error_t_lig * t_is_not_zero.squeeze()
            error_t_pocket = error_t_pocket * t_is_not_zero.squeeze()

        else:
            # Compute noise values for t = 0.
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), ligand['x'])

            # Sample z_0 given x, h for timestep t, from q(z_t | x, h)
            z_0_lig, z_0_pocket, eps_0_lig, eps_0_pocket = \
                self.noised_representation(xh_lig, xh_pocket, ligand['mask'],
                                           pocket['mask'], gamma_0)

            net_out_0_lig, net_out_0_pocket = self.dynamics(
                z_0_lig, z_0_pocket, t_zeros, ligand['mask'], pocket['mask'])

            log_p_x_given_z0_without_constants_ligand, \
            log_p_x_given_z0_without_constants_pocket, log_ph_given_z0 = \
                self.log_pxh_given_z0_without_constants(
                    ligand, z_0_lig, eps_0_lig, net_out_0_lig,
                    pocket, z_0_pocket, eps_0_pocket, net_out_0_pocket, gamma_0)
            loss_0_x_ligand = -log_p_x_given_z0_without_constants_ligand
            loss_0_x_pocket = -log_p_x_given_z0_without_constants_pocket
            loss_0_h = -log_ph_given_z0

        # sample size prior
        log_pN = self.log_pN(ligand['size'], pocket['size'])

        info = {
            'eps_hat_lig_x': scatter_mean(
                net_out_lig[:, :self.n_dims].abs().mean(1), ligand['mask'],
                dim=0).mean(),
            'eps_hat_lig_h': scatter_mean(
                net_out_lig[:, self.n_dims:].abs().mean(1), ligand['mask'],
                dim=0).mean(),
            'eps_hat_pocket_x': scatter_mean(
                net_out_pocket[:, :self.n_dims].abs().mean(1), pocket['mask'],
                dim=0).mean(),
            'eps_hat_pocket_h': scatter_mean(
                net_out_pocket[:, self.n_dims:].abs().mean(1), pocket['mask'],
                dim=0).mean(),
        }
        loss_terms = (delta_log_px, error_t_lig, error_t_pocket, SNR_weight,
                      loss_0_x_ligand, loss_0_x_pocket, loss_0_h,
                      neg_log_constants, kl_prior, log_pN,
                      t_int.squeeze(), xh_lig_hat)
        return (*loss_terms, info) if return_info else loss_terms

    def xh_given_zt_and_epsilon(self, z_t, epsilon, gamma_t, batch_mask):
        """ Equation (7) in the EDM paper """
        alpha_t = self.alpha(gamma_t, z_t)
        sigma_t = self.sigma(gamma_t, z_t)
        xh = z_t / alpha_t[batch_mask] - epsilon * sigma_t[batch_mask] / \
             alpha_t[batch_mask]
        return xh

    def sample_p_zt_given_zs(self, zs_lig, zs_pocket, ligand_mask, pocket_mask,
                             gamma_t, gamma_s, fix_noise=False):
        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zs_lig)

        mu_lig = alpha_t_given_s[ligand_mask] * zs_lig
        mu_pocket = alpha_t_given_s[pocket_mask] * zs_pocket
        zt_lig, zt_pocket = self.sample_normal(
            mu_lig, mu_pocket, sigma_t_given_s, ligand_mask, pocket_mask,
            fix_noise)

        # Remove center of mass
        zt_x = self.remove_mean_batch(
            torch.cat((zt_lig[:, :self.n_dims], zt_pocket[:, :self.n_dims]),
                      dim=0),
            torch.cat((ligand_mask, pocket_mask))
        )
        zt_lig = torch.cat((zt_x[:len(ligand_mask)],
                            zt_lig[:, self.n_dims:]), dim=1)
        zt_pocket = torch.cat((zt_x[len(ligand_mask):],
                               zt_pocket[:, self.n_dims:]), dim=1)

        return zt_lig, zt_pocket

    def sample_p_zs_given_zt(self, s, t, zt_lig, zt_pocket, ligand_mask,
                             pocket_mask, fix_noise=False):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt_lig)

        sigma_s = self.sigma(gamma_s, target_tensor=zt_lig)
        sigma_t = self.sigma(gamma_t, target_tensor=zt_lig)

        # Neural net prediction.
        eps_t_lig, eps_t_pocket = self.dynamics(
            zt_lig, zt_pocket, t, ligand_mask, pocket_mask)

        # Compute mu for p(zs | zt).
        combined_mask = torch.cat((ligand_mask, pocket_mask))
        self.assert_mean_zero_with_mask(
            torch.cat((zt_lig[:, :self.n_dims],
                       zt_pocket[:, :self.n_dims]), dim=0),
            combined_mask)
        self.assert_mean_zero_with_mask(
            torch.cat((eps_t_lig[:, :self.n_dims],
                       eps_t_pocket[:, :self.n_dims]), dim=0),
            combined_mask)

        # Note: mu_{t->s} = 1 / alpha_{t|s} z_t - sigma_{t|s}^2 / sigma_t / alpha_{t|s} epsilon
        # follows from the definition of mu_{t->s} and Equ. (7) in the EDM paper
        mu_lig = zt_lig / alpha_t_given_s[ligand_mask] - \
                 (sigma2_t_given_s / alpha_t_given_s / sigma_t)[ligand_mask] * \
                 eps_t_lig
        mu_pocket = zt_pocket / alpha_t_given_s[pocket_mask] - \
                    (sigma2_t_given_s / alpha_t_given_s / sigma_t)[pocket_mask] * \
                    eps_t_pocket

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the paramters derived from zt.
        zs_lig, zs_pocket = self.sample_normal(mu_lig, mu_pocket, sigma,
                                               ligand_mask, pocket_mask,
                                               fix_noise)

        # Project down to avoid numerical runaway of the center of gravity.
        zs_x = self.remove_mean_batch(
            torch.cat((zs_lig[:, :self.n_dims],
                       zs_pocket[:, :self.n_dims]), dim=0),
            torch.cat((ligand_mask, pocket_mask))
        )
        zs_lig = torch.cat((zs_x[:len(ligand_mask)],
                            zs_lig[:, self.n_dims:]), dim=1)
        zs_pocket = torch.cat((zs_x[len(ligand_mask):],
                               zs_pocket[:, self.n_dims:]), dim=1)
        return zs_lig, zs_pocket

    def sample_combined_position_feature_noise(self, lig_indices,
                                               pocket_indices):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise
        for z_h.
        """
        z_x = self.sample_center_gravity_zero_gaussian_batch(
            size=(len(lig_indices) + len(pocket_indices), self.n_dims),
            lig_indices=lig_indices,
            pocket_indices=pocket_indices
        )
        z_h_lig = self.sample_gaussian(
            size=(len(lig_indices), self.atom_nf),
            device=lig_indices.device)
        z_lig = torch.cat([z_x[:len(lig_indices)], z_h_lig], dim=1)
        z_h_pocket = self.sample_gaussian(
            size=(len(pocket_indices), self.residue_nf),
            device=pocket_indices.device)
        z_pocket = torch.cat([z_x[len(lig_indices):], z_h_pocket], dim=1)
        return z_lig, z_pocket

    @torch.no_grad()
    def sample(self, n_samples, num_nodes_lig, num_nodes_pocket,
               return_frames=1, timesteps=None, device='cpu'):
        """
        Draw samples from the generative model. Optionally, return intermediate
        states for visualization purposes.
        """
        timesteps = self.T if timesteps is None else timesteps
        assert 0 < return_frames <= timesteps
        assert timesteps % return_frames == 0

        lig_mask = utils.num_nodes_to_batch_mask(n_samples, num_nodes_lig,
                                                 device)
        pocket_mask = utils.num_nodes_to_batch_mask(n_samples, num_nodes_pocket,
                                                    device)

        combined_mask = torch.cat((lig_mask, pocket_mask))

        z_lig, z_pocket = self.sample_combined_position_feature_noise(
            lig_mask, pocket_mask)

        self.assert_mean_zero_with_mask(
            torch.cat((z_lig[:, :self.n_dims], z_pocket[:, :self.n_dims]), dim=0),
            combined_mask
        )

        out_lig = torch.zeros((return_frames,) + z_lig.size(),
                              device=z_lig.device)
        out_pocket = torch.zeros((return_frames,) + z_pocket.size(),
                                 device=z_pocket.device)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, timesteps)):
            s_array = torch.full((n_samples, 1), fill_value=s,
                                 device=z_lig.device)
            t_array = s_array + 1
            s_array = s_array / timesteps
            t_array = t_array / timesteps

            z_lig, z_pocket = self.sample_p_zs_given_zt(
                s_array, t_array, z_lig, z_pocket, lig_mask, pocket_mask)

            # save frame
            if (s * return_frames) % timesteps == 0:
                idx = (s * return_frames) // timesteps
                out_lig[idx], out_pocket[idx] = \
                    self.unnormalize_z(z_lig, z_pocket)

        # Finally sample p(x, h | z_0).
        x_lig, h_lig, x_pocket, h_pocket = self.sample_p_xh_given_z0(
            z_lig, z_pocket, lig_mask, pocket_mask, n_samples)

        self.assert_mean_zero_with_mask(
            torch.cat((x_lig, x_pocket), dim=0), combined_mask
        )

        # Correct CoM drift for examples without intermediate states
        if return_frames == 1:
            x = torch.cat((x_lig, x_pocket))
            max_cog = scatter_add(x, combined_mask, dim=0).abs().max().item()
            if max_cog > 5e-2:
                print(f'Warning CoG drift with error {max_cog:.3f}. Projecting '
                      f'the positions down.')
                x = self.remove_mean_batch(x, combined_mask)
                x_lig, x_pocket = x[:len(x_lig)], x[len(x_lig):]

        # Overwrite last frame with the resulting x and h.
        out_lig[0] = torch.cat([x_lig, h_lig], dim=1)
        out_pocket[0] = torch.cat([x_pocket, h_pocket], dim=1)

        # remove frame dimension if only the final molecule is returned
        return out_lig.squeeze(0), out_pocket.squeeze(0), lig_mask, pocket_mask

    def get_repaint_schedule(self, resamplings, jump_length, timesteps):
        """ Each integer in the schedule list describes how many denoising steps
        need to be applied before jumping back """
        repaint_schedule = []
        curr_t = 0
        while curr_t < timesteps:
            if curr_t + jump_length < timesteps:
                if len(repaint_schedule) > 0:
                    repaint_schedule[-1] += jump_length
                    repaint_schedule.extend([jump_length] * (resamplings - 1))
                else:
                    repaint_schedule.extend([jump_length] * resamplings)
                curr_t += jump_length
            else:
                residual = (timesteps - curr_t)
                if len(repaint_schedule) > 0:
                    repaint_schedule[-1] += residual
                else:
                    repaint_schedule.append(residual)
                curr_t += residual

        return list(reversed(repaint_schedule))

    @torch.no_grad()
    def inpaint(self, ligand, pocket, lig_fixed, pocket_fixed, resamplings=1,
                jump_length=1, return_frames=1, timesteps=None):
        """
        Draw samples from the generative model while fixing parts of the input.
        Optionally, return intermediate states for visualization purposes.
        See:
        Lugmayr, Andreas, et al.
        "Repaint: Inpainting using denoising diffusion probabilistic models."
        Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
        Recognition. 2022.
        """
        timesteps = self.T if timesteps is None else timesteps
        assert 0 < return_frames <= timesteps
        assert timesteps % return_frames == 0
        assert jump_length == 1 or return_frames == 1, \
            "Chain visualization is only implemented for jump_length=1"

        if len(lig_fixed.size()) == 1:
            lig_fixed = lig_fixed.unsqueeze(1)
        if len(pocket_fixed.size()) == 1:
            pocket_fixed = pocket_fixed.unsqueeze(1)

        ligand, pocket = self.normalize(ligand, pocket)

        n_samples = len(ligand['size'])
        combined_mask = torch.cat((ligand['mask'], pocket['mask']))
        xh0_lig = torch.cat([ligand['x'], ligand['one_hot']], dim=1)
        xh0_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)

        # Center initial system, subtract COM of known parts
        mean_known = scatter_mean(
            torch.cat((ligand['x'][lig_fixed.bool().view(-1)],
                       pocket['x'][pocket_fixed.bool().view(-1)])),
            torch.cat((ligand['mask'][lig_fixed.bool().view(-1)],
                       pocket['mask'][pocket_fixed.bool().view(-1)])),
            dim=0
        )
        xh0_lig[:, :self.n_dims] = \
            xh0_lig[:, :self.n_dims] - mean_known[ligand['mask']]
        xh0_pocket[:, :self.n_dims] = \
            xh0_pocket[:, :self.n_dims] - mean_known[pocket['mask']]

        # Noised representation at step t=T
        z_lig, z_pocket = self.sample_combined_position_feature_noise(
            ligand['mask'], pocket['mask'])

        # Output tensors
        out_lig = torch.zeros((return_frames,) + z_lig.size(),
                              device=z_lig.device)
        out_pocket = torch.zeros((return_frames,) + z_pocket.size(),
                                 device=z_pocket.device)

        # Iteratively sample according to a pre-defined schedule
        schedule = self.get_repaint_schedule(resamplings, jump_length, timesteps)
        s = timesteps - 1
        for i, n_denoise_steps in enumerate(schedule):
            for j in range(n_denoise_steps):
                # Denoise one time step: t -> s
                s_array = torch.full((n_samples, 1), fill_value=s,
                                     device=z_lig.device)
                t_array = s_array + 1
                s_array = s_array / timesteps
                t_array = t_array / timesteps

                # sample known nodes from the input
                gamma_s = self.inflate_batch_array(self.gamma(s_array),
                                                   ligand['x'])
                z_lig_known, z_pocket_known, _, _ = self.noised_representation(
                    xh0_lig, xh0_pocket, ligand['mask'], pocket['mask'], gamma_s)

                # sample inpainted part
                z_lig_unknown, z_pocket_unknown = self.sample_p_zs_given_zt(
                    s_array, t_array, z_lig, z_pocket, ligand['mask'],
                    pocket['mask'])

                # move center of mass of the noised part to the center of mass
                # of the corresponding denoised part before combining them
                # -> the resulting system should be COM-free
                com_noised = scatter_mean(
                    torch.cat((z_lig_known[:, :self.n_dims][lig_fixed.bool().view(-1)],
                               z_pocket_known[:, :self.n_dims][pocket_fixed.bool().view(-1)])),
                    torch.cat((ligand['mask'][lig_fixed.bool().view(-1)],
                               pocket['mask'][pocket_fixed.bool().view(-1)])),
                    dim=0
                )
                com_denoised = scatter_mean(
                    torch.cat((z_lig_unknown[:, :self.n_dims][lig_fixed.bool().view(-1)],
                               z_pocket_unknown[:, :self.n_dims][pocket_fixed.bool().view(-1)])),
                    torch.cat((ligand['mask'][lig_fixed.bool().view(-1)],
                               pocket['mask'][pocket_fixed.bool().view(-1)])),
                    dim=0
                )
                z_lig_known[:, :self.n_dims] = \
                    z_lig_known[:, :self.n_dims] + (com_denoised - com_noised)[ligand['mask']]
                z_pocket_known[:, :self.n_dims] = \
                    z_pocket_known[:, :self.n_dims] + (com_denoised - com_noised)[pocket['mask']]

                # combine
                z_lig = z_lig_known * lig_fixed + \
                        z_lig_unknown * (1 - lig_fixed)
                z_pocket = z_pocket_known * pocket_fixed + \
                           z_pocket_unknown * (1 - pocket_fixed)

                self.assert_mean_zero_with_mask(
                    torch.cat((z_lig[:, :self.n_dims],
                               z_pocket[:, :self.n_dims]), dim=0), combined_mask
                )

                # save frame at the end of a resample cycle
                if n_denoise_steps > jump_length or i == len(schedule) - 1:
                    if (s * return_frames) % timesteps == 0:
                        idx = (s * return_frames) // timesteps
                        out_lig[idx], out_pocket[idx] = \
                            self.unnormalize_z(z_lig, z_pocket)

                # Noise combined representation
                if j == n_denoise_steps - 1 and i < len(schedule) - 1:
                    # Go back jump_length steps
                    t = s + jump_length
                    t_array = torch.full((n_samples, 1), fill_value=t,
                                         device=z_lig.device)
                    t_array = t_array / timesteps

                    gamma_s = self.inflate_batch_array(self.gamma(s_array),
                                                       ligand['x'])
                    gamma_t = self.inflate_batch_array(self.gamma(t_array),
                                                       ligand['x'])

                    z_lig, z_pocket = self.sample_p_zt_given_zs(
                        z_lig, z_pocket, ligand['mask'], pocket['mask'],
                        gamma_t, gamma_s)

                    s = t

                s -= 1

        # Finally sample p(x, h | z_0).
        x_lig, h_lig, x_pocket, h_pocket = self.sample_p_xh_given_z0(
            z_lig, z_pocket, ligand['mask'], pocket['mask'], n_samples)

        self.assert_mean_zero_with_mask(
            torch.cat((x_lig, x_pocket), dim=0), combined_mask
        )

        # Correct CoM drift for examples without intermediate states
        if return_frames == 1:
            x = torch.cat((x_lig, x_pocket))
            max_cog = scatter_add(x, combined_mask, dim=0).abs().max().item()
            if max_cog > 5e-2:
                print(f'Warning CoG drift with error {max_cog:.3f}. Projecting '
                      f'the positions down.')
                x = self.remove_mean_batch(x, combined_mask)
                x_lig, x_pocket = x[:len(x_lig)], x[len(x_lig):]

        # Overwrite last frame with the resulting x and h.
        out_lig[0] = torch.cat([x_lig, h_lig], dim=1)
        out_pocket[0] = torch.cat([x_pocket, h_pocket], dim=1)

        # remove frame dimension if only the final molecule is returned
        return out_lig.squeeze(0), out_pocket.squeeze(0), ligand['mask'], \
               pocket['mask']

    @staticmethod
    def gaussian_KL(q_mu_minus_p_mu_squared, q_sigma, p_sigma, d):
        """Computes the KL distance between two normal distributions.
            Args:
                q_mu_minus_p_mu_squared: Squared difference between mean of
                    distribution q and distribution p: ||mu_q - mu_p||^2
                q_sigma: Standard deviation of distribution q.
                p_sigma: Standard deviation of distribution p.
                d: dimension
            Returns:
                The KL distance
            """
        return d * torch.log(p_sigma / q_sigma) + \
               0.5 * (d * q_sigma ** 2 + q_mu_minus_p_mu_squared) / \
               (p_sigma ** 2) - 0.5 * d

    @staticmethod
    def inflate_batch_array(array, target):
        """
        Inflates the batch array (array) with only a single axis
        (i.e. shape = (batch_size,), or possibly more empty axes
        (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)),
                                        target_tensor)

    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)),
                                        target_tensor)

    @staticmethod
    def SNR(gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)

    def normalize(self, ligand=None, pocket=None):
        if ligand is not None:
            ligand['x'] = ligand['x'] / self.norm_values[0]

            # Casting to float in case h still has long or int type.
            ligand['one_hot'] = \
                (ligand['one_hot'].float() - self.norm_biases[1]) / \
                self.norm_values[1]

        if pocket is not None:
            pocket['x'] = pocket['x'] / self.norm_values[0]
            pocket['one_hot'] = \
                (pocket['one_hot'].float() - self.norm_biases[1]) / \
                self.norm_values[1]

        return ligand, pocket

    def unnormalize(self, x, h_cat):
        x = x * self.norm_values[0]
        h_cat = h_cat * self.norm_values[1] + self.norm_biases[1]

        return x, h_cat

    def unnormalize_z(self, z_lig, z_pocket):
        # Parse from z
        x_lig, h_lig = z_lig[:, :self.n_dims], z_lig[:, self.n_dims:]
        x_pocket, h_pocket = z_pocket[:, :self.n_dims], z_pocket[:, self.n_dims:]

        # Unnormalize
        x_lig, h_lig = self.unnormalize(x_lig, h_lig)
        x_pocket, h_pocket = self.unnormalize(x_pocket, h_pocket)
        return torch.cat([x_lig, h_lig], dim=1), \
               torch.cat([x_pocket, h_pocket], dim=1)

    def subspace_dimensionality(self, input_size):
        """Compute the dimensionality on translation-invariant linear subspace
        where distributions on x are defined."""
        return (input_size - 1) * self.n_dims

    @staticmethod
    def remove_mean_batch(x, indices):
        mean = scatter_mean(x, indices, dim=0)
        x = x - mean[indices]
        return x

    @staticmethod
    def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
        largest_value = x.abs().max().item()
        error = scatter_add(x, node_mask, dim=0).abs().max().item()
        rel_error = error / (largest_value + eps)
        assert rel_error < 1e-2, f'Mean is not zero, relative_error {rel_error}'

    @staticmethod
    def sample_center_gravity_zero_gaussian_batch(size, lig_indices,
                                                  pocket_indices):
        assert len(size) == 2
        x = torch.randn(size, device=lig_indices.device)

        # This projection only works because Gaussian is rotation invariant
        # around zero and samples are independent!
        x_projected = EnVariationalDiffusion.remove_mean_batch(
            x, torch.cat((lig_indices, pocket_indices)))
        return x_projected

    @staticmethod
    def sum_except_batch(x, indices):
        return scatter_add(x.sum(-1), indices, dim=0)

    @staticmethod
    def cdf_standard_gaussian(x):
        return 0.5 * (1. + torch.erf(x / math.sqrt(2)))

    @staticmethod
    def sample_gaussian(size, device):
        x = torch.randn(size, device=device)
        return x


class DistributionNodes:
    def __init__(self, histogram):

        histogram = torch.tensor(histogram).float()
        histogram = histogram + 1e-3  # for numerical stability

        prob = histogram / histogram.sum()

        self.idx_to_n_nodes = torch.tensor(
            [[(i, j) for j in range(prob.shape[1])] for i in range(prob.shape[0])]
        ).view(-1, 2)

        self.n_nodes_to_idx = {tuple(x.tolist()): i
                               for i, x in enumerate(self.idx_to_n_nodes)}

        self.prob = prob
        self.m = torch.distributions.Categorical(self.prob.view(-1),
                                                 validate_args=True)

        self.n1_given_n2 = \
            [torch.distributions.Categorical(prob[:, j], validate_args=True)
             for j in range(prob.shape[1])]
        self.n2_given_n1 = \
            [torch.distributions.Categorical(prob[i, :], validate_args=True)
             for i in range(prob.shape[0])]

        # entropy = -torch.sum(self.prob.view(-1) * torch.log(self.prob.view(-1) + 1e-30))
        entropy = self.m.entropy()
        print("Entropy of n_nodes: H[N]", entropy.item())

    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        num_nodes_lig, num_nodes_pocket = self.idx_to_n_nodes[idx].T
        return num_nodes_lig, num_nodes_pocket

    def sample_conditional(self, n1=None, n2=None):
        assert (n1 is None) ^ (n2 is None), \
            "Exactly one input argument must be None"

        m = self.n1_given_n2 if n2 is not None else self.n2_given_n1
        c = n2 if n2 is not None else n1

        return torch.tensor([m[i].sample() for i in c], device=c.device)

    def log_prob(self, batch_n_nodes_1, batch_n_nodes_2):
        assert len(batch_n_nodes_1.size()) == 1
        assert len(batch_n_nodes_2.size()) == 1

        idx = torch.tensor(
            [self.n_nodes_to_idx[(n1, n2)]
             for n1, n2 in zip(batch_n_nodes_1.tolist(), batch_n_nodes_2.tolist())]
        )

        # log_probs = torch.log(self.prob.view(-1)[idx] + 1e-30)
        log_probs = self.m.log_prob(idx)

        return log_probs.to(batch_n_nodes_1.device)

    def log_prob_n1_given_n2(self, n1, n2):
        assert len(n1.size()) == 1
        assert len(n2.size()) == 1
        log_probs = torch.stack([self.n1_given_n2[c].log_prob(i.cpu())
                                 for i, c in zip(n1, n2)])
        return log_probs.to(n1.device)

    def log_prob_n2_given_n1(self, n2, n1):
        assert len(n2.size()) == 1
        assert len(n1.size()) == 1
        log_probs = torch.stack([self.n2_given_n1[c].log_prob(i.cpu())
                                 for i, c in zip(n2, n1)])
        return log_probs.to(n2.device)


class PositiveLinear(torch.nn.Module):
    """Linear layer with weights forced to be positive."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_init_offset: int = -2):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        positive_weight = F.softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)


class GammaNetwork(torch.nn.Module):
    """The gamma network models a monotonic increasing function.
    Construction as in the VDM paper."""
    def __init__(self):
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        self.gamma_0 = torch.nn.Parameter(torch.tensor([-5.]))
        self.gamma_1 = torch.nn.Parameter(torch.tensor([10.]))
        self.show_schedule()

    def show_schedule(self, num_steps=50):
        t = torch.linspace(0, 1, num_steps).view(num_steps, 1)
        gamma = self.forward(t)
        print('Gamma schedule:')
        print(gamma.detach().cpu().numpy().reshape(num_steps))

    def gamma_tilde(self, t):
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    def forward(self, t):
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        # Not super efficient.
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # Normalize to [0, 1]
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (
                gamma_tilde_1 - gamma_tilde_0)

        # Rescale to [gamma_0, gamma_1]
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma

        return gamma


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1.
    This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power))**2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined
    (non-learned) noise schedules.
    """
    def __init__(self, noise_schedule, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            alphas2 = cosine_beta_schedule(timesteps)
        elif 'polynomial' in noise_schedule:
            splits = noise_schedule.split('_')
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        else:
            raise ValueError(noise_schedule)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]
