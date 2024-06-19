import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from Bio.PDB import PDBParser
from rdkit import Chem
from torch_scatter import scatter_mean
from openbabel import openbabel
openbabel.obErrorLog.StopLogging()  # suppress OpenBabel messages

import utils
from lightning_modules import LigandPocketDDPM
from constants import FLOAT_TYPE, INT_TYPE
from analysis.molecule_builder import build_molecule, process_molecule


def prepare_from_sdf_files(sdf_files, atom_encoder):

    ligand_coords = []
    atom_one_hot = []
    for file in sdf_files:
        rdmol = Chem.SDMolSupplier(str(file), sanitize=False)[0]
        ligand_coords.append(
            torch.from_numpy(rdmol.GetConformer().GetPositions()).float()
        )
        types = torch.tensor([atom_encoder[a.GetSymbol()] for a in rdmol.GetAtoms()])
        atom_one_hot.append(
            F.one_hot(types, num_classes=len(atom_encoder))
        )

    return torch.cat(ligand_coords, dim=0), torch.cat(atom_one_hot, dim=0)


def prepare_ligand_from_pdb(biopython_atoms, atom_encoder):

    coord = torch.tensor(np.array([a.get_coord()
                                   for a in biopython_atoms]), dtype=FLOAT_TYPE)
    types = torch.tensor([atom_encoder[a.element.capitalize()]
                          for a in biopython_atoms])
    one_hot = F.one_hot(types, num_classes=len(atom_encoder))

    return coord, one_hot


def prepare_substructure(ref_ligand, fix_atoms, pdb_model):

    if fix_atoms[0].endswith(".sdf"):
        # ligand as sdf file
        coord, one_hot = prepare_from_sdf_files(fix_atoms, model.lig_type_encoder)

    else:
        # ligand contained in PDB; given in <chain>:<resi> format
        chain, resi = ref_ligand.split(':')
        ligand = utils.get_residue_with_resi(pdb_model[chain], int(resi))
        fixed_atoms = [a for a in ligand.get_atoms() if a.get_name() in set(fix_atoms)]
        coord, one_hot = prepare_ligand_from_pdb(fixed_atoms, model.lig_type_encoder)

    return coord, one_hot


def inpaint_ligand(model, pdb_file, n_samples, ligand, fix_atoms,
                   add_n_nodes=None, center='ligand', sanitize=False,
                   largest_frag=False, relax_iter=0, timesteps=None,
                   resamplings=1, save_traj=False):
    """
    Generate ligands given a pocket
    Args:
        model: Lightning model
        pdb_file: PDB filename
        n_samples: number of samples
        ligand: reference ligand given in <chain>:<resi> format if the ligand is
                contained in the PDB file, or path to an SDF file that
                contains the ligand; used to define the pocket
        fix_atoms: ligand atoms that should be fixed, e.g. "C1 N6 C5 C12"
        center: 'ligand' or 'pocket'
        add_n_nodes: number of ligand nodes to add, sampled randomly if 'None'
        sanitize: whether to sanitize molecules or not
        largest_frag: only return the largest fragment
        relax_iter: number of force field optimization steps
        timesteps: number of denoising steps, use training value if None
        resamplings: number of resampling iterations
        save_traj: save intermediate states to visualize a denoising trajectory
    Returns:
        list of molecules
    """
    if save_traj and n_samples > 1:
        raise NotImplementedError("Can only visualize trajectory with "
                                  "n_samples=1.")
    frames = timesteps if save_traj else 1
    sanitize = False if save_traj else sanitize
    relax_iter = 0 if save_traj else relax_iter
    largest_frag = False if save_traj else largest_frag

    # Load PDB
    pdb_model = PDBParser(QUIET=True).get_structure('', pdb_file)[0]

    # Define pocket based on reference ligand
    residues = utils.get_pocket_from_ligand(pdb_model, ligand)
    pocket = model.prepare_pocket(residues, repeats=n_samples)

    # Get fixed ligand substructure
    x_fixed, one_hot_fixed = prepare_substructure(ligand, fix_atoms, pdb_model)
    n_fixed = len(x_fixed)

    if add_n_nodes is None:
        num_nodes_lig = model.ddpm.size_distribution.sample_conditional(
            n1=None, n2=pocket['size'])
        num_nodes_lig = torch.clamp(num_nodes_lig, min=n_fixed)
    else:
        num_nodes_lig = torch.ones(n_samples, dtype=int) * n_fixed + add_n_nodes

    ligand_mask = utils.num_nodes_to_batch_mask(
        len(num_nodes_lig), num_nodes_lig, model.device)

    ligand = {
        'x': torch.zeros((len(ligand_mask), model.x_dims),
                         device=model.device, dtype=FLOAT_TYPE),
        'one_hot': torch.zeros((len(ligand_mask), model.atom_nf),
                               device=model.device, dtype=FLOAT_TYPE),
        'size': num_nodes_lig,
        'mask': ligand_mask
    }

    # fill in fixed atoms
    lig_fixed = torch.zeros_like(ligand_mask)
    for i in range(n_samples):
        sele = (ligand_mask == i)

        x_new = ligand['x'][sele]
        x_new[:n_fixed] = x_fixed
        ligand['x'][sele] = x_new

        h_new = ligand['one_hot'][sele]
        h_new[:n_fixed] = one_hot_fixed
        ligand['one_hot'][sele] = h_new

        fixed_new = lig_fixed[sele]
        fixed_new[:n_fixed] = 1
        lig_fixed[sele] = fixed_new

    # Pocket's center of mass
    pocket_com_before = scatter_mean(pocket['x'], pocket['mask'], dim=0)

    # Run sampling
    xh_lig, xh_pocket, lig_mask, pocket_mask = model.ddpm.inpaint(
        ligand, pocket, lig_fixed, center=center,
        resamplings=resamplings, timesteps=timesteps, return_frames=frames)

    # Treat intermediate states as molecules for downstream processing
    if save_traj:
        xh_lig = utils.reverse_tensor(xh_lig)
        xh_pocket = utils.reverse_tensor(xh_pocket)

        lig_mask = torch.arange(xh_lig.size(0), device=model.device
                                ).repeat_interleave(len(lig_mask))
        pocket_mask = torch.arange(xh_pocket.size(0), device=model.device
                                   ).repeat_interleave(len(pocket_mask))

        xh_lig = xh_lig.view(-1, xh_lig.size(2))
        xh_pocket = xh_pocket.view(-1, xh_pocket.size(2))

    # Move generated molecule back to the original pocket position
    pocket_com_after = scatter_mean(xh_pocket[:, :model.x_dims], pocket_mask, dim=0)

    xh_pocket[:, :model.x_dims] += \
        (pocket_com_before - pocket_com_after)[pocket_mask]
    xh_lig[:, :model.x_dims] += \
        (pocket_com_before - pocket_com_after)[lig_mask]

    # Build mol objects
    x = xh_lig[:, :model.x_dims].detach().cpu()
    atom_type = xh_lig[:, model.x_dims:].argmax(1).detach().cpu()

    molecules = []
    for mol_pc in zip(utils.batch_to_list(x, lig_mask),
                      utils.batch_to_list(atom_type, lig_mask)):

        mol = build_molecule(*mol_pc, model.dataset_info, add_coords=True)
        mol = process_molecule(mol,
                               add_hydrogens=False,
                               sanitize=sanitize,
                               relax_iter=relax_iter,
                               largest_frag=largest_frag)
        if mol is not None:
            molecules.append(mol)

    return molecules


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--pdbfile', type=str)
    parser.add_argument('--ref_ligand', type=str, default=None)
    parser.add_argument('--fix_atoms', type=str, nargs='+', default=None)
    parser.add_argument('--center', type=str, default='ligand', choices={'ligand', 'pocket'})
    parser.add_argument('--outfile', type=Path)
    parser.add_argument('--n_samples', type=int, default=20)
    parser.add_argument('--add_n_nodes', type=int, default=None)
    parser.add_argument('--relax', action='store_true')
    parser.add_argument('--sanitize', action='store_true')
    parser.add_argument('--resamplings', type=int, default=20)
    parser.add_argument('--timesteps', type=int, default=50)
    parser.add_argument('--save_traj', action='store_true')
    args = parser.parse_args()

    pdb_id = Path(args.pdbfile).stem

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = LigandPocketDDPM.load_from_checkpoint(
        args.checkpoint, map_location=device)
    model = model.to(device)

    molecules = inpaint_ligand(model, args.pdbfile, args.n_samples,
                               args.ref_ligand, args.fix_atoms,
                               args.add_n_nodes, center=args.center,
                               sanitize=args.sanitize,
                               largest_frag=False,
                               relax_iter=(200 if args.relax else 0),
                               timesteps=args.timesteps,
                               resamplings=args.resamplings,
                               save_traj=args.save_traj)

    # Make SDF files
    utils.write_sdf_file(args.outfile, molecules)
