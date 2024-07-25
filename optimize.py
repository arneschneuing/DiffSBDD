import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from Bio.PDB import PDBParser
from rdkit import Chem
import pandas as pd
import random
from torch_scatter import scatter_mean
from openbabel import openbabel
openbabel.obErrorLog.StopLogging()  # suppress OpenBabel messages

import utils
from lightning_modules import LigandPocketDDPM
from constants import FLOAT_TYPE, INT_TYPE
from analysis.molecule_builder import build_molecule, process_molecule
from analysis.metrics import MoleculeProperties


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


def prepare_ligands_from_mols(mols, atom_encoder, device='cpu'):

    ligand_coords = []
    atom_one_hots = []
    masks = []
    sizes = []
    for i, mol in enumerate(mols):
        coord = torch.tensor(mol.GetConformer().GetPositions(), dtype=FLOAT_TYPE)
        types = torch.tensor([atom_encoder[a.GetSymbol()] for a in mol.GetAtoms()], dtype=INT_TYPE)
        one_hot = F.one_hot(types, num_classes=len(atom_encoder))
        mask = torch.ones(len(types), dtype=INT_TYPE) * i
        ligand_coords.append(coord)
        atom_one_hots.append(one_hot)
        masks.append(mask)
        sizes.append(len(types))

    ligand = {
        'x': torch.cat(ligand_coords, dim=0).to(device),
        'one_hot': torch.cat(atom_one_hots, dim=0).to(device),
        'size': torch.tensor(sizes, dtype=INT_TYPE).to(device),
        'mask': torch.cat(masks, dim=0).to(device),
    }

    return ligand


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


def diversify_ligands(model, pocket, mols, timesteps,
                    sanitize=False,
                    largest_frag=False,
                    relax_iter=0):
    """
    Diversify ligands for a specified pocket.
    
    Parameters:
        model: The model instance used for diversification.
        pocket: The pocket information including coordinates and types.
        mols: List of RDKit molecule objects to be diversified.
        timesteps: Number of denoising steps to apply during diversification.
        sanitize: If True, performs molecule sanitization post-generation (default: False).
        largest_frag: If True, only the largest fragment of the generated molecule is returned (default: False).
        relax_iter: Number of iterations for force field relaxation of the generated molecules (default: 0).
    
    Returns:
        A list of diversified RDKit molecule objects.
    """

    ligand = prepare_ligands_from_mols(mols, model.lig_type_encoder, device=model.device)

    pocket_mask = pocket['mask']
    lig_mask = ligand['mask']

    # Pocket's center of mass
    pocket_com_before = scatter_mean(pocket['x'], pocket['mask'], dim=0)

    out_lig, out_pocket, _, _ = model.ddpm.diversify(ligand, pocket, noising_steps=timesteps)

    # Move generated molecule back to the original pocket position
    pocket_com_after = scatter_mean(out_pocket[:, :model.x_dims], pocket_mask, dim=0)

    out_pocket[:, :model.x_dims] += \
        (pocket_com_before - pocket_com_after)[pocket_mask]
    out_lig[:, :model.x_dims] += \
        (pocket_com_before - pocket_com_after)[lig_mask]

    # Build mol objects
    x = out_lig[:, :model.x_dims].detach().cpu()
    atom_type = out_lig[:, model.x_dims:].argmax(1).detach().cpu()

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
    parser.add_argument('--checkpoint', type=Path, default='checkpoints/crossdocked_fullatom_cond.ckpt')
    parser.add_argument('--pdbfile', type=str, default='example/5ndu.pdb')
    parser.add_argument('--ref_ligand', type=str, default='example/5ndu_linked_mols.sdf')
    parser.add_argument('--objective', type=str, default='sa', choices={'qed', 'sa'})
    parser.add_argument('--timesteps', type=int, default=100)
    parser.add_argument('--population_size', type=int, default=100)
    parser.add_argument('--evolution_steps', type=int, default=10)
    parser.add_argument('--top_k', type=int, default=7)
    parser.add_argument('--outfile', type=Path, default='output.sdf')
    parser.add_argument('--relax', action='store_true')


    args = parser.parse_args()

    pdb_id = Path(args.pdbfile).stem

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    population_size = args.population_size
    evolution_steps = args.evolution_steps
    top_k = args.top_k

    # Load model
    model = LigandPocketDDPM.load_from_checkpoint(
        args.checkpoint, map_location=device)
    model = model.to(device)

    # Prepare ligand + pocket
    # Load PDB
    pdb_model = PDBParser(QUIET=True).get_structure('', args.pdbfile)[0]
    # Define pocket based on reference ligand
    residues = utils.get_pocket_from_ligand(pdb_model, args.ref_ligand)
    pocket = model.prepare_pocket(residues, repeats=population_size)


    if args.objective == 'qed':
        objective_function = MoleculeProperties().calculate_qed
    elif args.objective == 'sa':
        objective_function = MoleculeProperties().calculate_sa
    else:
        ### IMPLEMENT YOUR OWN OBJECTIVE
        ### FUNCTIONS HERE 
        raise ValueError(f"Objective function {args.objective} not recognized.")

    ref_mol = Chem.SDMolSupplier(args.ref_ligand)[0]

    # Store molecules in history dataframe 
    buffer = pd.DataFrame(columns=['generation', 'score', 'fate' 'mol', 'smiles'])

    # Population initialization
    buffer = buffer.append({'generation': 0,
                            'score': objective_function(ref_mol),
                            'fate': 'initial', 'mol': ref_mol,
                            'smiles': Chem.MolToSmiles(ref_mol)}, ignore_index=True)

    for generation_idx in range(evolution_steps):

        if generation_idx == 0:
            molecules = buffer['mol'].tolist() * population_size
        else:
            # Select top k molecules from previous generation
            previous_gen = buffer[buffer['generation'] == generation_idx]
            top_k_molecules = previous_gen.nlargest(top_k, 'score')['mol'].tolist()
            molecules = top_k_molecules * (population_size // top_k)

            # Update the fate of selected top k molecules in the buffer
            buffer.loc[buffer['generation'] == generation_idx, 'fate'] = 'survived'

            # Ensure the right number of molecules
            if len(molecules) < population_size:
                molecules += [random.choice(molecules) for _ in range(population_size - len(molecules))]


        # Diversify molecules
        assert len(molecules) == population_size, f"Wrong number of molecules: {len(molecules)} when it should be {population_size}"
        print(f"Generation {generation_idx}, mean score: {np.mean([objective_function(mol) for mol in molecules])}")
        molecules = diversify_ligands(model,
                                    pocket,
                                    molecules,
                                timesteps=args.timesteps,
                                sanitize=True,
                                relax_iter=(200 if args.relax else 0))
        
        
        # Evaluate and save molecules
        for mol in molecules:
            buffer = buffer.append({'generation': generation_idx + 1,
            'score': objective_function(mol),
            'fate': 'purged',
            'mol': mol,
            'smiles': Chem.MolToSmiles(mol)}, ignore_index=True)


    # Make SDF files
    utils.write_sdf_file(args.outfile, molecules)
    # Save buffer
    buffer.drop(columns=['mol'])
    buffer.to_csv(args.outfile.with_suffix('.csv'))
