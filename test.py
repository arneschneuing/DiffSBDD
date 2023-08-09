import argparse
import warnings
from pathlib import Path
from time import time

import torch
from rdkit import Chem
from tqdm import tqdm

from lightning_modules import LigandPocketDDPM
from analysis.molecule_builder import process_molecule
import utils

MAXITER = 10
MAXNTRIES = 10


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--test_dir', type=Path)
    parser.add_argument('--test_list', type=Path, default=None)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--all_frags', action='store_true')
    parser.add_argument('--sanitize', action='store_true')
    parser.add_argument('--relax', action='store_true')
    parser.add_argument('--batch_size', type=int, default=120)
    parser.add_argument('--resamplings', type=int, default=10)
    parser.add_argument('--jump_length', type=int, default=1)
    parser.add_argument('--timesteps', type=int, default=None)
    parser.add_argument('--fix_n_nodes', action='store_true')
    parser.add_argument('--n_nodes_bias', type=int, default=0)
    parser.add_argument('--n_nodes_min', type=int, default=0)
    parser.add_argument('--skip_existing', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.outdir.mkdir(exist_ok=args.skip_existing)
    raw_sdf_dir = Path(args.outdir, 'raw')
    raw_sdf_dir.mkdir(exist_ok=args.skip_existing)
    processed_sdf_dir = Path(args.outdir, 'processed')
    processed_sdf_dir.mkdir(exist_ok=args.skip_existing)
    times_dir = Path(args.outdir, 'pocket_times')
    times_dir.mkdir(exist_ok=args.skip_existing)

    # Load model
    model = LigandPocketDDPM.load_from_checkpoint(
        args.checkpoint, map_location=device)
    model = model.to(device)

    test_files = list(args.test_dir.glob('[!.]*.sdf'))
    if args.test_list is not None:
        with open(args.test_list, 'r') as f:
            test_list = set(f.read().split(','))
        test_files = [x for x in test_files if x.stem in test_list]

    pbar = tqdm(test_files)
    time_per_pocket = {}
    for sdf_file in pbar:
        ligand_name = sdf_file.stem

        pdb_name, pocket_id, *suffix = ligand_name.split('_')
        pdb_file = Path(sdf_file.parent, f"{pdb_name}.pdb")
        txt_file = Path(sdf_file.parent, f"{ligand_name}.txt")
        sdf_out_file_raw = Path(raw_sdf_dir, f'{ligand_name}_gen.sdf')
        sdf_out_file_processed = Path(processed_sdf_dir,
                                      f'{ligand_name}_gen.sdf')
        time_file = Path(times_dir, f'{ligand_name}.txt')

        if args.skip_existing and time_file.exists() \
                and sdf_out_file_processed.exists() \
                and sdf_out_file_raw.exists():

            with open(time_file, 'r') as f:
                time_per_pocket[str(sdf_file)] = float(f.read().split()[1])

            continue

        for n_try in range(MAXNTRIES):

            try:
                t_pocket_start = time()

                with open(txt_file, 'r') as f:
                    resi_list = f.read().split()

                if args.fix_n_nodes:
                    # some ligands (e.g. 6JWS_bio1_PT1:A:801) could not be read with sanitize=True
                    suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=False)
                    num_nodes_lig = suppl[0].GetNumAtoms()
                else:
                    num_nodes_lig = None

                all_molecules = []
                valid_molecules = []
                processed_molecules = []  # only used as temporary variable
                iter = 0
                n_generated = 0
                n_valid = 0
                while len(valid_molecules) < args.n_samples:
                    iter += 1
                    if iter > MAXITER:
                        raise RuntimeError('Maximum number of iterations has been exceeded.')

                    num_nodes_lig_inflated = None if num_nodes_lig is None else \
                        torch.ones(args.batch_size, dtype=int) * num_nodes_lig

                    # Turn all filters off first
                    mols_batch = model.generate_ligands(
                        pdb_file, args.batch_size, resi_list,
                        num_nodes_lig=num_nodes_lig_inflated,
                        timesteps=args.timesteps, sanitize=False,
                        largest_frag=False, relax_iter=0,
                        n_nodes_bias=args.n_nodes_bias,
                        n_nodes_min=args.n_nodes_min,
                        resamplings=args.resamplings,
                        jump_length=args.jump_length)

                    all_molecules.extend(mols_batch)

                    # Filter to find valid molecules
                    mols_batch_processed = [
                        process_molecule(m, sanitize=args.sanitize,
                                         relax_iter=(200 if args.relax else 0),
                                         largest_frag=not args.all_frags)
                        for m in mols_batch
                    ]
                    processed_molecules.extend(mols_batch_processed)
                    valid_mols_batch = [m for m in mols_batch_processed if m is not None]

                    n_generated += args.batch_size
                    n_valid += len(valid_mols_batch)
                    valid_molecules.extend(valid_mols_batch)

                # Remove excess molecules from list
                valid_molecules = valid_molecules[:args.n_samples]

                # Reorder raw files
                all_molecules = \
                    [all_molecules[i] for i, m in enumerate(processed_molecules)
                     if m is not None] + \
                    [all_molecules[i] for i, m in enumerate(processed_molecules)
                     if m is None]

                # Write SDF files
                utils.write_sdf_file(sdf_out_file_raw, all_molecules)
                utils.write_sdf_file(sdf_out_file_processed, valid_molecules)

                # Time the sampling process
                time_per_pocket[str(sdf_file)] = time() - t_pocket_start
                with open(time_file, 'w') as f:
                    f.write(f"{str(sdf_file)} {time_per_pocket[str(sdf_file)]}")

                pbar.set_description(
                    f'Last processed: {ligand_name}. '
                    f'Validity: {n_valid / n_generated * 100:.2f}%. '
                    f'{(time() - t_pocket_start) / len(valid_molecules):.2f} '
                    f'sec/mol.')

                break  # no more tries needed

            except (RuntimeError, ValueError) as e:
                if n_try >= MAXNTRIES - 1:
                    raise RuntimeError("Maximum number of retries exceeded")
                warnings.warn(f"Attempt {n_try + 1}/{MAXNTRIES} failed with "
                              f"error: '{e}'. Trying again...")

    with open(Path(args.outdir, 'pocket_times.txt'), 'w') as f:
        for k, v in time_per_pocket.items():
            f.write(f"{k} {v}\n")

    times_arr = torch.tensor([x for x in time_per_pocket.values()])
    print(f"Time per pocket: {times_arr.mean():.3f} \pm "
          f"{times_arr.std(unbiased=False):.2f}")
