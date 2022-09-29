import sys
import torch
import shutil
from pathlib import Path

from rdkit import Chem
from tqdm import tqdm


basedir = sys.argv[1]
structure_dir = Path(basedir, 'crossdocked_pocket10')

test_set = torch.load(Path(basedir, 'split_by_name.pt'))['test']

receptor_dir = Path(basedir, 'receptor_pdbs')
receptor_dir.mkdir(exist_ok=True)

ref_ligand_dir = Path(basedir, 'reference_ligands')
ref_ligand_dir.mkdir(exist_ok=True)

methods = ['cvae', 'sbdd', 'p2m']
for method in methods:
    method_lig_dir = Path(basedir, f'{method}_processed')
    method_lig_dir.mkdir(exist_ok=True)

for pocket_idx, (receptor_name, ligand_name) in enumerate(tqdm(test_set)):

    # copy receptor file and remove underscores
    new_rec_name = Path(receptor_name).stem.replace('_', '-')
    shutil.copy(Path(structure_dir, receptor_name), Path(receptor_dir, new_rec_name + '.pdb'))

    # copy and rename reference ligands
    new_lig_name = new_rec_name + '_' + Path(ligand_name).stem.replace('_', '-')
    shutil.copy(Path(structure_dir, ligand_name), Path(ref_ligand_dir, new_lig_name + '.sdf'))

    for method in methods:

        method_pocket_dir = Path(basedir, method, f'pocket_{pocket_idx}')

        generated_mols = [Chem.SDMolSupplier(str(file), sanitize=False)[0]
                          for file in method_pocket_dir.glob(f'mol_*.sdf')]

        # only select first 100 molecules
        generated_mols = generated_mols[:100]
        if len(generated_mols) < 1:
            print('No molecule found for this pocket')
            continue
        if len(generated_mols) < 100:
            print('Less than 100 molecules found for this pocket')

        # write a combined sdf file
        sdf_path = Path(basedir, f'{method}_processed', f'{new_rec_name}_mols-pocket-{pocket_idx}.sdf')
        with Chem.SDWriter(str(sdf_path)) as w:
            for mol in generated_mols:
                w.write(mol)
