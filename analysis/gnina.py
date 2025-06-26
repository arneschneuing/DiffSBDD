import os
from rdkit import Chem

def parse_gnina_log(filename, multiple=False):
    if not multiple:
        d={'Affinity':[],'RMSD':[],'CNNscore':[],'CNNaffinity':[],'CNNvariance':[]}
        with open(filename,'r') as f:
            for line in f:
                for key in d:
                    if line[:len(key)]==key:
                        d[key].append(float(line.strip().split(' (kcal/mol)')[0].split(' ')[-1]))
        return d
    else:
        d={'affinity':[],'intramol':[],'CNNscore':[],'CNNaffinity':[]}
        with open(filename,'r') as f:
            for line in f:
                if line[:5]=='    1':
                    for i, key in enumerate(d):
                        d[key].append(float(line[5+i*13:18+i*13]))
        return d
    
class Gnina:
    def __init__(self, pdb_file):
        self.gnina='/home/domain/data/prog/micromamba/envs/drugflow/bin/gnina'
        self.tmp_ligand='/tmp/tmp_gnina.sdf'
        self.pdb_file=pdb_file

    def calculate_metrics(self,rdmol):

        writer = Chem.SDWriter(self.tmp_ligand)
        writer.write(mol=rdmol)
        writer.close()

        cmd=f'{self.gnina} -r {self.pdb_file} -l {self.tmp_ligand}  --minimize'
        output = os.popen(cmd, 'r')
        d={'Affinity':[],'RMSD':[],'CNNscore':[],'CNNaffinity':[],'CNNvariance':[]}
        for line in output:
            for key in d:
                if line.startswith(key):
                    d[key].append(float(line.strip().split(' ')[1]))
        return d
    
    def affinity(self,rdmol):
        d=self.calculate_metrics(rdmol)
        return -d['Affinity'][0]

    def CNNaffinity(self,rdmol):
        d=self.calculate_metrics(rdmol)
        return d['CNNaffinity'][0]
