# ZnSiteFinder


## Usage

Install the dependencies with `pip install -r requirements.txt`, then place your structure as `1abc.cif` in the same folder as `predict.py`. Run `python predict.py --pdb 1abc` to print the predicted Zn sites with their confidence residues. Use `--threshold` to change the residue probability cutoff (default 0.5).

Input must be an mmCIF file (`.cif`). The output lists each predicted site with the coordinating residues and a per-residue probability. Only standard amino-acid residues are scored; ligands and waters are ignored. A GPU is used automatically if available, otherwise it runs on CPU.
