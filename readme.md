# ZnSiteFinder


## Usage

Install the dependencies with `pip install -r requirements.txt`, then place your structure as `1abc.cif` in the same folder as `predict.py`. Run `python predict.py --pdb 1abc` to print the predicted Zn sites with their confidence residues. Use `--threshold` to change the residue probability cutoff (default 0.5).
