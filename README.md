# GRACE

Ion mobility mass spectrometry uses collision cross section (CCS) as an
orthogonal descriptor for molecular annotation, but CCS prediction remains
difficult because the measured value reflects the size, shape, and ionization
state of a gas-phase molecular ion. Most machine-learning CCS predictors either
ignore explicit 3D structure or treat adduct identity as a late categorical
feature, which limits their ability to capture adduct-dependent geometric
effects.

GRACE (Geometric Residual Adduct Conditioning via Early-fusion) is a 3D CCS
predictor that adapts a pretrained molecular geometry encoder. It combines two
inductive biases: a residual objective relative to an adduct-aware physical
descriptor baseline, and adduct conditioning within the encoder via a learned
adduct token and low-rank attention adapters.

For further details please see our manuscript on [arXiv](https://arxiv.org/abs/2609.12223).

## Install

Python 3.10 and [uv](https://github.com/astral-sh/uv).

```bash
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

The encoder's pretrained weights, about 180 MB, are downloaded on first use.

## Data

In our [manuscript](https://arxiv.org/abs/2609.12223), the model is trained with over 9,000 experimental molecule/adduct CCS records covering
`[M+H]+`, `[M-H]-` and `[M+Na]+`.  Here, we provide a script `scripts/fetch_sample_data.py` to download a sample of 185 molecule/adduct/CCS datapoints from [PubChem](https://pubchem.ncbi.nlm.nih.gov) [CCSBase annotations](https://pubchem.ncbi.nlm.nih.gov/source/24290) that are by default saved to `data/data.csv`.

```bash
python scripts/fetch_sample_data.py
```

`data.csv` has four columns: `index`, `smiles`, `adducts`, `label`, with the label
in square angstroms. `data/splits/` holds two splits, each a JSON file with
`train`, `val` and `test` lists of row indices.

| Split | How it partitions | What it measures |
|-------|-------------------|------------------|
| `random` | rows at random, 80/10/10 | interpolation |
| `adduct_sensitive` | by molecule; whichever molecules shift most across adducts go to val and test | adduct-driven generalization |

Build the conformer cache before training. The cache is a pickle of tokenised
encoder inputs and is not kept in the repository.

```bash
python -m ccs3d.launch.build_conformer_cache                      # one conformer
python -m ccs3d.launch.build_conformer_cache --num_conformers 10  # ten, with MMFF energies
```

The ten-conformer build may take some time, since every molecule is processed using [ETKDG and MMFF](https://www.rdkit.org/docs/GettingStartedInPython.html).

Regenerate the cache whenever `data.csv` changes. Cache filenames record only the
conformer count and whether hydrogens were removed, and row i of the pickle is
matched to row i of the CSV by position. Changing the number of rows raises an
`IndexError` from the split indices. Reordering rows or editing a SMILES in place
raises nothing, and training then pairs one molecule's geometry with another
molecule's CCS value. An existing cache is never overwritten, so delete it first.

```bash
rm data/cache/unimol_inputs_all_h.pkl
python -m ccs3d.launch.build_conformer_cache
```

## Train

```bash
python -m ccs3d.launch.train_finetune \
    --split random \
    --run_name random/single/residual \
    --residual_target
```

Seeds 0 through 4 run at the defaults: 200 epochs, batch 32, one conformer, LoRA
rank 16, cosine schedule with 10 warmup epochs.

```bash
--seeds 0                 # one seed instead of five
--num_conformers 10 --pooling boltzmann
--residual_target         # predict y - ridge(descriptors)
--no_lora                 # adduct token only, no adapters
--encoder_repr gasteiger  # charge-biased atom pooling; needs the --gasteiger cache
```

`--pooling` chooses how several conformers collapse into one embedding: `single`,
`uniform`, `boltzmann`, or `learned`. It is ignored when `--num_conformers 1`.

Each run writes to `experiments/<run_name>/seed_<s>/`: the best-validation
checkpoint and periodic snapshots under `checkpoints/`, metrics in
`best_val_metrics.json` and `test_at_epochs.csv`, and predictions as `.npy`
arrays. Aggregates across seeds go to `experiments/<run_name>/results_<split>.json`.

The `tfevents` file is written to the seed directory. Pass `--tb_logdir` to
collect logs from several runs under one root instead.

```bash
tensorboard --logdir experiments/
```

## Evaluate on external sets with trained model

`scripts/run_external_eval.py` takes a checkpoint file and a ridge model pickle and predicts CCS values for a given set of molecules.  One set available for direct testing is [GPCL](https://pubs.acs.org/jcisd8/article-abstract/64/3/749/987751/Molecular-Gas-Phase-Conformational-Ensembles?redirectedFrom=fulltext), a 20-compound amino acid and metabolite set used in our manuscript, downloadable [here](https://zenodo.org/records/22902586) as `gpcl_smiles_adducts.csv`.

Input dataset CSVs use the same column-format as `data.csv` described above.

```bash
python scripts/run_external_eval.py \
    --ckpt        experiments/seed_0/checkpoints/best_val.ckpt \
    --data        my_dataset.csv \
    --output-dir  external_evaluation \
    --ridge-model data/ridge_model_random.pkl  # required if model trained with --residual_target
```

Evaluation writes a conformer cache pickle, `<output-dir>/conformer_cache_k<K>.pkl`, keyed by SMILES.
It is reused on later runs and extended automatically when new molecules are encountered — no manual
deletion is needed when switching datasets. To force a full rebuild, delete the file first.

```bash
rm results/external_eval/conformer_cache_k1.pkl
```

Outputs are two CSV files: per-molecule predictions and evaluation metrics. Metrics are only written when ground-truth labels are provided in the input.

## Trained model for evaluation

One of the models described in the companion paper that is trained with the full training dataset using a random split with single conformer pooling and the residual_target option is provided [here](https://github.com/IBM/GRACE/releases/tag/v0.1.0).  As already described above, the model for evaluation requires two files, a checkpoint and a ridge model pickle file.

## Citation

```
@article{suryanarayanan2026predicting,
  title={Predicting Collision Cross Sections with GRACE: Geometric Residual Adduct Conditioning via Early-fusion},
  author={Suryanarayanan, Parthasarathy and Das, Susanta and Sethi, Shreyans and Merz Jr, Kenneth M and Morrone, Joseph A},
  journal={arXiv preprint arXiv:2609.12223},
  year={2026}
}
```

## License

This code is released under the Apache 2.0 license. To read the full text of the license, see [LICENSE](LICENSE)