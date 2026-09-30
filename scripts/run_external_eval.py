# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""
Evaluate a CCS3D checkpoint against a CSV dataset.

Usage
-----
    # Non-residual checkpoint:
    python scripts/run_external_eval.py \
        --ckpt  experiments/seed_0/checkpoints/best_val.ckpt \
        --data  my_dataset.csv

    # Residual checkpoint (trained with --residual_target):
    python scripts/run_external_eval.py \
        --ckpt        experiments/seed_0/checkpoints/best_val.ckpt \
        --data        my_dataset.csv \
        --ridge-model data/ridge_model_random.pkl

Input CSV columns: smiles, adducts, label

Outputs (written to --output-dir):
    <dataset>_predictions.csv   per-molecule predictions
    <dataset>_metrics.csv       RMSE, MeanPctDiff, PearsonR, SpearmanR, KendallTau
"""

import argparse
import pickle
from pathlib import Path

import ccs3d.utils.rdkit_patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from unimol_tools.data import DataHub
from unimol_tools.models.unimol import UniMolModel

from ccs3d import CCS3D, CCS3DTrainer
from ccs3d.data.ccs_data import (
    ADDUCT_ORDER,
    CCSDataset,
    MultiConformerCCSDataset,
    ResidualCCSDataset,
    make_collate_fn,
    make_multi_collate_fn,
    make_residual_collate_fn,
)
from ccs3d.utils.metrics import compute_metrics


def build_external_cache(smiles_list: list, K: int, remove_hs: bool,
                          cache_path: Path) -> dict:
    cache = {}
    if cache_path.exists():
        print(f"  Loading external conformer cache: {cache_path}")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)

    unique_smiles = list(dict.fromkeys(smiles_list))
    missing = [s for s in unique_smiles if s not in cache]

    if not missing:
        return cache

    print(f"  Generating conformers for {len(missing)} new molecules (K={K}) ...")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if K == 1:
        datahub = DataHub(
            data=missing,
            task="repr",
            is_train=False,
            model_name="unimolv1",
            data_type="molecule",
            remove_hs=remove_hs,
        )
        inputs = datahub.data["unimol_input"]
        cache.update({smi: inp for smi, inp in zip(missing, inputs)})
    else:
        from unimol_tools.data.datahub import ConformerGen
        from ccs3d.launch.build_conformer_cache import _mol_to_entry
        cgen       = ConformerGen(data_type="molecule", remove_hs=remove_hs)
        dictionary = cgen.dictionary
        for smi in missing:
            cache[smi] = _mol_to_entry(smi, K=K, seed=42, remove_hs=remove_hs,
                                       dictionary=dictionary)

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"  Saved -> {cache_path}")
    return cache


def _mol_features(smiles: str) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.rdMolDescriptors import CalcLabuteASA, CalcTPSA
    from rdkit.Chem.Crippen import MolLogP, MolMR
    from rdkit.Chem import rdPartialCharges

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(9, dtype=np.float32)
    rdPartialCharges.ComputeGasteigerCharges(mol)
    charges = [float(a.GetPropsAsDict().get("_GasteigerCharge", 0.0))
               for a in mol.GetAtoms()]
    charges = [c for c in charges if np.isfinite(c)]
    m = Descriptors.ExactMolWt(mol)
    return np.array([
        m ** (2.0 / 3.0),
        CalcLabuteASA(mol), CalcTPSA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        MolMR(mol), MolLogP(mol),
        (max(charges) - min(charges)) if charges else 0.0,
        sum(abs(c) for c in charges),
    ], dtype=np.float32)


def predict_ridge_external(smiles_list: list, adducts_list: list,
                            scaler, ridge) -> np.ndarray:
    desc = np.array([_mol_features(s) for s in smiles_list], dtype=np.float32)
    ohe  = np.stack(
        [(np.array(adducts_list) == a).astype(np.float32) for a in ADDUCT_ORDER], axis=1
    )
    X = np.concatenate([desc, ohe], axis=1)
    return ridge.predict(scaler.transform(X)).astype(np.float32)


def build_loader(entries_or_inputs: list, adducts: list, labels: np.ndarray,
                 ridge_preds: np.ndarray | None, K: int, padding_idx: int,
                 batch_size: int, num_workers: int) -> DataLoader:
    adduct_ohe = np.stack(
        [(np.array(adducts) == a).astype(np.float32) for a in ADDUCT_ORDER], axis=1
    )
    labels_arr = np.array(labels, dtype=np.float32)

    if K == 1:
        base_ds   = CCSDataset(entries_or_inputs, adduct_ohe, labels_arr)
        base_coll = make_collate_fn(padding_idx)
    else:
        base_ds   = MultiConformerCCSDataset(entries_or_inputs, adduct_ohe, labels_arr, K=K)
        base_coll = make_multi_collate_fn(padding_idx, K)

    if ridge_preds is not None:
        dataset    = ResidualCCSDataset(base_ds, ridge_preds)
        collate_fn = make_residual_collate_fn(base_coll)
    else:
        dataset    = base_ds
        collate_fn = base_coll

    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, collate_fn=collate_fn)


def _to_device(net_inputs: dict, device) -> dict:
    return {k: v.to(device) for k, v in net_inputs.items()}


def run_inference(lit: CCS3DTrainer, loader: DataLoader, device) -> tuple:
    lit.eval()
    lit.to(device)
    K         = lit.hparams.num_conformers
    residual  = lit.hparams.residual_target
    gasteiger = lit.hparams.gasteiger_atom

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            if residual:
                *batch, ridge_preds = batch
                ridge_preds = ridge_preds.to(device)
            else:
                ridge_preds = None

            if gasteiger and K > 1:
                net_inputs, bw, mask, adduct, labels, charges = batch
                raw = lit(_to_device(net_inputs, device), adduct.to(device),
                          bw.to(device), mask.to(device), charges.to(device))
            elif gasteiger:
                net_inputs, adduct, labels, charges, _mol_ids = batch
                raw = lit(_to_device(net_inputs, device), adduct.to(device),
                          None, None, charges.to(device))
            elif K > 1:
                net_inputs, bw, mask, adduct, labels = batch
                raw = lit(_to_device(net_inputs, device), adduct.to(device),
                          bw.to(device), mask.to(device), None)
            else:
                net_inputs, adduct, labels, _mol_ids = batch
                raw = lit(_to_device(net_inputs, device), adduct.to(device))

            if ridge_preds is not None:
                raw = raw + ridge_preds
            all_preds.append(raw.cpu().numpy())
            all_labels.append(np.asarray(labels))

    return np.concatenate(all_preds), np.concatenate(all_labels)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ckpt",        required=True, type=Path,
                   help="Path to the Lightning checkpoint file (*.ckpt)")
    p.add_argument("--data",        required=True, type=Path,
                   help="CSV with columns: smiles, adducts, label")
    p.add_argument("--ridge-model", default=None, type=Path,
                   help="Path to ridge_model_<split>.pkl. Required when the "
                        "checkpoint was trained with --residual_target.")
    p.add_argument("--output-dir",  required=True, type=Path,
                   help="Root directory for all outputs. Results written to "
                        "<output-dir>/. Conformer cache shared at "
                        "<output-dir>/conformer_cache_k<K>.pkl.")
    p.add_argument("--cache",       default=None, type=Path,
                   help="Path to a conformer-cache pkl. "
                        "Created here if the file does not exist.")
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    ckpt_path = args.ckpt.resolve()
    data_path = args.data.resolve()
    out_dir   = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt_data = torch.load(str(ckpt_path), map_location="cpu")
    hp        = ckpt_data.get("hyper_parameters", {})
    hp.setdefault("remove_hs",          False)
    hp.setdefault("hidden_dim",         256)   # not saved in ckpt; mismatch causes state_dict error
    hp.setdefault("dropout",            0.1)
    hp.setdefault("gasteiger_atom",     False)
    hp.setdefault("lora",               True)
    hp.setdefault("lora_rank",          16)
    hp.setdefault("lora_alpha",         32.0)
    hp.setdefault("scheduler",          "cosine")
    hp.setdefault("max_epochs",         200)
    hp.setdefault("lora_warmup_epochs", 10)
    hp.setdefault("residual_target",    False)
    hp.setdefault("multi_adduct_loss",  False)
    hp.setdefault("mal_weight",         0.1)
    hp.setdefault("num_conformers",     1)
    hp.setdefault("pooling",            "single")

    K        = hp["num_conformers"]
    residual = hp["residual_target"]

    scaler, ridge = (None, None)
    if residual:
        if args.ridge_model is None:
            p.error("Checkpoint was trained with --residual_target but "
                    "--ridge-model was not supplied.")
        print(f"Loading Ridge model: {args.ridge_model}")
        with open(args.ridge_model, "rb") as f:
            scaler, ridge = pickle.load(f)

    print(f"Loading dataset: {data_path}")
    df           = pd.read_csv(data_path)
    smiles       = df["smiles"].tolist()
    adducts      = df["adducts"].tolist()
    labels       = df["label"].values.astype(np.float32)
    dataset_name = data_path.stem

    cache_path = (args.cache or out_dir / f"conformer_cache_k{K}.pkl").resolve()
    print(f"\nBuilding/loading conformer cache (K={K}) ...")
    ext_cache = build_external_cache(smiles, K, hp["remove_hs"], cache_path)

    ridge_preds = None
    if residual:
        print(f"  Computing Ridge preds for {dataset_name} ...")
        ridge_preds = predict_ridge_external(smiles, adducts, scaler, ridge)

    print("\nInstantiating encoder for padding_idx ...")
    encoder_proto = UniMolModel(output_dim=1, data_type="molecule",
                                remove_hs=hp["remove_hs"])
    padding_idx   = encoder_proto.padding_idx
    del encoder_proto

    loader = build_loader(
        [ext_cache[s] for s in smiles], adducts, labels, ridge_preds,
        K, padding_idx, args.batch_size, args.num_workers,
    )

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = UniMolModel(output_dim=1, data_type="molecule",
                          remove_hs=hp["remove_hs"])
    head    = CCS3D(
        encoder_dim=512,
        hidden_dim=hp["hidden_dim"],
        dropout=hp["dropout"],
        pooling=hp["pooling"],
        gasteiger_atom=hp["gasteiger_atom"],
    )
    lit = CCS3DTrainer.load_from_checkpoint(
        str(ckpt_path), encoder=encoder, head=head,
        num_conformers=hp["num_conformers"],
        gasteiger_atom=hp["gasteiger_atom"],
        lora=hp["lora"],
        lora_rank=hp["lora_rank"],
        lora_alpha=hp["lora_alpha"],
        scheduler=hp["scheduler"],
        max_epochs=hp["max_epochs"],
        lora_warmup_epochs=hp["lora_warmup_epochs"],
        residual_target=hp["residual_target"],
        multi_adduct_loss=hp["multi_adduct_loss"],
        mal_weight=hp["mal_weight"],
    )

    print(f"\nRunning inference on {len(df)} molecules ...")
    preds, true_labels = run_inference(lit, loader, device)

    out_csv = out_dir / f"{dataset_name}_predictions.csv"
    pd.DataFrame({
        "SMILES":        df["smiles"],
        "Adduct":        df["adducts"],
        "True CCS":      true_labels,
        "Predicted CCS": preds,
    }).to_csv(out_csv, index=False)
    print(f"\nPredictions -> {out_csv}")

    if (true_labels <= 0).all():
        print("\nSkipping metrics: all labels are <= 0 (dummy/unknown labels).")
    else:
        m = compute_metrics(true_labels, preds)
        print(f"\nResults on {dataset_name}:")
        print(f"  RMSE={m['RMSE']:.4f}  MPD%={m['MeanPctDiff']:.4f}  "
              f"R={m['PearsonR']:.4f}  rho={m['SpearmanR']:.4f}  "
              f"tau={m['KendallTau']:.4f}")
        metrics_csv = out_dir / f"{dataset_name}_metrics.csv"
        pd.DataFrame([m]).to_csv(metrics_csv, index=False)
        print(f"Metrics -> {metrics_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
