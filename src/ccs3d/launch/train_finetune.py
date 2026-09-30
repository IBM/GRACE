# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

import sys
import argparse
import csv
import json
import pickle
import warnings
from pathlib import Path

import ccs3d.utils.rdkit_patch

import numpy as np
import pandas as pd
import torch
import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from unimol_tools.models.unimol import UniMolModel

from ccs3d import CCS3D, CCS3DTrainer, CCSDataModule
from ccs3d.utils.metrics import compute_metrics, seed_row, aggregate_seeds

ROOT       = Path(__file__).resolve().parents[3]
DATA_CSV   = ROOT / "data" / "data.csv"
SPLITS_DIR = ROOT / "data" / "splits"
EXP_ROOT   = ROOT / "experiments"
CACHE_DIR  = ROOT / "data" / "cache"

SPLIT_PATHS = {
    "random":           SPLITS_DIR / "random"           / "split.json",
    "scaffold":         SPLITS_DIR / "scaffold"         / "split.json",
    "adduct_sensitive": SPLITS_DIR / "adduct_sensitive" / "split.json",
}

SEEDS              = [0, 1, 2, 3, 4]
CHECKPOINT_EPOCHS  = [10, 50, 100, 150, 200]


def _compute_ridge_preds(split: str, split_json_path: Path) -> np.ndarray:
    out_path = ROOT / "data" / f"ridge_preds_{split}.npy"
    if out_path.exists():
        print(f"  Loading cached Ridge preds from {out_path.name}")
        return np.load(str(out_path))

    warnings.filterwarnings("ignore")
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, rdPartialCharges
    from rdkit.Chem.rdMolDescriptors import CalcLabuteASA, CalcTPSA
    from rdkit.Chem.Crippen import MolLogP, MolMR

    ADDUCT_ORDER = ["[M+H]+", "[M-H]-", "[M+Na]+"]

    def mol_features(smiles):
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

    df = pd.read_csv(DATA_CSV)
    splits = json.load(open(split_json_path))
    train_idx = splits["train"]

    print("  Computing descriptors for Ridge baseline...", flush=True)
    desc_all = np.array([mol_features(s) for s in df["smiles"]], dtype=np.float32)
    ohe_all  = np.stack(
        [(df["adducts"] == a).astype(np.float32).values for a in ADDUCT_ORDER], axis=1
    )
    X_all = np.concatenate([desc_all, ohe_all], axis=1)

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_all[train_idx])
    y_tr   = df["label"].values[train_idx].astype(np.float32)

    ridge  = RidgeCV(alphas=np.logspace(-1, 5, 30), cv=5)
    ridge.fit(X_tr, y_tr)

    X_all_scaled = scaler.transform(X_all)
    preds = ridge.predict(X_all_scaled).astype(np.float32)

    np.save(str(out_path), preds)

    model_path = ROOT / "data" / f"ridge_model_{split}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump((scaler, ridge), f)

    train_rmse = float(np.sqrt(np.mean((y_tr - preds[train_idx])**2)))
    print(f"  Ridge fitted (alpha={ridge.alpha_:.2e}).  Train RMSE={train_rmse:.2f}")
    print(f"  Saved -> {out_path.name}, {model_path.name}")
    return preds


class SaveAtEpochs(Callback):
    def __init__(self, epochs, dirpath):
        self.epochs  = set(epochs)
        self.dirpath = Path(dirpath)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_1indexed = trainer.current_epoch + 1
        if epoch_1indexed in self.epochs:
            self.dirpath.mkdir(parents=True, exist_ok=True)
            path = self.dirpath / f"epoch_{epoch_1indexed:04d}.ckpt"
            trainer.save_checkpoint(str(path))


def predict_loader(lit, loader, device):
    lit.eval()
    lit.to(device)
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            net_inputs, adduct_onehot, labels, bw, mask, charges, ridge_preds, _mol_ids = \
                lit._unpack_batch(batch)
            net_inputs    = {k: v.to(device) for k, v in net_inputs.items()}
            adduct_onehot = adduct_onehot.to(device)
            bw      = bw.to(device)      if bw      is not None else None
            mask    = mask.to(device)    if mask    is not None else None
            charges = charges.to(device) if charges is not None else None
            raw_preds = lit(net_inputs, adduct_onehot, bw, mask, charges)
            if ridge_preds is not None:
                raw_preds = raw_preds + ridge_preds.to(device)
            all_preds.append(raw_preds.cpu().numpy())
            all_targets.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def evaluate_checkpoint(ckpt_path, args, dm, device):
    encoder = UniMolModel(output_dim=1, data_type="molecule", remove_hs=args.remove_hs)
    head    = CCS3D(encoder_dim=512,
                    hidden_dim=args.hidden_dim, dropout=args.dropout,
                    pooling=args.pooling, gasteiger_atom=args.gasteiger_atom)
    lit = CCS3DTrainer.load_from_checkpoint(
        str(ckpt_path), encoder=encoder, head=head,
        num_conformers=args.num_conformers, gasteiger_atom=args.gasteiger_atom,
        lora=args.lora,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        scheduler=args.scheduler, max_epochs=args.max_epochs,
        lora_warmup_epochs=args.lora_warmup_epochs,
        residual_target=args.residual_target,
        multi_adduct_loss=args.multi_adduct_loss,
        mal_weight=args.mal_weight,
    )

    train_pred, train_true = predict_loader(lit, dm.train_eval_dataloader(), device)
    test_pred,  test_true  = predict_loader(lit, dm.test_dataloader(),       device)

    return {
        "train": compute_metrics(train_true, train_pred),
        "test":  compute_metrics(test_true,  test_pred),
        "train_pred": train_pred, "train_true": train_true,
        "test_pred":  test_pred,  "test_true":  test_true,
    }


def _prepare_cache(args, split_json, cache_dir):
    encoder = UniMolModel(output_dim=1, data_type="molecule", remove_hs=args.remove_hs)
    dm = CCSDataModule(
        encoder=encoder, data_csv=DATA_CSV, split_json=split_json,
        cache_dir=cache_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, remove_hs=args.remove_hs,
        num_conformers=args.num_conformers, gasteiger_atom=args.gasteiger_atom,
    )
    dm.prepare_data()


def _train_seed(seed, args, exp_dir, split_json, cache_dir, device, run_name):
    seed_dir = exp_dir / f"seed_{seed}"
    ckpt_dir = seed_dir / "checkpoints"
    seed_dir.mkdir(parents=True, exist_ok=True)

    encoder = UniMolModel(output_dim=1, data_type="molecule", remove_hs=args.remove_hs)
    head    = CCS3D(encoder_dim=512,
                    hidden_dim=args.hidden_dim, dropout=args.dropout,
                    pooling=args.pooling, gasteiger_atom=args.gasteiger_atom)
    lit = CCS3DTrainer(
        encoder=encoder, head=head,
        lr_head=args.lr_head, lr_encoder=args.lr_encoder,
        freeze_epochs=args.freeze_epochs, weight_decay=args.weight_decay,
        num_conformers=args.num_conformers, gasteiger_atom=args.gasteiger_atom,
        lora=args.lora,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        scheduler=args.scheduler, max_epochs=args.max_epochs,
        lora_warmup_epochs=args.lora_warmup_epochs,
        residual_target=args.residual_target,
        multi_adduct_loss=args.multi_adduct_loss,
        mal_weight=args.mal_weight,
    )

    dm = CCSDataModule(
        encoder=encoder, data_csv=DATA_CSV, split_json=split_json,
        cache_dir=cache_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, remove_hs=args.remove_hs,
        num_conformers=args.num_conformers, gasteiger_atom=args.gasteiger_atom,
        ridge_preds=args.ridge_preds_array,
        mol_ids=args.mol_ids_array,
    )
    dm.setup()

    best_val_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir), filename="best_val",
        monitor="val/rmse", mode="min", save_top_k=1,
    )
    epoch_cb = SaveAtEpochs(
        epochs=[e for e in CHECKPOINT_EPOCHS if e <= args.max_epochs],
        dirpath=ckpt_dir,
    )
    early_cb = EarlyStopping(monitor="val/rmse", patience=args.patience, mode="min")
    if args.tb_logdir:
        logger = TensorBoardLogger(save_dir=args.tb_logdir, name=run_name, version=f"seed_{seed}")
    else:
        logger = TensorBoardLogger(save_dir=str(exp_dir), name=f"seed_{seed}", version="")

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        val_check_interval=args.val_check_interval,
        accumulate_grad_batches=args.accumulate_grad_batches,
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1,
        precision="16-mixed" if device.type == "cuda" else "32",
        callbacks=[best_val_cb, epoch_cb, early_cb],
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=1,
        deterministic=False,
        gradient_clip_val=args.gradient_clip_val if args.gradient_clip_val > 0 else None,
    )
    trainer.fit(lit, dm)

    return best_val_cb.best_model_path, ckpt_dir, dm


def run_seed(seed, args, exp_dir, split_json, cache_dir, device, run_name):
    L.seed_everything(seed, workers=True)

    seed_dir = exp_dir / f"seed_{seed}"

    best_model_path, ckpt_dir, dm = _train_seed(
        seed, args, exp_dir, split_json, cache_dir, device, run_name
    )

    epoch_rows = []
    for epoch in CHECKPOINT_EPOCHS:
        ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.ckpt"
        if not ckpt_path.exists():
            break
        results = evaluate_checkpoint(ckpt_path, args, dm, device)
        row = {"epoch": epoch}
        for split in ("train", "test"):
            for metric, val in results[split].items():
                row[f"{split}_{metric}"] = round(val, 6)
        epoch_rows.append(row)
        print(f"    epoch {epoch:3d}  train_RMSE={results['train']['RMSE']:.4f}  "
              f"test_RMSE={results['test']['RMSE']:.4f}  "
              f"test_R={results['test']['PearsonR']:.4f}  "
              f"test_rho={results['test']['SpearmanR']:.4f}  "
              f"test_tau={results['test']['KendallTau']:.4f}")

    if epoch_rows:
        csv_path = seed_dir / "test_at_epochs.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=epoch_rows[0].keys())
            writer.writeheader()
            writer.writerows(epoch_rows)

    best_results = evaluate_checkpoint(best_model_path, args, dm, device)

    np.save(seed_dir / "train_preds_best_val.npy",   best_results["train_pred"])
    np.save(seed_dir / "train_targets_best_val.npy", best_results["train_true"])
    np.save(seed_dir / "test_preds_best_val.npy",    best_results["test_pred"])
    np.save(seed_dir / "test_targets_best_val.npy",  best_results["test_true"])

    bv_train = best_results["train"]
    bv_test  = best_results["test"]
    print(f"  seed={seed}  best_val -> "
          f"train_RMSE={bv_train['RMSE']:.4f}  "
          f"test_RMSE={bv_test['RMSE']:.4f}  "
          f"test_%diff={bv_test['MeanPctDiff']:.4f}  "
          f"test_R={bv_test['PearsonR']:.4f}  "
          f"test_rho={bv_test['SpearmanR']:.4f}  "
          f"test_tau={bv_test['KendallTau']:.4f}")

    with open(seed_dir / "best_val_metrics.json", "w") as f:
        json.dump({"train": bv_train, "test": bv_test}, f, indent=2)

    return seed_row(
        y_train_true=best_results["train_true"], y_train_pred=best_results["train_pred"],
        y_test_true=best_results["test_true"],   y_test_pred=best_results["test_pred"],
        split=args.split, seed=seed,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split",         default="random", help="Split name: random, scaffold, random_0.1, ...")
    p.add_argument("--seeds",         type=int, nargs="+", default=None, help="Seeds to run (default: all 5)")
    p.add_argument("--run_name",      default=None,     help="Experiment dir (default: finetune_<split>)")
    p.add_argument("--max_epochs",    type=int,   default=200)
    p.add_argument("--freeze_epochs", type=int,   default=0)
    p.add_argument("--patience",      type=int,   default=20)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--lr_head",       type=float, default=1e-4)
    p.add_argument("--lr_encoder",    type=float, default=1e-5)
    p.add_argument("--weight_decay",  type=float, default=1e-2)
    p.add_argument("--hidden_dim",      type=int,   default=256)
    p.add_argument("--dropout",         type=float, default=0.1)
    p.add_argument("--remove_hs",       action="store_true")
    p.add_argument("--num_conformers",  type=int,   default=1,
                   help="1 = single conformer (A0); 10 = multi-conformer (A1-A3)")
    p.add_argument("--pooling",         default="single",
                   choices=["single", "uniform", "boltzmann", "learned"],
                   help="Conformer pooling mode (ignored when num_conformers=1)")
    p.add_argument("--encoder_repr",    default="cls",
                   choices=["cls", "gasteiger"],
                   help="Encoder representation: 'cls' = CLS token (default); "
                        "'gasteiger' = Gasteiger-biased atom attention pooled alongside CLS. "
                        "Orthogonal to --pooling. Requires gasteiger_charges.pkl - "
                        "build with: build_conformer_cache.py --gasteiger")
    p.add_argument("--max_steps",        type=int,   default=-1,
                   help="Max training steps total (-1 = unlimited, determined by max_epochs)")
    p.add_argument("--val_check_interval", type=int, default=None,
                   help="Run validation every N training steps (default: once per epoch)")
    p.add_argument("--tb_logdir",       default=None,
                   help="Shared TensorBoard root (all runs log here as separate names). "
                        "Default: per-run experiments/<run_name>/")
    p.add_argument("--no_lora", action="store_false", dest="lora", default=True,
                   help="Disable LoRA(Q,V) adapters. Adduct delta is always applied; "
                        "this flag trains only the delta (~1.5K params) without LoRA.")
    p.add_argument("--lora_rank",  type=int,   default=16,
                   help="LoRA rank r (default 16; use 8 for fewer params)")
    p.add_argument("--lora_alpha", type=float, default=32.0,
                   help="LoRA alpha scaling factor (default 32.0; scale = alpha/rank)")
    p.add_argument("--scheduler", default="cosine", choices=["cosine", "plateau"],
                   help="LR scheduler: 'cosine' (CosineAnnealingLR) or 'plateau' (ReduceLROnPlateau)")
    p.add_argument("--lora_warmup_epochs", type=int, default=10,
                   help="Epochs of linear LR warmup for LoRA params (default 10)")
    p.add_argument("--gradient_clip_val", type=float, default=1.0,
                   help="Gradient clipping max norm (default 1.0; set 0 to disable)")
    p.add_argument("--accumulate_grad_batches", type=int, default=1,
                   help="Gradient accumulation steps. Effective batch = batch_size x this.")
    p.add_argument("--residual_target", action="store_true", default=False,
                   help="Train on residual y - Ridge(descriptors). Fits RidgeCV once on "
                        "train split; predictions cached to data/ridge_preds_<split>.npy. "
                        "Head final layer is zero-initialised so prediction = Ridge at t=0.")
    p.add_argument("--multi_adduct_loss", action="store_true", default=False,
                   help="Add multi-adduct pairwise loss (MAL): penalises wrong pairwise CCS "
                        "differences for same-molecule pairs within a batch. Requires SMILES "
                        "duplicates across adducts in the training set.")
    p.add_argument("--mal_weight", type=float, default=0.1,
                   help="Weight lambda for multi-adduct loss (default 0.1). "
                        "total_loss = MSE + lambda * MAL")
    return p.parse_args()


def main():
    args     = parse_args()
    args.gasteiger_atom = (args.encoder_repr == "gasteiger")
    run_name = args.run_name or f"finetune_{args.split}"
    exp_dir   = EXP_ROOT / run_name
    cache_dir = CACHE_DIR
    if args.split not in SPLIT_PATHS:
        raise ValueError(f"Unknown split '{args.split}'. Choose from: {list(SPLIT_PATHS)}")
    split_json = SPLIT_PATHS[args.split]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir.mkdir(parents=True, exist_ok=True)
    seeds = args.seeds if args.seeds is not None else SEEDS
    print(f"Run: {run_name}  |  split: {args.split}  |  seeds: {seeds}")
    print(f"checkpoint epochs: {CHECKPOINT_EPOCHS}")
    print(f"max_epochs={args.max_epochs}  freeze_epochs={args.freeze_epochs}  "
          f"lr_head={args.lr_head}  lr_encoder={args.lr_encoder}")
    if args.residual_target:
        print("residual_target=True: training on y - Ridge(descriptors)")
    if args.multi_adduct_loss:
        print(f"multi_adduct_loss=True: MAL weight={args.mal_weight}")

    args.ridge_preds_array = None
    if args.residual_target:
        args.ridge_preds_array = _compute_ridge_preds(args.split, split_json)

    args.mol_ids_array = None
    if args.multi_adduct_loss:
        df_full = pd.read_csv(DATA_CSV)
        seen: dict = {}
        mol_ids_list = []
        for s in df_full["smiles"]:
            if s not in seen:
                seen[s] = len(seen)
            mol_ids_list.append(seen[s])
        args.mol_ids_array = np.array(mol_ids_list, dtype=np.int64)
        n_unique = len(seen)
        n_multi  = sum(1 for cnt in pd.Series(mol_ids_list).value_counts() if cnt > 1)
        print(f"  mol_ids: {n_unique} unique SMILES, {n_multi} appear with >1 adduct")

    _prepare_cache(args, split_json, cache_dir)

    per_seed_rows = []
    for seed in seeds:
        print(f"\n=== seed {seed} ===")
        row = run_seed(seed, args, exp_dir, split_json, cache_dir, device, run_name)
        per_seed_rows.append(row)

    agg = aggregate_seeds(per_seed_rows, split=args.split)

    print(f"\n=== Aggregated over {len(SEEDS)} seeds (best-val checkpoint) ===")
    print(f"  train: RMSE={agg['train_RMSE_mean']:.4f} +/- {agg['train_RMSE_std']:.4f}  "
          f"%diff={agg['train_MeanPctDiff_mean']:.4f} +/- {agg['train_MeanPctDiff_std']:.4f}")
    print(f"  test:  RMSE={agg['test_RMSE_mean']:.4f} +/- {agg['test_RMSE_std']:.4f}  "
          f"%diff={agg['test_MeanPctDiff_mean']:.4f} +/- {agg['test_MeanPctDiff_std']:.4f}  "
          f"R={agg['test_PearsonR_mean']:.4f} +/- {agg['test_PearsonR_std']:.4f}  "
          f"rho={agg['test_SpearmanR_mean']:.4f} +/- {agg['test_SpearmanR_std']:.4f}  "
          f"tau={agg['test_KendallTau_mean']:.4f} +/- {agg['test_KendallTau_std']:.4f}")

    args_dict = {k: v.tolist() if hasattr(v, "tolist") else v
                 for k, v in vars(args).items()}
    out = {
        "run_name": run_name, "split": args.split, "args": args_dict,
        "aggregate": agg, "per_seed": per_seed_rows,
    }
    out_json = exp_dir / f"results_{args.split}.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved -> {out_json}")


if __name__ == "__main__":
    main()
