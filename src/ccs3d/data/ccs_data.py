# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

import json
import pickle
from pathlib import Path

import ccs3d.utils.rdkit_patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L

from unimol_tools.data import DataHub
from unimol_tools.utils.util import pad_1d_tokens, pad_2d, pad_coords

ADDUCT_ORDER = ["[M+H]+", "[M-H]-", "[M+Na]+"]


class CCSDataset(Dataset):
    def __init__(self, unimol_inputs, adduct_onehots, labels, mol_ids=None):
        assert len(unimol_inputs) == len(adduct_onehots) == len(labels)
        self.unimol_inputs = unimol_inputs
        self.adduct_onehots = torch.tensor(adduct_onehots, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        if mol_ids is not None:
            self.mol_ids = torch.tensor(mol_ids, dtype=torch.long)
        else:
            self.mol_ids = torch.full((len(labels),), -1, dtype=torch.long)

    def __len__(self):
        return len(self.unimol_inputs)

    def __getitem__(self, idx):
        return (self.unimol_inputs[idx], self.adduct_onehots[idx],
                self.labels[idx], self.mol_ids[idx])

class CollateFn:
    def __init__(self, padding_idx: int):
        self.padding_idx = padding_idx

    def __call__(self, samples):
        net_inputs = {
            "src_tokens":   pad_1d_tokens([torch.tensor(s[0]["src_tokens"]).long()  for s in samples], pad_idx=self.padding_idx),
            "src_distance": pad_2d(       [torch.tensor(s[0]["src_distance"]).float() for s in samples], pad_idx=0.0),
            "src_coord":    pad_coords(   [torch.tensor(s[0]["src_coord"]).float()   for s in samples], pad_idx=0.0),
            "src_edge_type":pad_2d(       [torch.tensor(s[0]["src_edge_type"]).long() for s in samples], pad_idx=self.padding_idx),
        }
        adduct_onehot = torch.stack([s[1] for s in samples])
        labels        = torch.stack([s[2] for s in samples])
        mol_ids       = torch.stack([s[3] for s in samples])
        return net_inputs, adduct_onehot, labels, mol_ids

# keep make_collate_fn as a thin alias if other code calls it
def make_collate_fn(padding_idx: int) -> CollateFn:
    return CollateFn(padding_idx)

class MultiConformerCCSDataset(Dataset):
    def __init__(self, entries, adduct_onehots, labels, K: int):
        assert len(entries) == len(adduct_onehots) == len(labels)
        self.entries = entries
        self.adduct_onehots = torch.tensor(adduct_onehots, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.K = K

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        confs = entry["conformers"][:self.K]
        bw    = entry["boltzmann_weights"][:self.K]
        n     = min(entry["n_conformers"], self.K)

        bw_sum = sum(bw)
        if bw_sum > 0:
            bw = [w / bw_sum for w in bw]

        pad = self.K - n
        confs_padded = confs + [confs[-1]] * pad
        bw_padded    = bw    + [0.0]       * pad
        mask         = [True] * n + [False] * pad

        return (
            confs_padded,
            torch.tensor(bw_padded, dtype=torch.float32),
            torch.tensor(mask,      dtype=torch.bool),
            self.adduct_onehots[idx],
            self.labels[idx],
        )


class GasteigerCCSDataset(MultiConformerCCSDataset):
    def __init__(self, entries, adduct_onehots, labels, K: int,
                 gasteiger_charges: list):
        super().__init__(entries, adduct_onehots, labels, K)
        self.gasteiger_charges = gasteiger_charges

    def __getitem__(self, idx):
        *base, = super().__getitem__(idx)
        charges = torch.tensor(self.gasteiger_charges[idx], dtype=torch.float32)
        return tuple(base) + (charges,)


class GasteigerSingleCCSDataset(CCSDataset):
    def __init__(self, unimol_inputs, adduct_onehots, labels, gasteiger_charges: list,
                 mol_ids=None):
        super().__init__(unimol_inputs, adduct_onehots, labels, mol_ids=mol_ids)
        self.gasteiger_charges = gasteiger_charges

    def __getitem__(self, idx):
        inp, adduct, label, mol_id = super().__getitem__(idx)
        charges = torch.tensor(self.gasteiger_charges[idx], dtype=torch.float32)
        return inp, adduct, label, charges, mol_id


class GasteigerSingleCollateFn:
    def __init__(self, padding_idx: int):
        self.padding_idx = padding_idx

    def __call__(self, samples):
        net_inputs = {
            "src_tokens": pad_1d_tokens(
                [torch.tensor(s[0]["src_tokens"]).long() for s in samples],
                pad_idx=self.padding_idx),
            "src_distance": pad_2d(
                [torch.tensor(s[0]["src_distance"]).float() for s in samples],
                pad_idx=0.0),
            "src_coord": pad_coords(
                [torch.tensor(s[0]["src_coord"]).float() for s in samples],
                pad_idx=0.0),
            "src_edge_type": pad_2d(
                [torch.tensor(s[0]["src_edge_type"]).long() for s in samples],
                pad_idx=self.padding_idx),
        }
        adduct_onehot = torch.stack([s[1] for s in samples])
        labels        = torch.stack([s[2] for s in samples])
        charge_list   = [s[3] for s in samples]
        mol_ids       = torch.stack([s[4] for s in samples])
        L_max = max(c.shape[0] for c in charge_list)
        charges_padded = torch.zeros(len(samples), L_max, dtype=torch.float32)
        for i, c in enumerate(charge_list):
            charges_padded[i, :c.shape[0]] = c
        return net_inputs, adduct_onehot, labels, charges_padded, mol_ids


def make_gasteiger_single_collate_fn(padding_idx: int) -> GasteigerSingleCollateFn:
    return GasteigerSingleCollateFn(padding_idx)


class MultiCollateFn:
    def __init__(self, padding_idx: int, K: int):
        self.padding_idx = padding_idx
        self.K = K

    def __call__(self, samples):
        flat_confs = [c for s in samples for c in s[0]]

        net_inputs_flat = {
            "src_tokens": pad_1d_tokens(
                [torch.tensor(c["src_tokens"]).long() for c in flat_confs],
                pad_idx=self.padding_idx,
            ),
            "src_distance": pad_2d(
                [torch.tensor(c["src_distance"]).float() for c in flat_confs],
                pad_idx=0.0,
            ),
            "src_coord": pad_coords(
                [torch.tensor(c["src_coord"]).float() for c in flat_confs],
                pad_idx=0.0,
            ),
            "src_edge_type": pad_2d(
                [torch.tensor(c["src_edge_type"]).long() for c in flat_confs],
                pad_idx=self.padding_idx,
            ),
        }
        boltzmann_weights = torch.stack([s[1] for s in samples])
        conformer_mask    = torch.stack([s[2] for s in samples])
        adduct_onehot     = torch.stack([s[3] for s in samples])
        labels            = torch.stack([s[4] for s in samples])

        return net_inputs_flat, boltzmann_weights, conformer_mask, adduct_onehot, labels


def make_multi_collate_fn(padding_idx: int, K: int) -> MultiCollateFn:
    return MultiCollateFn(padding_idx, K)


class GasteigerCollateFn:
    def __init__(self, padding_idx: int, K: int):
        self.base_collate = MultiCollateFn(padding_idx, K)

    def __call__(self, samples):
        base_samples = [s[:-1] for s in samples]
        net_inputs_flat, bw, mask, adduct, labels = self.base_collate(base_samples)

        charge_list = [s[-1] for s in samples]
        L_max = max(c.shape[0] for c in charge_list)
        charges_padded = torch.zeros(len(samples), L_max, dtype=torch.float32)
        for i, c in enumerate(charge_list):
            charges_padded[i, :c.shape[0]] = c

        return net_inputs_flat, bw, mask, adduct, labels, charges_padded


def make_gasteiger_collate_fn(padding_idx: int, K: int) -> GasteigerCollateFn:
    return GasteigerCollateFn(padding_idx, K)


class ResidualCCSDataset(Dataset):
    def __init__(self, base_dataset: Dataset, ridge_preds: np.ndarray):
        self.base = base_dataset
        self.ridge_preds = torch.tensor(ridge_preds, dtype=torch.float32)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.base[idx] + (self.ridge_preds[idx],)


class ResidualCollateFn:
    def __init__(self, base_collate_fn):
        self.base_collate_fn = base_collate_fn

    def __call__(self, samples):
        base_samples = [s[:-1] for s in samples]
        ridge_preds  = torch.stack([s[-1] for s in samples])
        return self.base_collate_fn(base_samples) + (ridge_preds,)


def make_residual_collate_fn(base_collate_fn) -> ResidualCollateFn:
    return ResidualCollateFn(base_collate_fn)


class CCSDataModule(L.LightningDataModule):
    def __init__(
        self,
        encoder,
        data_csv,
        split_json,
        cache_dir,
        batch_size: int = 32,
        num_workers: int = 4,
        remove_hs: bool = False,
        num_conformers: int = 1,
        gasteiger_atom: bool = False,
        ridge_preds: np.ndarray = None,
        mol_ids: np.ndarray = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.data_csv = Path(data_csv)
        self.split_json = Path(split_json)
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.remove_hs = remove_hs
        self.num_conformers = num_conformers
        self.gasteiger_atom = gasteiger_atom
        self.ridge_preds = ridge_preds
        self.mol_ids = mol_ids

        base_collate: callable
        if gasteiger_atom and num_conformers > 1:
            base_collate = make_gasteiger_collate_fn(encoder.padding_idx, num_conformers)
        elif gasteiger_atom:
            base_collate = make_gasteiger_single_collate_fn(encoder.padding_idx)
        elif num_conformers == 1:
            base_collate = make_collate_fn(encoder.padding_idx)
        else:
            base_collate = make_multi_collate_fn(encoder.padding_idx, num_conformers)

        if ridge_preds is not None:
            self._collate_fn = make_residual_collate_fn(base_collate)
        else:
            self._collate_fn = base_collate


    def _cache_file(self):
        suffix = "no_hs" if self.remove_hs else "all_h"
        if self.num_conformers == 1:
            return self.cache_dir / f"unimol_inputs_{suffix}.pkl"
        for k in range(self.num_conformers, self.num_conformers + 20):
            candidate = self.cache_dir / f"unimol_inputs_k{k}_{suffix}.pkl"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"No multi-conformer cache found for K>={self.num_conformers} in {self.cache_dir}. "
            f"Run build_conformer_cache.py --num_conformers {self.num_conformers} first."
        )

    def prepare_data(self):
        cache_file = self._cache_file()
        if cache_file.exists():
            return
        if self.num_conformers > 1:
            raise FileNotFoundError(
                f"Multi-conformer cache not found: {cache_file}. "
                f"Run build_conformer_cache.py --num_conformers {self.num_conformers} first."
            )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(self.data_csv)
        smiles_list = df["smiles"].tolist()

        datahub = DataHub(
            data=smiles_list,
            task="repr",
            is_train=False,
            model_name="unimolv1",
            data_type="molecule",
            remove_hs=self.remove_hs,
        )
        unimol_inputs = datahub.data["unimol_input"]

        with open(self._cache_file(), "wb") as f:
            pickle.dump(unimol_inputs, f)


    def setup(self, stage=None):
        cache_file = self._cache_file()
        with open(cache_file, "rb") as f:
            unimol_inputs = pickle.load(f)

        df = pd.read_csv(self.data_csv)
        labels = df["label"].values.astype(np.float32)
        adduct_onehots = np.stack(
            [(df["adducts"] == a).astype(np.float32).values for a in ADDUCT_ORDER],
            axis=1,
        )

        with open(self.split_json) as f:
            splits = json.load(f)

        if self.gasteiger_atom:
            charges_file = self.cache_dir / "gasteiger_charges.pkl"
            if not charges_file.exists():
                raise FileNotFoundError(
                    f"Gasteiger charges cache not found: {charges_file}. "
                    "Run: python -m ccs3d.launch.build_conformer_cache --gasteiger"
                )
            with open(charges_file, "rb") as f:
                all_charges = pickle.load(f)

            if self.num_conformers == 1:
                def _subset(indices):
                    return GasteigerSingleCCSDataset(
                        unimol_inputs=[unimol_inputs[i] for i in indices],
                        adduct_onehots=adduct_onehots[indices],
                        labels=labels[indices],
                        gasteiger_charges=[all_charges[i] for i in indices],
                        mol_ids=self.mol_ids[indices] if self.mol_ids is not None else None,
                    )
            else:
                def _subset(indices):
                    return GasteigerCCSDataset(
                        entries=[unimol_inputs[i] for i in indices],
                        adduct_onehots=adduct_onehots[indices],
                        labels=labels[indices],
                        K=self.num_conformers,
                        gasteiger_charges=[all_charges[i] for i in indices],
                    )
        elif self.num_conformers == 1:
            def _subset(indices):
                return CCSDataset(
                    unimol_inputs=[unimol_inputs[i] for i in indices],
                    adduct_onehots=adduct_onehots[indices],
                    labels=labels[indices],
                    mol_ids=self.mol_ids[indices] if self.mol_ids is not None else None,
                )
        else:
            def _subset(indices):
                return MultiConformerCCSDataset(
                    entries=[unimol_inputs[i] for i in indices],
                    adduct_onehots=adduct_onehots[indices],
                    labels=labels[indices],
                    K=self.num_conformers,
                )

        self.train_dataset = _subset(splits["train"])
        self.val_dataset   = _subset(splits["val"])
        self.test_dataset  = _subset(splits["test"])

        if self.ridge_preds is not None:
            rp = self.ridge_preds
            self.train_dataset = ResidualCCSDataset(self.train_dataset, rp[splits["train"]])
            self.val_dataset   = ResidualCCSDataset(self.val_dataset,   rp[splits["val"]])
            self.test_dataset  = ResidualCCSDataset(self.test_dataset,  rp[splits["test"]])


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def train_eval_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
