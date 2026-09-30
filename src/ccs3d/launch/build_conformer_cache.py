# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

import os
import argparse
import pickle
import warnings
from pathlib import Path

import ccs3d.utils.rdkit_patch
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from unimol_tools.data import DataHub
from unimol_tools.data.conformer import ConformerGen, coords2unimol

RDLogger.DisableLog("rdApp.*")

ROOT      = Path(__file__).resolve().parents[3]
DATA_CSV  = ROOT / "data" / "data.csv"
CACHE_DIR = ROOT / "data" / "cache"


def _build_single(data_csv: Path, cache_dir: Path, remove_hs: bool):
    suffix     = "no_hs" if remove_hs else "all_h"
    cache_file = cache_dir / f"unimol_inputs_{suffix}.pkl"

    if cache_file.exists():
        print(f"Cache already exists: {cache_file}  (delete to rebuild)")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    df          = pd.read_csv(data_csv)
    smiles_list = df["smiles"].tolist()
    print(f"Generating single conformer for {len(smiles_list)} molecules ...")

    datahub = DataHub(
        data=smiles_list, task="repr", is_train=False,
        model_name="unimolv1", data_type="molecule", remove_hs=remove_hs,
    )
    unimol_inputs = datahub.data["unimol_input"]

    with open(cache_file, "wb") as f:
        pickle.dump(unimol_inputs, f)
    print(f"Done -> {cache_file}")


def _embed_multi(mol, K: int, seed: int):
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=K, randomSeed=seed))
    if not ids:
        ids = list(AllChem.EmbedMultipleConfs(
            mol, numConfs=K, maxAttempts=5000, randomSeed=seed
        ))
    return ids


_KT_KCAL = 0.5924


def _boltzmann_weights(energies: list) -> list:
    e = np.array(energies, dtype=np.float64)
    e -= e.min()
    w = np.exp(-e / _KT_KCAL)
    return (w / w.sum()).tolist()


def _mol_to_entry(smi: str, K: int, seed: int, remove_hs: bool, dictionary) -> dict:
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]

    conf_ids = _embed_multi(mol, K, seed)

    if not conf_ids:
        AllChem.Compute2DCoords(mol)
        coords = mol.GetConformer().GetPositions().astype(np.float32)
        unimol_dict = coords2unimol(atoms, coords, dictionary,
                                    max_atoms=256, remove_hs=remove_hs)
        return {"conformers": [unimol_dict], "energies": [0.0],
                "boltzmann_weights": [1.0], "n_conformers": 1, "is_3d": False}

    mmff_results = AllChem.MMFFOptimizeMoleculeConfs(mol)
    energies = [r[1] for r in mmff_results] if mmff_results else [0.0] * len(conf_ids)

    conformers = []
    valid_energies = []
    for i, cid in enumerate(conf_ids):
        coords = mol.GetConformer(cid).GetPositions().astype(np.float32)
        unimol_dict = coords2unimol(atoms, coords, dictionary,
                                    max_atoms=256, remove_hs=remove_hs)
        conformers.append(unimol_dict)
        valid_energies.append(energies[i] if i < len(energies) else 0.0)

    return {"conformers": conformers, "energies": valid_energies,
            "boltzmann_weights": _boltzmann_weights(valid_energies),
            "n_conformers": len(conformers), "is_3d": True}


def _build_multi(data_csv: Path, cache_dir: Path, remove_hs: bool, K: int, seed: int):
    suffix     = "no_hs" if remove_hs else "all_h"
    cache_file = cache_dir / f"unimol_inputs_k{K}_{suffix}.pkl"

    if cache_file.exists():
        print(f"Cache already exists: {cache_file}  (delete to rebuild)")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    df          = pd.read_csv(data_csv)
    smiles_list = df["smiles"].tolist()
    print(f"Generating K={K} conformers for {len(smiles_list)} molecules ...")

    cgen       = ConformerGen(data_type="molecule", remove_hs=remove_hs)
    dictionary = cgen.dictionary

    entries   = []
    n_2d      = 0
    n_partial = 0
    for smi in tqdm(smiles_list, unit="mol"):
        entry = _mol_to_entry(smi, K, seed, remove_hs, dictionary)
        entries.append(entry)
        if not entry["is_3d"]:
            n_2d += 1
        elif entry["n_conformers"] < K:
            n_partial += 1

    print(f"  3D success (K={K}):  {len(entries) - n_2d} "
          f"({100*(len(entries)-n_2d)/len(entries):.1f}%)")
    print(f"  Partial (<{K} confs): {n_partial}")
    print(f"  2D fallback:         {n_2d} ({100*n_2d/len(entries):.1f}%)")

    with open(cache_file, "wb") as f:
        pickle.dump(entries, f)
    print(f"Done -> {cache_file}")


def _build_gasteiger(data_csv: Path, cache_dir: Path):
    cache_file = cache_dir / "gasteiger_charges.pkl"
    if cache_file.exists():
        print(f"Cache already exists: {cache_file}  (delete to rebuild)")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_csv)
    smiles_list = df["smiles"].tolist()
    print(f"Computing Gasteiger charges for {len(smiles_list)} molecules ...")

    all_charges = []
    n_failed = 0
    for smi in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            all_charges.append(np.zeros(0, dtype=np.float32))
            n_failed += 1
            continue
        mol = AllChem.AddHs(mol)
        AllChem.ComputeGasteigerCharges(mol)
        charges = []
        for atom in mol.GetAtoms():
            q = atom.GetDoubleProp("_GasteigerCharge")
            charges.append(0.0 if (q != q or abs(q) == float("inf")) else q)
        all_charges.append(np.array(charges, dtype=np.float32))

    with open(cache_file, "wb") as f:
        pickle.dump(all_charges, f)
    print(f"Done -> {cache_file}  ({n_failed} failures set to zero-charge)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_csv",       default=str(DATA_CSV),  help="Path to data.csv")
    p.add_argument("--cache_dir",      default=str(CACHE_DIR), help="Cache directory")
    p.add_argument("--remove_hs",      action="store_true",    help="Use heavy-atom model")
    p.add_argument("--num_conformers", type=int, default=1,
                   help="Conformers per molecule (1 = single, existing format; >1 = multi)")
    p.add_argument("--seed",           type=int, default=42,   help="RDKit random seed")
    p.add_argument("--gasteiger",      action="store_true",
                   help="Build Gasteiger charges cache (gasteiger_charges.pkl)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.gasteiger:
        _build_gasteiger(Path(args.data_csv), Path(args.cache_dir))
    elif args.num_conformers == 1:
        _build_single(Path(args.data_csv), Path(args.cache_dir), args.remove_hs)
    else:
        _build_multi(Path(args.data_csv), Path(args.cache_dir),
                     args.remove_hs, args.num_conformers, args.seed)
