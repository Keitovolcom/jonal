"""Interpolation inference for all pair modes, written in the ``alpha_h5`` format.

Every checkpoint of a ``train.py`` run is loaded once and evaluated on the
interpolation lines ``x(alpha) = (1 - alpha) * x0 + alpha * x1`` of all requested
pair modes. Results go to one HDF5 file per mode with the same layout as the
existing ``alpha_h5/alpha_test/<dataset>/<gamma>/64/<kind>.h5`` files:

    <output-root>/<experiment>/<dataset>/<gamma>/<channels>/{noise,no_noise,test_test_no_noise}.h5

    alpha            (A,)          float32
    epochs           (E,)          int32
    predicted_label  (P, E, A)     uint8
    probs            (P, E, A, C)  float32
    label_match      (P, E, A)     int8   -1 if the prediction equals the x0 label, else 0
    present          (P, E)        bool
    pair_names       (P,)          bytes  b"pair<i>/<x0 label>_<x1 training label>"

plus ``x0_index``, ``x1_index`` and ``distance`` (P,). As in the existing files,
clean-noisy pairs store the clean sample at alpha=0 and the noisy one at
alpha=1. Epochs are written in blocks and marked in ``present``; a rerun
resumes from the first block that is not complete.

    python3 interpolation_h5.py --run-dir results/<experiment_name> --gpu 0
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from dataset import load_dataset
from interpolation import build_pairs, find_checkpoints, load_images, make_alphas, parse_epochs
from model import _scaled_channels, create_model
from train import DATASET_SPECS, nonnegative_int, positive_int, resolve_device


MODES = ("clean-noisy", "clean-clean", "test-test")
KIND = {"clean-noisy": "noise", "clean-clean": "no_noise", "test-test": "test_test_no_noise"}
FORMAT_VERSION = 1


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--num-pairs", type=positive_int, default=1000)
    parser.add_argument("--pair-seed", type=nonnegative_int, default=0)
    parser.add_argument("--alpha-start", type=float, default=-0.5)
    parser.add_argument("--alpha-end", type=float, default=1.5)
    parser.add_argument("--alpha-step", type=float, default=0.01)
    parser.add_argument("--epochs", type=parse_epochs, default=None,
                        help="Checkpoint epochs, e.g. '1-4000' (default: all)")
    parser.add_argument("--output-root", type=Path, default=Path("alpha_h5"))
    parser.add_argument("--experiment-name", default=None,
                        help="Directory under --output-root (default: name of --run-dir)")
    parser.add_argument("--gpu", default="auto")
    parser.add_argument("--batch-size", type=positive_int, default=8192)
    parser.add_argument("--epoch-block", type=positive_int, default=50,
                        help="Epochs buffered in memory before each write")
    parser.add_argument("--num-workers", type=nonnegative_int, default=4)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-manifest", action="store_true",
                        help="Do not append to <output-root>/MANIFEST.jsonl")
    parser.add_argument("--pairs-dir", type=Path, default=None,
                        help="Load pairs from <pairs-dir>/<kind>_pairs.json (built there if "
                             "missing) so that several runs share identical pairs")
    parser.add_argument("--pairs-only", action="store_true",
                        help="Only build the pairs in --pairs-dir, then exit")
    return parser


def output_directory(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    experiment = args.experiment_name or args.run_dir.resolve().name
    dataset = config["dataset"].replace("-", "_")
    channels = _scaled_channels(config["model_width"])[0]
    return args.output_root / experiment / dataset / str(config["label_noise"]) / str(channels)


def make_pairs(mode: str, args, train_images, original, training, test_images, test_labels,
               device) -> list[dict[str, Any]]:
    pairs = build_pairs(mode, args.num_pairs, args.pair_seed, train_images, original,
                        training, test_images, test_labels, device)
    if mode == "clean-noisy":
        # build_pairs uses the noisy sample as the query x0; alpha_h5 keeps it at alpha=1.
        for pair in pairs:
            pair["x0"], pair["x1"] = pair["x1"], pair["x0"]
    return pairs


def pair_name(pair: dict[str, Any]) -> str:
    return (f"pair{pair['pair_id'] + 1}/"
            f"{pair['x0']['original_label']}_{pair['x1']['training_label']}")


def write_pairs_csv(path: Path, pairs: list[dict[str, Any]]) -> None:
    fields = ["split", "index", "original_label", "training_label", "is_noisy"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["h5_index", "pair_name", *(f"x0_{f}" for f in fields),
                         *(f"x1_{f}" for f in fields), "distance"])
        for position, pair in enumerate(pairs):
            writer.writerow([position, pair_name(pair),
                             *(pair["x0"][f] for f in fields),
                             *(pair["x1"][f] for f in fields), pair["distance"]])


def create_h5(path: Path, pairs, epochs, alphas, num_classes, attrs) -> None:
    temporary = path.with_name(path.name + ".creating")
    num_pairs, num_epochs, num_alphas = len(pairs), len(epochs), len(alphas)
    with h5py.File(temporary, "w") as log:
        log.create_dataset("alpha", data=alphas.astype(np.float32))
        log.create_dataset("epochs", data=epochs.astype(np.int32))
        log.create_dataset("predicted_label", (num_pairs, num_epochs, num_alphas), np.uint8)
        log.create_dataset("probs", (num_pairs, num_epochs, num_alphas, num_classes), np.float32)
        log.create_dataset("label_match", (num_pairs, num_epochs, num_alphas), np.int8)
        log.create_dataset("present", data=np.zeros((num_pairs, num_epochs), bool))
        log.create_dataset("pair_names", data=np.array([pair_name(p).encode() for p in pairs],
                                                       dtype=object),
                           dtype=h5py.special_dtype(vlen=bytes))
        log.create_dataset("x0_index", data=np.array([p["x0"]["index"] for p in pairs], np.int64))
        log.create_dataset("x1_index", data=np.array([p["x1"]["index"] for p in pairs], np.int64))
        log.create_dataset("distance", data=np.array([p["distance"] for p in pairs], np.float64))
        log.attrs.update(attrs)
        log.attrs["complete"] = False
    temporary.replace(path)


def load_or_build_shared_pairs(mode: str, args, data) -> list[dict[str, Any]]:
    """Pairs stored once in ``--pairs-dir`` so that every run of a sweep uses the same ones."""

    path = args.pairs_dir / f"{KIND[mode]}_pairs.json"
    settings = {"mode": mode, "num_pairs": args.num_pairs, "pair_seed": args.pair_seed}
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored["settings"] != settings:
            raise ValueError(f"{path} was built with {stored['settings']}, not {settings}")
        return stored["pairs"]
    pairs = make_pairs(mode, args, *data)
    args.pairs_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".creating")
    temporary.write_text(json.dumps({"settings": settings, "pairs": pairs}, indent=1) + "\n",
                         encoding="utf-8")
    temporary.replace(path)
    write_pairs_csv(path.with_name(f"{KIND[mode]}_pairs.csv"), pairs)
    return pairs


def open_or_create(path: Path, mode: str, args, data, epochs, alphas, num_classes,
                   base_attrs) -> tuple[h5py.File, list[dict[str, Any]]]:
    """Return the open log and its pairs, creating the file if it does not exist."""

    settings = {"mode": mode, "num_pairs": args.num_pairs, "pair_seed": args.pair_seed,
                "alphas": alphas.tolist(), "epochs": epochs.tolist()}
    if path.exists():
        log = h5py.File(path, "r+")
        stored = json.loads(log.attrs["settings"])
        if stored != settings:
            log.close()
            raise ValueError(f"{path} was created with different settings; remove it first")
        pairs_path = path.with_name(f"{path.stem}_pairs.json")
        pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
        return log, pairs

    pairs = (load_or_build_shared_pairs(mode, args, data) if args.pairs_dir
             else make_pairs(mode, args, *data))
    path.with_name(f"{path.stem}_pairs.json").write_text(
        json.dumps(pairs, indent=1) + "\n", encoding="utf-8")
    write_pairs_csv(path.with_name(f"{path.stem}_pairs.csv"), pairs)
    attrs = {**base_attrs, "mode": mode, "settings": json.dumps(settings),
             "x0": "clean (alpha=0)" if mode == "clean-noisy" else "query (alpha=0)",
             "x1": ("noisy, nearest to x0 with x0's label as original label (alpha=1)"
                    if mode == "clean-noisy" else "nearest same-label neighbour (alpha=1)")}
    create_h5(path, pairs, epochs, alphas, num_classes, attrs)
    return h5py.File(path, "r+"), pairs


@torch.inference_mode()
def main(argv: list[str] | None = None) -> None:
    args = create_parser().parse_args(argv)
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    device = resolve_device(args.gpu)
    alphas = make_alphas(args.alpha_start, args.alpha_end, args.alpha_step)
    checkpoints = find_checkpoints(args.run_dir, args.epochs)
    epochs = np.asarray(list(checkpoints), dtype=np.int64)
    out_dir = output_directory(args, config)
    out_dir.mkdir(parents=True, exist_ok=True)
    specification = DATASET_SPECS[config["dataset"]]
    num_classes = specification["num_classes"]

    train_dataset, test_dataset = load_dataset(
        config["dataset"], data_augmentation=False, label_noise_rate=config["label_noise"],
        seed=config["seed"], input_channels=config["input_channels"],
        data_dir=args.data_dir, download=not args.no_download,
    )
    train_images = load_images(train_dataset, args.num_workers)
    original = train_dataset.original_targets.numpy()
    training = train_dataset.training_targets.numpy()
    test_images = load_images(test_dataset, args.num_workers)
    test_labels = np.asarray(test_dataset.targets, dtype=np.int64)
    data = (train_images, original, training, test_images, test_labels, device)
    if args.pairs_only:
        if args.pairs_dir is None:
            raise ValueError("--pairs-only requires --pairs-dir")
        for mode in args.modes:
            pairs = load_or_build_shared_pairs(mode, args, data)
            print(f"{args.pairs_dir / (KIND[mode] + '_pairs.json')}: {len(pairs)} pairs",
                  flush=True)
        return

    base_attrs = {
        "converter": Path(__file__).name, "format_version": FORMAT_VERSION,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_dir": str(args.run_dir), "train_config": json.dumps(config),
        "max_epoch": int(epochs.max()), "n_bad_files": 0,
        "prob_columns": json.dumps([f"prob_{c}" for c in range(num_classes)]),
    }
    jobs = []
    for mode in args.modes:
        path = out_dir / f"{KIND[mode]}.h5"
        log, pairs = open_or_create(path, mode, args, data, epochs, alphas, num_classes,
                                    base_attrs)
        x0 = torch.stack([(test_images if p["x0"]["split"] == "test" else train_images)
                          [p["x0"]["index"]] for p in pairs])
        x1 = torch.stack([(test_images if p["x1"]["split"] == "test" else train_images)
                          [p["x1"]["index"]] for p in pairs])
        alpha_tensor = torch.as_tensor(alphas, dtype=torch.float32).view(1, -1, 1, 1, 1)
        inputs = (x0.unsqueeze(1) + alpha_tensor * (x1 - x0).unsqueeze(1)).flatten(0, 1)
        x0_labels = np.array([p["x0"]["original_label"] for p in pairs])
        jobs.append({"mode": mode, "path": path, "log": log, "inputs": inputs.to(device),
                     "was_complete": bool(log.attrs["complete"]),
                     "x0_labels": x0_labels[:, None, None], "num_pairs": len(pairs)})
    del train_images, test_images

    model_options = {"in_channels": config["input_channels"], "num_classes": num_classes,
                     "width_multiplier": config["model_width"]}
    if config["model"] == "cnn5layer":
        model_options["input_size"] = specification["input_size"]
    model = create_model(config["model"], **model_options).to(device).eval()

    started = time.time()
    epoch_items = list(checkpoints.items())
    for block_start in range(0, len(epoch_items), args.epoch_block):
        block = epoch_items[block_start:block_start + args.epoch_block]
        stop = block_start + len(block)
        todo = [job for job in jobs if not job["log"]["present"][:, block_start:stop].all()]
        if not todo:
            continue
        buffers = {job["mode"]: np.empty((job["num_pairs"], len(block), len(alphas),
                                          num_classes), np.float32) for job in todo}
        for offset, (epoch, path) in enumerate(block):
            state = torch.load(path, map_location=device, weights_only=False)
            if int(state.get("epoch", epoch)) != epoch:
                raise ValueError(f"{path} stores epoch {state['epoch']}")
            model.load_state_dict(state["model_state_dict"])
            for job in todo:
                inputs = job["inputs"]
                probs = torch.cat([
                    model(inputs[i:i + args.batch_size]).float().softmax(dim=1)
                    for i in range(0, len(inputs), args.batch_size)
                ])
                buffers[job["mode"]][:, offset] = (
                    probs.view(job["num_pairs"], len(alphas), num_classes).cpu().numpy())
        for job in todo:
            probs = buffers[job["mode"]]
            preds = probs.argmax(axis=-1).astype(np.uint8)
            log = job["log"]
            log["probs"][:, block_start:stop] = probs
            log["predicted_label"][:, block_start:stop] = preds
            log["label_match"][:, block_start:stop] = np.where(
                preds == job["x0_labels"], -1, 0).astype(np.int8)
            log.flush()
            log["present"][:, block_start:stop] = True
            log.flush()
        elapsed = time.time() - started
        print(f"epochs {block[0][0]}-{block[-1][0]} written ({stop}/{len(epoch_items)}), "
              f"{elapsed / 60:.1f} min elapsed", flush=True)

    for job in jobs:
        log = job["log"]
        complete = bool(log["present"][:].all())
        log.attrs["complete"] = complete
        log.attrs["n_files"] = int(log["present"][:].sum())
        log.close()
        print(f"{job['path']}: complete={complete}", flush=True)
        if complete and not job["was_complete"] and not args.no_manifest:
            entry = {"h5": str(job["path"]), "source": str(args.run_dir), "mode": job["mode"],
                     "pairs": job["num_pairs"], "epochs": len(epochs),
                     "bytes": job["path"].stat().st_size,
                     "minutes": round((time.time() - started) / 60, 1)}
            with (args.output_root / "MANIFEST.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
