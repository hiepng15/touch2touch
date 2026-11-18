"""Evaluate Soft Bubble accuracy on the main, rotation, or GelSlim padding test."""

import argparse
import csv
import json
import math
import os
import re
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from .compare_checkpoints import (
    DIFFUSION, FINGERS, GELSLIM_PADDING_SCALE, REPO, SEED, configure_diffusion_imports,
    existing_predictions, fid_features, frechet, load, load_diffusion_predictions,
    pad_gelslim, paired_samples, pixel_metrics, run_diffusion, save_csv, save_json,
    time_seconds, vqvae_depth,
)


METHODS = {
    "bubble_gt": "Bubble ground truth",
    "vqvae_drive": "VQ-VAE",
    "diffusion_unshifted_aligned": "Diffusion",
    "diffusion_shift_aligned": "Diffusion + Shift",
}
PADDING_METHODS = {
    "bubble_gt": "Bubble ground truth",
    "diffusion_shift_aligned": "Diffusion + Shift",
    "diffusion_padded_shift_aligned": "Padded Diffusion + Shift",
}
PADDING_CONDITIONS = ("unpadded", "padded")
BUBBLE_FIELDS = ("bubble_depth_ref", "K", "bubbles_tr_quat", "grasp_frame_quat", "theta", "x", "y")
BUBBLE_SHAPES = {
    "bubble_imprint": (2, 1, 140, 175),
    "bubble_depth_ref": (2, 1, 140, 175),
    "K": (2, 3, 3),
    "bubbles_tr_quat": (2, 1, 7),
    "grasp_frame_quat": (1, 7),
}
POSE_COLUMNS = ("x", "y", "theta")
ICP_METRICS = ("mean_error_deg", "std_error_deg", "within_5_deg_pct", "within_10_deg_pct")
IMAGE_METRICS = ("psnr_db", "ssim", "fid")
ROTATION_ANGLES = (0, 90, 180, 270)
ROTATION_TOOL_IDS = ("pattern_04", "pattern_05", "pattern_31", "pattern_33", "pattern_35", "pattern_36")
MASK_EPOCHS = (5, 10, 20, 30)


@dataclass(frozen=True)
class Sample:
    bubble: Path
    gelslim: Path
    prediction_name: str
    nominal_rotation_deg: Optional[int] = None


def test_manifest(data, by_tool):
    with (data / "data_info.csv").open(encoding="utf-8-sig", newline="") as source:
        manifest = {}
        for row in csv.DictReader(source):
            name = f"{row['Tool_name']}_data_{row['Data_idx']}.pt"
            if name in manifest:
                raise ValueError(f"Duplicate metadata row: {name}")
            pose = {key: float(row[key]) for key in POSE_COLUMNS}
            if not all(map(math.isfinite, pose.values())):
                raise ValueError(f"Nonfinite metadata row: {name}")
            manifest[name] = pose
    expected = {name for names in by_tool.values() for name in names}
    if set(manifest) != expected:
        raise ValueError(f"data_info.csv and paired filenames differ: {sorted(set(manifest) ^ expected)[:5]}")
    return manifest


def rotation_samples(data, cad_dir):
    cad_tools = {}
    for path in cad_dir.glob("*.stl"):
        match = re.match(r"pattern_\d+", path.stem)
        if match and match.group() in ROTATION_TOOL_IDS:
            tool_id = match.group()
            if tool_id in cad_tools:
                raise ValueError(f"Multiple CAD meshes for rotation tool {tool_id} in {cad_dir}")
            cad_tools[tool_id] = path.stem
    missing = set(ROTATION_TOOL_IDS) - set(cad_tools)
    if missing:
        raise ValueError(f"Missing rotation CAD tools: {sorted(missing)}")

    fixed = {f"pattern_04_rotation_test_{angle}": angle for angle in ROTATION_ANGLES}
    groups = defaultdict(list)
    mixed_angles = set()
    for group in (*fixed, "rotation_test"):
        directory = data / group
        bubble_files = {path.name for path in (directory / "bubbles").glob("*.pt") if path.is_file()}
        gelslim_files = {path.name for path in (directory / "gelslim").glob("*.pt") if path.is_file()}
        expected_count = 12 if group in fixed else 24
        if bubble_files != gelslim_files or len(bubble_files) != expected_count:
            raise ValueError(f"{directory}: expected {expected_count} matching Bubble/GelSlim pairs, "
                             f"found {len(bubble_files)} Bubble and {len(gelslim_files)} GelSlim files; "
                             f"unpaired examples: {sorted(bubble_files ^ gelslim_files)[:5]}")
        for index, name in enumerate(sorted(bubble_files)):
            if group in fixed:
                tool_id, angle = "pattern_04", fixed[group]
            else:
                tool_match = re.match(r"pattern_\d+(?=[_-])", name)
                angle_match = re.search(r"[_-](0|90|180|270)\.pt$", name)
                if (tool_match is None or angle_match is None
                        or tool_match.group() not in ROTATION_TOOL_IDS):
                    raise ValueError(f"Unrecognized rotation tool or angle: {directory / name}")
                tool_id, angle = tool_match.group(), int(angle_match[1])
                if (tool_id, angle) in mixed_angles:
                    raise ValueError(f"Duplicate combination of tool and angle in {directory}: {name}")
                mixed_angles.add((tool_id, angle))
            # Group names disambiguate repeated basenames. The estimator needs a trailing integer.
            prediction_name = f"{group}__{Path(name).stem}_data_{index}.pt"
            groups[group, cad_tools[tool_id]].append(
                Sample(directory / "bubbles" / name, directory / "gelslim" / name,
                       prediction_name, angle)
            )
    return groups


def rotation_angle_error_deg(angle, target):
    return abs((angle - target + 180) % 360 - 180)


def read_prediction_pair(sample, paths):
    for side, path in zip(FINGERS, paths):
        if path.name != sample.prediction_name or f"results_samples_{side}" not in path.parts:
            raise ValueError(f"{sample.prediction_name}: invalid {side} prediction path {path}")
    return load_diffusion_predictions(paths)


def read_sample(sample, paths=None, pose=None):
    name = sample.prediction_name
    bubble = load(sample.bubble)
    gel = load(sample.gelslim)
    for key, shape in BUBBLE_SHAPES.items():
        value = bubble[key]
        if value.shape != shape or not torch.isfinite(value).all():
            raise ValueError(f"{name}: invalid Bubble {key}, expected finite tensor of shape {shape}")
    if not (bubble["K"][:, (0, 1), (0, 1)] > 0).all():
        raise ValueError(f"{name}: nonpositive camera focal length")
    for key in ("bubbles_tr_quat", "grasp_frame_quat"):
        if not (bubble[key][..., 3:].square().sum(dim=-1) > 0).all():
            raise ValueError(f"{name}: zero quaternion in {key}")

    image, reference = gel["gelslim"], gel["gelslim_ref"]
    if image.ndim != 4 or image.shape[:2] != (2, 3) or min(image.shape[-2:]) < 1 or image.shape != reference.shape:
        raise ValueError(f"{name}: invalid GelSlim image or reference shape")
    if not torch.isfinite(image).all() or not torch.isfinite(reference).all():
        raise ValueError(f"{name}: nonfinite GelSlim image or reference")
    for key in POSE_COLUMNS:
        recorded = bubble[key]
        if recorded.numel() != 1 or not torch.isfinite(recorded).all():
            raise ValueError(f"{name}: invalid Bubble {key}")
        target = float(recorded) if pose is None else pose[key]
        reference = "Bubble metadata" if pose is None else "data_info.csv"
        for sensor, value in (("Bubble", recorded), ("GelSlim", gel[key])):
            if value.numel() != 1 or not math.isclose(float(value), target, rel_tol=1e-6, abs_tol=1e-8):
                raise ValueError(f"{name}: {sensor} {key} differs from {reference}")

    predictions = read_prediction_pair(sample, paths) if paths is not None else None
    return bubble, gel, predictions


def generate_predictions(groups, data, output, weights, dataset, deadline):
    from types import SimpleNamespace

    for path in (weights, weights.with_name("training_stats.pt"),
                 DIFFUSION / "models/first_stage_models/kl-f8/model.ckpt"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError(f"No CUDA GPU available for {dataset} inference")
    samples = [(tool, sample.prediction_name) for (_, tool), items in groups.items() for sample in items]
    gelslim_files = {sample.prediction_name: sample.gelslim for items in groups.values() for sample in items}
    options = SimpleNamespace(diffusion_weights=weights, ddim_steps=200, batch_pairs=12)
    configure_diffusion_imports()
    output.mkdir(parents=True, exist_ok=False)
    run = {
        "status": "running", "dataset": dataset, "source": str(data),
        "diffusion_weights": str(weights), "ddim_steps": options.ddim_steps, "seed": SEED,
        "input_files": {name: str(path.relative_to(data)) for name, path in gelslim_files.items()},
    }
    if dataset == "padding":
        run["padding_scale_per_axis"] = GELSLIM_PADDING_SCALE
        run["condition_pairs"] = {}
    save_json(output / "run.json", run)
    conditions = PADDING_CONDITIONS if dataset == "padding" else ("",)
    cwd = Path.cwd()
    try:
        os.chdir(DIFFUSION)
        for condition in conditions:
            destination = output / condition
            if condition:
                destination.mkdir(exist_ok=False)
            print(f"Generating {len(samples)} pairs for {condition or dataset}", flush=True)
            # run_diffusion seeds the entire sampler on each call. Both conditions
            # use identical sample order, batch sizes, checkpoint and DDIM settings.
            completed, error = run_diffusion(samples, gelslim_files, destination, options, deadline,
                                             padded=condition == "padded")
            run["completed_pairs"] = len(completed)
            if condition:
                run["condition_pairs"][condition] = len(completed)
            run["inference_error"] = error
            if len(completed) != len(samples):
                run["status"] = "partial"
                save_json(output / "run.json", run)
                raise RuntimeError(f"{condition or dataset} predictions incomplete: {len(completed)}/{len(samples)}")
            save_json(output / "run.json", run)
    finally:
        os.chdir(cwd)
    run["status"] = "complete"
    save_json(output / "run.json", run)
    print(f"Saved {len(samples)} {dataset} pairs per condition to {output}", flush=True)


def aligned_diffusion_depth(predictions, size):
    from torchvision.transforms.functional import resize, rgb_to_grayscale

    # Undo the fixed rotation of the Bubble training targets.
    return torch.cat([resize(rgb_to_grayscale(pred), size) for pred in predictions]).flip((-2, -1))


def load_inception():
    import cv2
    from pytorch_fid.inception import InceptionV3

    return InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).eval()


def save_examples(output, sample, gel, images, masks_by_epoch, mask_predictor, methods, show_padding):
    from matplotlib import pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    label = sample.bubble.stem
    if sample.nominal_rotation_deg is not None:
        label += f" | nominal rotation {sample.nominal_rotation_deg}°"
    titles = tuple(methods.values())
    input_columns = 2 if show_padding else 1
    depths = torch.stack(images)
    lo, hi = float(depths.min()), float(depths.max())
    # Use a common scale for both fingers and all Bubble outputs in the figure.
    fig, axes = plt.subplots(2, input_columns + len(titles), figsize=(13, 5), constrained_layout=True)
    try:
        for index, side in enumerate(FINGERS):
            rgb = gel["gelslim"][index].float().permute(1, 2, 0).clamp(0, 255) / 255
            axes[index, 0].imshow(rgb)
            axes[index, 0].set_title(f"GelSlim RGB ({side})")
            if show_padding:
                axes[index, 1].imshow(pad_gelslim(rgb.permute(2, 0, 1)).permute(1, 2, 0))
                axes[index, 1].set_title(f"Padded GelSlim ({side})")
            for column, (title, image) in enumerate(zip(titles, images), input_columns):
                display = axes[index, column].imshow(image[index, 0], cmap="viridis", vmin=lo, vmax=hi)
                axes[index, column].set_title(f"{title} ({side})")
        for axis in axes.flat:
            axis.set_axis_off()
        fig.colorbar(display, ax=axes[:, input_columns:].ravel().tolist(), label="Depth change (m)")
        fig.suptitle(label)
        fig.savefig(output / f"{sample.bubble.stem}_images.png", dpi=160)
    finally:
        plt.close(fig)

    mask_inputs = depths.flatten(0, 1)
    fig, axes = plt.subplots(len(mask_inputs), len(masks_by_epoch), figsize=(8, 14), constrained_layout=True)
    try:
        for column, (epoch, path) in enumerate(masks_by_epoch.items()):
            masks = mask_predictor(mask_inputs, str(path))
            if masks.shape != mask_inputs.shape or not torch.isfinite(masks).all():
                raise ValueError(f"{sample.bubble.name}: invalid E{epoch} contact masks")
            for row, axis in enumerate(axes[:, column]):
                axis.imshow(masks[row, 0], cmap="gray", vmin=0, vmax=1)
                axis.set_axis_off()
                if column == 0:
                    axis.set_title(f"{titles[row // 2]} ({FINGERS[row % 2]})", fontsize=8, loc="left")
                if row == 0:
                    axis.text(0.5, 0.92, f"E{epoch}", transform=axis.transAxes,
                              color="white", ha="center",
                              bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.7})
        fig.suptitle(f"Contact masks | {label}")
        fig.savefig(output / f"{sample.bubble.stem}_masks.png", dpi=160)
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("main", "rotation", "padding"), default="main")
    parser.add_argument("--predictions", type=Path, help="Prediction directory or a run with completed.json")
    parser.add_argument("--output", type=Path, help="New directory for scores, images or generated predictions")
    parser.add_argument(
        "--samples-per-tool", type=int,
        help="Limit each tool and dataset group to this many samples for a pilot run.",
    )
    parser.add_argument("--examples-per-tool", type=int, default=0,
                        help="Save this many image and E5/E10/E20/E30 mask comparisons per tool and orientation")
    parser.add_argument("--image-metrics", action="store_true", help="Also compute PSNR, SSIM and FID per tool")
    parser.add_argument("--time-left", help="Slurm time left for generation and scoring, e.g. 05:55:00")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight-only", action="store_true", help="Validate all inputs without scoring")
    modes.add_argument("--generate-predictions", action="store_true",
                       help="Generate rotation or paired padding predictions on GPU with the released weights")
    args = parser.parse_args()
    if args.samples_per_tool is not None and args.samples_per_tool < 1:
        parser.error("--samples-per-tool must be positive")
    if args.examples_per_tool < 0:
        parser.error("--examples-per-tool cannot be negative")
    if args.preflight_only and (args.output is not None or args.samples_per_tool is not None):
        parser.error("--preflight-only checks all samples without writing an output directory")
    if args.generate_predictions and (args.image_metrics or args.examples_per_tool):
        parser.error("Generate predictions first, then score and export examples in a separate invocation")
    if args.time_left is not None and not args.generate_predictions:
        parser.error("--time-left only applies to prediction generation")
    if not args.preflight_only and args.output is None:
        parser.error("--output is required")
    rotation = args.dataset == "rotation"
    padding = args.dataset == "padding"
    if args.generate_predictions and (args.dataset == "main" or args.predictions is not None):
        parser.error("--generate-predictions requires --dataset rotation or padding and uses the released weights")
    if args.generate_predictions and rotation and args.samples_per_tool is not None:
        parser.error("Rotation prediction generation requires all 72 samples")
    if args.dataset != "main" and not (args.preflight_only or args.generate_predictions) and args.predictions is None:
        parser.error(f"Provide --predictions from a {args.dataset} inference run")

    started = time.monotonic()
    test_data = REPO / "data/test"
    data = test_data / ("out_distribution_orientations" if rotation else "test_only")
    cad_dir = REPO / "data/tools_stls"
    diffusion = REPO / "checkpoints/diffusion/rot_flip"
    vq_weights = REPO / "checkpoints/vq_vae/rot_flip/checkpoint.pt"
    mask_root = REPO / "checkpoints/masking_unet"
    masks_by_epoch = {
        epoch: mask_root / f"masking_sdf_new_dataset_model_new_dataset_tools_E{epoch}_B32_LR0.00001.pth"
        for epoch in MASK_EPOCHS
    }
    mask = masks_by_epoch[30]
    splits = load(diffusion / "tools.pt")
    seen, unseen = set(splits["train_tools"]), set(splits["test_tools"])
    manifest = {}
    if rotation:
        groups = rotation_samples(data, cad_dir)
    else:
        if not padding:
            args.predictions = args.predictions or REPO / "results/comparison-126686"
        by_tool = defaultdict(list)
        for tool, name in paired_samples(data):
            by_tool[tool].append(name)
        if (len(seen) != 9 or len(unseen) != 3 or seen & unseen or (seen | unseen) != set(by_tool)
                or any(len(names) != 46 for names in by_tool.values())):
            raise ValueError("Expected 9 seen and 3 unseen tools, 46 paired samples per tool in test_only")
        manifest = test_manifest(data, by_tool)
        groups = {
            ("unseen_grasps" if tool in seen else "unseen_tools", tool): [
                Sample(data / "bubbles" / name, data / "gelslims" / name, name) for name in names]
            for tool, names in by_tool.items() if not padding or tool in unseen
        }

    groups = {key: items[:args.samples_per_tool] for key, items in groups.items()}
    samples = [(tool, sample.prediction_name) for (_, tool), items in groups.items() for sample in items]
    torch.set_num_threads(1)
    if args.generate_predictions:
        for items in groups.values():
            for sample in items:
                read_sample(sample, pose=manifest.get(sample.prediction_name))
        deadline = math.inf
        if args.time_left is not None:
            evaluation_reserve = 90 * 60 if padding and len(samples) == 138 else 10 * 60
            deadline = started + time_seconds(args.time_left) - evaluation_reserve
            if time.monotonic() >= deadline:
                raise RuntimeError("Insufficient time left once the evaluation reserve is set aside")
        generate_predictions(groups, data, args.output.resolve(), diffusion / "checkpoint.ckpt", args.dataset, deadline)
        return

    if not padding:
        vq_splits = load(vq_weights.with_name("tools.pt"))
        if seen != set(vq_splits["train_tools"]) or unseen != set(vq_splits["test_tools"]):
            raise ValueError("VQ-VAE and diffusion tools.pt splits differ")
    stats = load(diffusion / "stats.pt")
    gt_mean, gt_std = map(float, stats["gt_stats"])
    pred_mean, pred_std = map(float, stats["pred_stats"])
    if not all(map(math.isfinite, (gt_mean, gt_std, pred_mean, pred_std))) or min(gt_std, pred_std) <= 0:
        raise ValueError("Invalid Shift statistics in stats.pt")

    completed = {}
    padded_completed = {}
    if args.predictions is not None:
        source = args.predictions.resolve()
        if padding:
            origin = json.loads((source / "run.json").read_text(encoding="utf-8"))
            if (origin.get("dataset"), origin.get("status"), origin.get("padding_scale_per_axis")) != (
                "padding", "complete", GELSLIM_PADDING_SCALE
            ):
                raise ValueError(f"Expected a completed paired padding run in {source}")
            completed = existing_predictions(samples, source / PADDING_CONDITIONS[0], data.name)
            padded_completed = existing_predictions(samples, source / PADDING_CONDITIONS[1], data.name)
        else:
            completed = existing_predictions(samples, source, data.name)
        for _, name in samples:
            if name not in completed or (padding and name not in padded_completed):
                raise ValueError(f"Missing prediction pair for {name} in {source}")
    required_masks = list(masks_by_epoch.values()) if args.examples_per_tool else [mask]
    required = [*required_masks, *(cad_dir / f"{tool}.stl" for _, tool in groups)]
    if not padding:
        required.append(vq_weights)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    if args.preflight_only:
        if completed:
            from torchvision.transforms.functional import rgb_to_grayscale

        channel_gap = 0.0
        theta_by_orientation = defaultdict(list)
        total_so_far = 0
        for (split, _), items in groups.items():
            for sample in items:
                name = sample.prediction_name
                bubble, _, predictions = read_sample(sample, completed.get(name), manifest.get(name))
                if sample.nominal_rotation_deg is not None:
                    theta_by_orientation[split, sample.nominal_rotation_deg].append(
                        math.degrees(float(bubble["theta"])))
                if padded_completed:
                    predictions += read_prediction_pair(sample, padded_completed[name])
                for pred in predictions or ():
                    delta = (pred.mean(1, keepdim=True) - rgb_to_grayscale(pred)).abs().max()
                    channel_gap = max(channel_gap, delta.item())
            total_so_far += len(items)
            print(f"Checked {total_so_far}/{len(samples)} samples", flush=True)
        for (split, nominal), angles in sorted(theta_by_orientation.items()):
            print(f"{split}, nominal {nominal}°: {len(angles)} pairs, "
                  f"recorded theta range {min(angles):.4g}° to {max(angles):.4g}°")
        if args.image_metrics:
            from touch2touch.evaluation.ICP_dataset_cal import icp_dataset_cal
            load_inception()
        cached = 2 * (len(completed) + len(padded_completed))
        print(f"Preflight OK: {len(samples)} pairs, {cached} cached right/left predictions")
        if completed:
            print(f"Max RGB-mean vs torchvision-grayscale difference on saved scale: {channel_gap:.6g}")
        return

    if args.examples_per_tool:
        import matplotlib
        matplotlib.use("Agg")
    from touch2touch.evaluation.ICP_dataset_cal import (
        get_bubbles_data_TST, get_bubbles_masks, get_metrics, icp_dataset_cal,
    )

    inception = load_inception() if args.image_metrics else None
    if not padding:
        from touch2touch.vq_vae.modules import model_definition
        vq = model_definition("VQ-VAE-small", 3, 1, 256, 16385, "cpu", single=True).eval()
        vq.load_state_dict(load(vq_weights), strict=True)
    methods = PADDING_METHODS if padding else METHODS
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    run = {
        "status": "running", "dataset": args.dataset, "samples": len(samples),
        "samples_per_tool": args.samples_per_tool,
        "prediction_source": str(args.predictions.resolve()), "test_data": str(data),
        "vq_checkpoint": None if padding else str(vq_weights), "icp_checkpoint": str(mask),
        "shift_stats": str(diffusion / "stats.pt"), "diffusion_rotation_deg": 180,
        "diffusion_unshifted_scale": "saved prediction scale before applying Shift",
        "shift_on_saved_scale": {"gt_mean": gt_mean, "gt_std": gt_std, "pred_mean": pred_mean, "pred_std": pred_std},
        "diffusion_rgb_reduction": "torchvision grayscale, matching the published Shift code",
        "angle_success_thresholds_deg": [5, 10], "threshold_inclusive": True,
        "angle_error": "shortest distance modulo 360" if rotation else "absolute difference",
        "image_metrics": args.image_metrics, "aggregation": "equal weight per tool within each split",
        "examples_per_tool_and_orientation": args.examples_per_tool,
        "example_mask_epochs": list(masks_by_epoch) if args.examples_per_tool else [],
        "unseen_grasps_tools": sorted(seen), "unseen_tools": sorted(unseen),
    }
    if padding:
        run.update(
            padding_scale_per_axis=GELSLIM_PADDING_SCALE,
            padding_study="centered zero-padding before resize and normalization at inference only",
            padding_scope="unseen tools",
        )
    save_json(output / "run.json", run)
    print(f"Evaluating {len(samples)} {args.dataset} samples on CPU", flush=True)
    rows = []
    per_tool = []
    with tempfile.TemporaryDirectory(prefix="touch2touch-icp-") as work, torch.inference_mode():
        for (split, tool), items in groups.items():
            inputs = {method: [] for method in methods}
            examples_written = defaultdict(int)
            for sample in items:
                name = sample.prediction_name
                bubble, gel, saved = read_sample(sample, completed[name], manifest.get(name))
                gt = bubble["bubble_imprint"]
                size = list(gt.shape[-2:])
                # Training targets are rotated. Align both rows to the same sensor frame
                # and apply the affine Shift to only one copy of the saved prediction.
                diff = aligned_diffusion_depth(saved, size)
                shifted = (diff - pred_mean) / pred_std * gt_std + gt_mean
                info = {"bubbles_data": {key: bubble[key].unsqueeze(0) for key in BUBBLE_FIELDS}}
                if padding:
                    padded_depth = aligned_diffusion_depth(read_prediction_pair(sample, padded_completed[name]), size)
                    images = (gt, shifted, (padded_depth - pred_mean) / pred_std * gt_std + gt_mean)
                else:
                    images = (gt, vqvae_depth(vq, gel, size, "cpu"), diff, shifted)
                for method, image in zip(methods, images):
                    if image.shape != (2, 1, 140, 175) or not torch.isfinite(image).all():
                        raise ValueError(f"Invalid Bubble image for {method}: {name}")
                    inputs[method].extend(get_bubbles_data_TST(image.unsqueeze(0), info))
                orientation = sample.nominal_rotation_deg
                if examples_written[orientation] < args.examples_per_tool:
                    save_examples(output / "visuals" / split / tool, sample, gel,
                                  images, masks_by_epoch, get_bubbles_masks, methods, show_padding=padding)
                    examples_written[orientation] += 1

            stl = str(cad_dir / f"{tool}.stl")
            if inception is not None:
                gt_images = torch.cat([item["bubble_imprint"] for item in inputs["bubble_gt"]])
                gt_features = fid_features(gt_images, inception, "cpu", work)
            for method, bubbles in inputs.items():
                print(f"{split} {tool} {method}: {len(items)} samples", flush=True)
                # The original ICP function persists intermediates. Scratch directories scope them to this run.
                angles, targets, scores, _ = icp_dataset_cal(
                    stl, bubbles, str(Path(work) / split / tool / method),
                    icp_masking=True, icp_model_path=str(mask),
                )
                if len(angles) != len(items) or len(targets) != len(items) or not all(
                    map(math.isfinite, (*angles, *targets))
                ):
                    raise ValueError(f"Invalid ICP angles for {tool}: {method}")
                errors = []
                for sample, angle, target in zip(items, angles, targets):
                    angle, target = float(angle), float(target)
                    error = rotation_angle_error_deg(angle, target) if rotation else abs(angle - target)
                    errors.append(error)
                    row = dict(
                        split=split, tool=tool, sample=sample.bubble.name, method=method, estimated_deg=angle,
                        target_deg=target, absolute_error_deg=error,
                    )
                    if rotation:
                        row["nominal_rotation_deg"] = sample.nominal_rotation_deg
                    rows.append(row)
                result = dict(split=split, tool=tool, method=method, samples=len(items))
                if rotation:
                    _, mean, std, within_5, within_10 = get_metrics(errors, [0.0] * len(errors))
                    result.update(zip(ICP_METRICS, (mean, std, within_5, within_10)))
                else:
                    result.update(zip(ICP_METRICS, (scores["mean"], scores["std"], scores["acc_5"], scores["acc_10"])))
                if inception is not None:
                    result.update(dict.fromkeys(IMAGE_METRICS, ""))
                    if method != "bubble_gt":
                        images = torch.cat([item["bubble_imprint"] for item in bubbles])
                        pixel = pixel_metrics(gt_images, images, "cpu")
                        result.update(psnr_db=pixel["psnr_db"], ssim=pixel["ssim"],
                                      fid=frechet(gt_features, fid_features(images, inception, "cpu", work)))
                per_tool.append(result)
            save_csv(output / "angles.csv", rows)
            save_csv(output / "metrics_by_tool.csv", per_tool)

    groups = defaultdict(list)
    for row in per_tool:
        groups[row["split"], row["method"]].append(row)
    summary = []
    for (split, method), group in groups.items():
        result = dict(split=split, method=method, samples=sum(row["samples"] for row in group))
        for key in ICP_METRICS:
            if key == "std_error_deg":
                # The official notebook reports the root mean square of each tool's std.
                result[key] = math.sqrt(sum(row[key] ** 2 for row in group) / len(group))
            else:
                result[key] = sum(row[key] for row in group) / len(group)
        if inception is not None:
            for key in IMAGE_METRICS:
                result[key] = "" if method == "bubble_gt" else sum(row[key] for row in group) / len(group)
        summary.append(result)
    save_csv(output / "summary.csv", summary)
    if rotation:
        by_angle = defaultdict(list)
        for row in rows:
            by_angle[row["split"], row["nominal_rotation_deg"], row["method"]].append(row)
        rotation_summary = []
        for (split, nominal, method), group in sorted(by_angle.items()):
            errors = [row["absolute_error_deg"] for row in group]
            _, mean, std, within_5, within_10 = get_metrics(errors, [0.0] * len(errors))
            rotation_summary.append(dict(
                split=split, nominal_rotation_deg=nominal, method=method, samples=len(group),
                mean_error_deg=mean, std_error_deg=std,
                within_5_deg_pct=within_5, within_10_deg_pct=within_10,
            ))
        save_csv(output / "rotation_by_angle.csv", rotation_summary)
    run["status"] = "complete"
    save_json(output / "run.json", run)
    print(f"Saved ICP angles and summary to {output}")


if __name__ == "__main__":
    main()
