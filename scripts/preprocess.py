import os
import yaml
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


# preprocessing modules
from preprocessing.utils          import load_nifti_as_array, get_spacing_from_affine
from preprocessing.resample_utils import resample_volume, resample_mask
from preprocessing.skullstrip     import apply_brain_mask
from preprocessing.windowing      import apply_window, normalize_volume, apply_clahe, apply_gamma
from preprocessing.shape_utils    import to_standard_axis, pad_or_crop_3d
from preprocessing.volume_utils   import mip_projection, aip_projection, mid_plane

# --- Projection function map ---
PROJ_FNS = {
    "mip": mip_projection,
    "aip": aip_projection,
    "mid": mid_plane,
}

def preprocess_case(case_id: str, cfg: dict):
    # load paths & config
    data_dir     = Path(os.environ.get("DATA_DIR", cfg["data_dir"]))
    raw_path     = data_dir / "raw"         / f"{case_id}{cfg['extensions']['raw']}"
    brainm_path  = data_dir / "brain_masks" / f"{case_id}{cfg['extensions']['brain_mask']}"
    mask_path    = data_dir / "masks"       / f"{case_id}{cfg['extensions']['lesion']}"
    out_path     = data_dir / "processed"   / f"{case_id}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Load + reorient
    vol, affine = load_nifti_as_array(str(raw_path), reorient=True)
    mask, _     = load_nifti_as_array(str(mask_path), reorient=True)
    brainm, _   = load_nifti_as_array(str(brainm_path), reorient=True)

    # 2) Resample to isotropic
    orig_sp = tuple(cfg["spacings"]["original"])
    tgt_sp  = tuple(cfg["spacings"]["target"])
    vol_r    = resample_volume(vol,    original_spacing=orig_sp, target_spacing=tgt_sp, order=1)
    mask_r   = resample_mask(mask,     original_spacing=orig_sp, target_spacing=tgt_sp)
    brainm_r = resample_mask(brainm,   original_spacing=orig_sp, target_spacing=tgt_sp)

    # 3) Skull strip
    vol_s = apply_brain_mask(vol_r, brainm_r)
    vol_s = to_standard_axis(vol_s)  # 방향 정렬 

    # vol_s = center_crop_3d(vol_s, crop_shape=VOL_SHAPE)  # 필요 시 사용
    
    print("원본 HU 범위:", vol.min(), vol.max())
    print("Resampled 후 HU 범위:", vol_r.min(), vol_r.max())
    print("Skullstrip 후 HU 범위:", vol_s.min(), vol_s.max())
    print("Mask sum:", mask.sum(), "Brain mask sum:", brainm.sum())

    # 4) Window / normalize / enhancements → channels
    W_EXP    = cfg["window"]["experiments"]
    enh_cfg  = cfg.get("enhancements", {})
    clahe_en = enh_cfg.get("clahe", {}).get("enable", False)
    clip_lim = enh_cfg.get("clahe", {}).get("clip_limit", 0.03)
    tile_sz  = tuple(enh_cfg.get("clahe", {}).get("tile_grid_size", [8,8]))
    gamma_en = enh_cfg.get("gamma", {}).get("enable", False)
    gamma_vs = enh_cfg.get("gamma", {}).get("values", [1.0])

    volume_channels = []
    for clip_min, clip_max in W_EXP:
        # window + normalize
        level = (clip_min + clip_max) / 2
        width = clip_max - clip_min
        win  = apply_window(vol_s, level=level, width=width)
        norm = normalize_volume(win, clip_min=clip_min, clip_max=clip_max)
        # CLAHE
        if clahe_en:
            norm = apply_clahe(norm, clip_limit=clip_lim, tile_grid_size=tile_sz)
        # Gamma
        if gamma_en:
            for γ in gamma_vs:
                volume_channels.append(apply_gamma(norm, gamma=γ))
        else:
            volume_channels.append(norm)
            
    for i, ch in enumerate(volume_channels):
        print(f"volume_channels[{i}] mean/std: {ch.mean():.6f} {ch.std():.6f}")

    # 5) Pad/crop to volume shape
    VOL_SHAPE = tuple(cfg["shape"]["volume"])
    processed_vols = [
        pad_or_crop_3d(chan, target_shape=VOL_SHAPE)
        for chan in volume_channels
    ]
    vol_all  = np.stack(processed_vols, axis=0)
    mask_r = to_standard_axis(mask_r)  # 방향 보정
    mask_all = pad_or_crop_3d(mask_r, target_shape=VOL_SHAPE)

    # 6) Projections
    SLICE_SHAPE = tuple(cfg["shape"]["slice"])
    AXES    = cfg["projections"]["axes"]
    METHODS = cfg["projections"]["methods"]
    projs = []
    for axis in AXES:
        for method in METHODS:
            proj = PROJ_FNS[method](vol_s, axis=axis)
            # ensure (H,W)
            h, w = proj.shape
            if h < w:
                proj = proj.T
            # window/normalize first experiment
            clip0 = W_EXP[0]
            level = (clip0[0]+clip0[1])/2
            width = clip0[1]-clip0[0]
            proj = normalize_volume(
                       apply_window(proj, level=level, width=width),
                       clip_min=clip0[0], clip_max=clip0[1]
                   )
            proj = pad_or_crop_3d(proj[np.newaxis], target_shape=(1,)+SLICE_SHAPE).squeeze(0)
            projs.append(proj)
    projs = np.stack(projs, axis=0)

    # 7) Save .pt
    torch.save({
        "volume":      torch.tensor(vol_all, dtype=torch.float32),
        "mask":        torch.tensor(mask_all, dtype=torch.float32).unsqueeze(0),
        "projections": torch.tensor(projs, dtype=torch.float32),
        "meta": {
            "id":        case_id,
            "spacing":   tgt_sp,
            "win_exp":   W_EXP,
            "axes":      AXES,
            "methods":   METHODS,
            "clahe": {
                "enabled":        clahe_en,
                "clip_limit":     clip_lim,
                "tile_grid_size": tile_sz
            },
            "gamma": {
                "enabled": gamma_en,
                "values":  gamma_vs
            }
        }
    }, str(out_path))
    print(f"[✔] Preprocessed and saved: {case_id}.pt")


def visualize_case(case_id: str, cfg: dict):
    data_dir  = Path(os.environ.get("DATA_DIR", cfg["data_dir"]))
    pt_path   = data_dir / "processed" / f"{case_id}.pt"
    data      = torch.load(pt_path)
    vol_all   = data["volume"].numpy()    # (C,D,H,W)
    vol0      = vol_all[0]                # first channel
    mask_all  = data["mask"].numpy()[0]   # (D,H,W)
    W_EXP     = cfg["window"]["experiments"]

    # histogram
    if cfg["visualization"]["show_hist"]:
        bins = cfg["visualization"]["hist_bins"]
        exps = cfg["visualization"]["hist_exps"]
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,4))
        ax1.hist(vol0.ravel(), bins=bins, color="gray", alpha=0.7)
        ax1.set_title("Normalized Channel Histogram")
        for idx in exps:
            ch = vol_all[idx]
            ax2.hist(ch.ravel(), bins=bins, alpha=0.5, label=f"W_EXP[{idx}]")
        ax2.legend(), ax2.set_title("Multiple Window Experiments")
        plt.show()

    # slice compare
    if cfg["visualization"]["show_slice"]:
        axes   = cfg["visualization"]["slice_axes"]
        indices= cfg["visualization"]["slice_indices"] or [vol0.shape[a]//2 for a in axes]
        for axis, idx in zip(axes, indices):
            orig = np.take(vol0, idx, axis=axis)
            proc = np.take(vol0, idx, axis=axis)  # or other channel
            fig, axs = plt.subplots(1,2,figsize=(8,4))
            for ax,img,title in zip(axs, [orig,proc], ["Original","Preproc"]):
                ax.imshow(img, cmap="gray", aspect="equal", origin="lower")
                ax.set_title(f"{title} axis={axis}, idx={idx}")
                ax.axis("off")
            plt.tight_layout(), plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess and visualize CT cases."
    )
    # data_dir override
    parser.add_argument(
        "--data_dir",
        type=str,
        help="data 디렉토리 경로 (기본: configs/preprocessing.yaml의 data_dir)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--cases",
        nargs="+",
        help="처리할 case ID 리스트 (예: 049 052 101)"
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="data/raw 폴더에 있는 모든 케이스를 처리"
    )
    args = parser.parse_args()

    # 전역 config 로드
    cfg_path = Path(__file__).parents[1] / "configs" / "preprocessing.yaml"
    cfg = yaml.safe_load(open(cfg_path))

    # data_dir 결정: 커맨드라인 > config
    data_dir = Path(args.data_dir) if args.data_dir else Path(cfg["data_dir"])
    os.environ["DATA_DIR"] = str(data_dir)  # 이하 함수들이 이 env var 사용

    # case_list 결정
    if args.all:
        case_list = [p.stem for p in (data_dir/"raw").glob("*.nii*")]
    else:
        case_list = args.cases

    # 처리 루프
    for case_id in case_list:
        print(f"\n>>> Processing case {case_id}")
        preprocess_case(case_id, cfg)
        visualize_case(case_id, cfg)

if __name__ == "__main__":
    main()
