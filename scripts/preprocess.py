import os
import yaml
import torch
import numpy as np
import nibabel as nib
from pathlib import Path

# 1️⃣ 모듈 임포트
from preprocessing.utils          import load_nifti_as_array, get_spacing_from_affine
from preprocessing.resample_utils import resample_volume, resample_mask
from preprocessing.skullstrip     import apply_brain_mask
from preprocessing.windowing      import apply_window, normalize_volume
from preprocessing.shape_utils    import to_standard_axis, pad_or_crop_3d, center_crop_3d
from preprocessing.volume_utils   import mip_projection, aip_projection, mid_plane

# ============================================
# 0. 설정 로드 (configs/preprocessing.yaml)
# ============================================
cfg_path = Path(__file__).resolve().parents[1] / "configs" / "preprocessing.yaml"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

# 파일 확장자    
RAW_EXT        = ".nii.gz"   # 원본 CT
BRAIN_MASK_EXT = ".nii.gz"   # skull‑strip 마스크
LESION_EXT     = ".nii"      # 병변 마스크 (Label)

# case 리스트
cfg_data_dir = cfg.get("data_dir", "data")

DATA_DIR = Path(os.environ.get("DATA_DIR", cfg_data_dir))
RAW_DIR     = DATA_DIR / "raw"
MASK_DIR    = DATA_DIR / "masks"
OUT_DIR     = DATA_DIR / "processed"

# spacing
orig_sp = tuple(cfg["spacings"]["original"])
tgt_sp  = tuple(cfg["spacings"]["target"])

# window
WL       = cfg["window"]["level"]
WW       = cfg["window"]["width"]
W_EXP    = cfg["window"]["experiments"]   # e.g. [[0,80],[0,200]]

# shapes
VOL_SHAPE   = tuple(cfg["shape"]["volume"])  # (D,H,W)
SLICE_SHAPE = tuple(cfg["shape"]["slice"])   # (H,W)

# projections
AXES    = cfg["projections"]["axes"]      # [0,1,2]
METHODS = cfg["projections"]["methods"]   # ["mip","aip","mid"]

# ============================================
# 1. 케이스 순회
# ============================================
# 전체 폴더 처리
# case_ids = [p.stem for p in RAW_DIR.glob("*.nii*")]

# 하나의 파일 처리
case_ids = ["049"]
for case_id in case_ids:
    print(f"\n=== Processing case {case_id} ===")
    # 경로 설정
    img_path    = RAW_DIR        / f"{case_id}{RAW_EXT}"
    brainm_path = DATA_DIR       / "brain_masks" / f"{case_id}{BRAIN_MASK_EXT}"
    mask_path   = MASK_DIR       / f"{case_id}{LESION_EXT}"
    out_path   = OUT_DIR  / f"{case_id}.pt"
    os.makedirs(out_path.parent, exist_ok=True)

    # ============================================
    # 2. Load & RAS 정렬
    # ============================================
    vol, affine = load_nifti_as_array(str(img_path), reorient=True)
    mask, _     = load_nifti_as_array(str(mask_path), reorient=True)
    brainm, _   = load_nifti_as_array(str(brainm_path), reorient=True)

    # ============================================
    # 3. Spacing 추출 & 3D Resample
    # ============================================
    spacing = get_spacing_from_affine(affine)
    vol_r   = resample_volume(vol,    original_spacing=orig_sp, target_spacing=tgt_sp, order=1)
    mask_r  = resample_mask(  mask,   original_spacing=orig_sp, target_spacing=tgt_sp)
    brainm_r= resample_mask(brainm,   original_spacing=orig_sp, target_spacing=tgt_sp)

    # ============================================
    # 4. Skull strip & Center Crop (optional)
    # ============================================
    vol_s = apply_brain_mask(vol_r, brainm_r)
    # vol_s = center_crop_3d(vol_s, crop_shape=VOL_SHAPE)  # 실험용

    # ============================================
    # 5. Window & Normalize (여러 범위 실험)
    # ============================================
    # 최종 볼륨 채널 저장 배열 (C, D, H, W)
    volume_channels = []
    for clip_min, clip_max in W_EXP:
        # ... window & normalize ...
        volume_channels.append(norm)

    # ============================================
    # 6. Pad / Crop to target volume shape
    # ============================================
    processed_vols = []
    for norm in volume_channels:
        aligned = pad_or_crop_3d(
            to_standard_axis(norm),
            target_shape=VOL_SHAPE
        )
        processed_vols.append(aligned)
    # 이제야 shape=(len(W_EXP), D', H', W')
    vol_all = np.stack(processed_vols, axis=0)

    # 라벨도 같은 방식으로
    mask_all = pad_or_crop_3d(
        to_standard_axis(mask_r),
        target_shape=VOL_SHAPE
    )

    # ============================================
    # 7. Projection 생성 (2D 채널)
    # ============================================
    proj_list = []
    for axis in AXES:
        for method in METHODS:
            if method == "mip":
                proj = mip_projection(vol_s, axis=axis)
            elif method == "aip":
                proj = aip_projection(vol_s, axis=axis)
            elif method == "mid":
                proj = mid_plane(vol_s, axis=axis)
            # 윈도우 & 정규화 (첫번째 exp) + pad2d
            proj = apply_window(proj,  level=WL, width=WW)
            proj = normalize_volume(proj, clip_min=WL-WW/2, clip_max=WL+WW/2)
            proj = pad_or_crop_3d(proj, target_shape=(1,)+SLICE_SHAPE).squeeze(0)
            proj_list.append(proj)
    projs = np.stack(proj_list, axis=0)  # shape = (N_proj, H, W)

    # ============================================
    # 8. .pt 저장
    # ============================================
    torch.save({
        "volume":      torch.tensor(vol_all, dtype=torch.float32),
        "mask":        torch.tensor(mask_all, dtype=torch.float32).unsqueeze(0),
        "projections": torch.tensor(projs, dtype=torch.float32),
        "meta": {
            "id":        case_id,
            "spacing":   tgt_sp,
            "win_exp":   W_EXP,
            "axes":      AXES,
            "methods":   METHODS
        }
    }, str(out_path))

    print(f"[✔] Saved: {case_id}")
