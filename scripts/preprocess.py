import os
import yaml
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

# 모듈 임포트
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

# window 설정 (WL, WW 값은 중앙/폭으로 계산)
W_EXP    = cfg["window"]["experiments"]   # e.g. [[0,80],[0,200]]

# shapes
VOL_SHAPE   = tuple(cfg["shape"]["volume"])  # (D,H,W)
SLICE_SHAPE = tuple(cfg["shape"]["slice"])   # (H,W)

# projections
AXES    = cfg["projections"]["axes"]      # [0,1,2]
METHODS = cfg["projections"]["methods"]   # ["mip","aip","mid"]

# 시각화 설정 (configs/preprocessing.yaml 내 visualization 블록)
viz_cfg       = cfg.get("visualization", {})
SHOW_PROJ     = viz_cfg.get("show_proj", False)
PROJ_AXES     = viz_cfg.get("proj_axes", AXES)
PROJ_METHODS  = viz_cfg.get("proj_methods", METHODS)
SHOW_HIST     = viz_cfg.get("show_hist", False)
HIST_BINS     = viz_cfg.get("hist_bins", 100)
HIST_EXPS     = viz_cfg.get("hist_exps", [0])
EQUALIZE_HIST = viz_cfg.get("equalize_hist", False)
SHOW_SLICE    = viz_cfg.get("show_slice", False)
SLICE_AXES    = viz_cfg.get("slice_axes", AXES)
SLICE_INDICES = viz_cfg.get("slice_indices", None)

# 분석 설정 (configs/preprocessing.yaml 내 analysis 블록)
analysis_cfg           = cfg.get("analysis", {})
COMPUTE_ROI            = analysis_cfg.get("compute_roi_metrics", False)
ROI_MASK_SOURCE        = analysis_cfg.get("roi_mask_source", "mask")
BACKGROUND_MASK_SOURCE = analysis_cfg.get("background_mask_source", "invert")
NOISE_REGION           = analysis_cfg.get("noise_region", "volume")

# ============================================
# 1. 케이스 순회
# ============================================
case_ids = ["049"]  # 테스트용
for case_id in case_ids:
    print(f"\n=== Processing case {case_id} ===")
    # 경로 설정
    img_path    = RAW_DIR        / f"{case_id}{RAW_EXT}"
    brainm_path = DATA_DIR       / "brain_masks" / f"{case_id}{BRAIN_MASK_EXT}"
    mask_path   = MASK_DIR       / f"{case_id}{LESION_EXT}"
    out_path    = OUT_DIR        / f"{case_id}.pt"
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
    vol_r    = resample_volume(vol,    original_spacing=orig_sp, target_spacing=tgt_sp, order=1)
    mask_r   = resample_mask(mask,     original_spacing=orig_sp, target_spacing=tgt_sp)
    brainm_r = resample_mask(brainm,   original_spacing=orig_sp, target_spacing=tgt_sp)

    # ============================================
    # 4. Skull strip & Center Crop (optional)
    # ============================================
    vol_s = apply_brain_mask(vol_r, brainm_r)
    # vol_s = center_crop_3d(vol_s, crop_shape=VOL_SHAPE)  # 필요 시 사용

    # ============================================
    # 5. Window & Normalize (여러 범위 실험)
    # ============================================
    volume_channels = []  # (C, D, H, W)
    for clip_min, clip_max in W_EXP:
        # apply_window 함수가 level, width 필요
        level = (clip_min + clip_max) / 2
        width = (clip_max - clip_min)
        windowed = apply_window(vol_s, level=level, width=width)
        norm = normalize_volume(windowed, clip_min=clip_min, clip_max=clip_max)
        volume_channels.append(norm)

    # ============================================
    # 6. Pad / Crop to target volume shape
    # ============================================
    processed_vols = []
    for norm in volume_channels:
        aligned = pad_or_crop_3d(to_standard_axis(norm), target_shape=VOL_SHAPE)
        processed_vols.append(aligned)
    vol_all = np.stack(processed_vols, axis=0)

    # 라벨도 같은 방식으로
    mask_all = pad_or_crop_3d(to_standard_axis(mask_r), target_shape=VOL_SHAPE)

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
            # 동일한 window & normalize 첫 번째 실험 설정 사용
            level = (W_EXP[0][0] + W_EXP[0][1]) / 2
            width = (W_EXP[0][1] - W_EXP[0][0])
            proj = apply_window(proj, level=level, width=width)
            proj = normalize_volume(proj, clip_min=W_EXP[0][0], clip_max=W_EXP[0][1])
            proj = pad_or_crop_3d(proj[np.newaxis, ...], target_shape=(1,) + SLICE_SHAPE).squeeze(0)
            proj_list.append(proj)
    projs = np.stack(proj_list, axis=0)
    
    # --- 7.5 원본 vs 전처리 프로젝션 비교 ---
    if SHOW_PROJ:
        for axis in PROJ_AXES:
            for method in PROJ_METHODS:
                # 1) 원본 프로젝션 (리샘플된 vol_r 사용)
                orig_proj = {
                    "mip": lambda v,a: mip_projection(v, a),
                    "aip": lambda v,a: aip_projection(v, a),
                    "mid": lambda v,a: mid_plane(v, a)
                }[method](vol_r, axis)
                # 2) 전처리 프로젝션 (skull-strip 후 vol_s 사용)
                proc_proj = {
                    "mip": lambda v,a: mip_projection(v, a),
                    "aip": lambda v,a: aip_projection(v, a),
                    "mid": lambda v,a: mid_plane(v, a)
                }[method](vol_s, axis)
                # 3) 시각화
                fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                axs[0].imshow(orig_proj, cmap='gray')
                axs[0].set_title(f"ORIG {method.upper()} axis={axis}")
                axs[1].imshow(proc_proj, cmap='gray')
                axs[1].set_title(f"PROC {method.upper()} axis={axis}")
                plt.tight_layout()
                plt.show()

    # --- 7.6 히스토그램 분석 섹션 ---
    if SHOW_HIST:
        # 1) 원본 볼륨 히스토그램
        plt.figure()
        plt.hist(vol_r.ravel(), bins=HIST_BINS, alpha=0.5, label="Original")
        # 2) 전처리 채널별 히스토그램
        for idx in HIST_EXPS:
            ch = volume_channels[idx]
            plt.hist(ch.ravel(), bins=HIST_BINS, alpha=0.5,
                     label=f"Preproc W_EXP[{idx}]")
            if EQUALIZE_HIST:
                from skimage.exposure import equalize_hist
                eq = equalize_hist(ch)
                plt.hist(eq.ravel(), bins=HIST_BINS, alpha=0.3,
                         label=f"Equalized[{idx}]")
        plt.legend()
        plt.title(f"Case {case_id} Histogram")
        plt.show()
        
    # --- 7.7 ROI 기반 SNR/CNR 측정 섹션 ---
    if COMPUTE_ROI:
        # 1) ROI 마스크 로드
        #    roi_mask_source: "mask" 이면 위에서 load_nifti_as_array로 읽은 mask_r 사용
        #                   "brain_mask" 이면 brainm_r 사용
        #                   그 외 문자열이면 Path(cfg["analysis"]["roi_mask_source"]) 경로에서 nibabel 로드
        if ROI_MASK_SOURCE == "mask":
            roi_mask = mask_r > 0
        elif ROI_MASK_SOURCE == "brain_mask":
            roi_mask = brainm_r > 0
        else:
            roi_mask = nib.load(str(Path(ROI_MASK_SOURCE))).get_fdata() > 0

        # 2) 노이즈(배경) 마스크 생성
        #    background_mask_source: "invert" → roi_mask 반전, "outside_brain" → brain mask 반전,
        #                            그 외 문자열은 해당 파일 로드
        if BACKGROUND_MASK_SOURCE == "invert":
            bg_mask = ~roi_mask
        elif BACKGROUND_MASK_SOURCE == "outside_brain":
            bg_mask = nib.load(str(Path(cfg["analysis"]["roi_mask_source"]))).get_fdata() == 0
        else:
            bg_mask = nib.load(str(Path(BACKGROUND_MASK_SOURCE))).get_fdata() > 0

        # 3) SNR/CNR 계산
        #    mean_signal = 볼륨[roi_mask] 평균
        #    std_noise   = NOISE_REGION=="volume" → 전체 vol_r.std()
        #                  "background" → 볼륨[bg_mask].std()
        vol_flat = vol_s.flatten()  # skull-strip 후 볼륨
        mean_signal = vol_s[roi_mask].mean()
        if NOISE_REGION == "background":
            std_noise = vol_s[bg_mask].std()
        else:
            std_noise = vol_s.std()
        snr = mean_signal / (std_noise + 1e-6)

        mean_bg = vol_s[bg_mask].mean()
        cnr = abs(mean_signal - mean_bg) / (std_noise + 1e-6)

        # 4) 결과 출력
        print(f"[ROI Metrics] case={case_id}")
        print(f"  • Mean(signal): {mean_signal:.4f}")
        print(f"  • Std(noise):   {std_noise:.4f}  (region={NOISE_REGION})")
        print(f"  • SNR:          {snr:.4f}")
        print(f"  • CNR:          {cnr:.4f}")
     
    # --- 7.8 시각적 검증: 대표 슬라이스 비교 섹션 ---    
    if SHOW_SLICE:
       # 축별 슬라이스 인덱스 결정
       axes   = SLICE_AXES
       indices= SLICE_INDICES or [vol_s.shape[a]//2 for a in axes]
       for axis, idx in zip(axes, indices):
           # 원본 vs 전처리 슬라이스 추출
           orig_slice = np.take(vol_r,   idx, axis=axis)
           proc_slice = np.take(vol_s,   idx, axis=axis)
           fig, axs = plt.subplots(1,2,figsize=(8,4))
           axs[0].imshow(orig_slice, cmap='gray')
           axs[0].set_title(f"ORIG axis={axis}, idx={idx}")
           axs[1].imshow(proc_slice, cmap='gray')
           axs[1].set_title(f"PROC axis={axis}, idx={idx}")
           plt.tight_layout(); plt.show()      
        
    # ============================================
    # 8. .pt 저장
    # ============================================
    torch.save({
        "volume":      torch.tensor(vol_all, dtype=torch.float32),
        "mask":        torch.tensor(mask_all, dtype=torch.float32).unsqueeze(0),
        "projections": torch.tensor(projs, dtype=torch.float32),
        "meta": {
            "id":      case_id,
            "spacing": tgt_sp,
            "win_exp": W_EXP,
            "axes":    AXES,
            "methods": METHODS
        }
    }, str(out_path))

    print(f"[✔] Saved: {case_id}")