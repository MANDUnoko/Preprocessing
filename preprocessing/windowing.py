import numpy as np
from skimage import exposure

# ========================================
# HU 기반 윈도우잉 적용 (값 클리핑)
# ========================================
def apply_window(volume, level=40, width=80):
    """
    HU 볼륨에 Window Level (WL)과 Window Width (WW)를 적용

    Parameters:
        volume (np.ndarray): 3D 볼륨 배열 (D, H, W) 또는 2D 슬라이스
        level (float): Window Level (중심 HU 값), 기본 40
        width (float): Window Width (폭), 기본 80

    Returns:
        np.ndarray: WL–WW 범위로 클리핑된 볼륨
    """
    # 실험용으로 level/width 조정 가능
    min_val = level - (width / 2)
    max_val = level + (width / 2)
    return np.clip(volume, min_val, max_val)


# ========================================
# [0~1] 정규화
# ========================================
def normalize_volume(volume, clip_min, clip_max):
    """
    클리핑된 볼륨을 0~1 범위로 정규화

    Parameters:
        volume (np.ndarray): apply_window 결과
        clip_min (float): 윈도우 최소값 (apply_window 시 계산값)
        clip_max (float): 윈도우 최대값 (apply_window 시 계산값)

    Returns:
        np.ndarray: 0~1 스케일로 변환된 볼륨
    """
    # 작은 epsilon 추가로 0 나누기 방지
    eps = 1e-8
    norm = (volume - clip_min) / (clip_max - clip_min + eps)
    # overflow/underflow 잘라주기
    return np.clip(norm, 0.0, 1.0)

# ========================================
# CLAHE
# ========================================
def apply_clahe(volume: np.ndarray,
                clip_limit: float = 0.03,
                tile_grid_size: tuple[int,int] = (8, 8)
               ) -> np.ndarray:
    """
    CLAHE를 2D slice 단위로 적용
    - volume: 정규화된 [0,1] float32 3D array (D,H,W)
    - clip_limit: 히스토그램 균등화 클립 리밋 (0~1 사이 추천)
    - tile_grid_size: 타일 크기 (h, w)
    """
    out = np.zeros_like(volume, dtype=np.float32)
    for z in range(volume.shape[0]):
        slice_ = (volume[z] * 255).astype(np.uint8)
        # skimage.exposure.equalize_adapthist 반환은 [0,1]
        slice_eq = exposure.equalize_adapthist(
                        slice_,
                        clip_limit=clip_limit,
                        kernel_size=tile_grid_size
                   )
        out[z] = slice_eq
    return out

# ========================================
# Gamma Correction
# ========================================
def apply_gamma(volume: np.ndarray,
                gamma: float = 0.8
               ) -> np.ndarray:
    """
    Gamma correction을 적용
    - volume: 정규화된 [0,1] float32 3D array (D,H,W)
    - gamma: 교정 지수. 1보다 작으면 어두운 부위를 강조
    """
    # 작은 값 차이를 잃지 않도록 epsilon 추가
    return np.power(volume + 1e-8, gamma)



