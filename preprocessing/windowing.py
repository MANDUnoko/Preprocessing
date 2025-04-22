import numpy as np

# ========================================
# 1️⃣ HU 기반 윈도우잉 적용 (값 클리핑)
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
# 2️⃣ [0~1] 정규화
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
