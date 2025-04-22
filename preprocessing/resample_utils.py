import numpy as np
from scipy.ndimage import zoom

# ========================================
# 1️⃣ 3D 볼륨 리샘플링 (CT 이미지)
# ========================================
def resample_volume(volume, original_spacing, target_spacing=(1.0, 1.0, 1.0), order=1):
    """
    3D 이미지 볼륨을 isotropic spacing 또는 원하는 spacing으로 리샘플링합니다.

    Parameters:
        volume (np.ndarray): 입력 3D 배열, 형태 (D, H, W)
        original_spacing (tuple of float): 원본 voxel spacing (z, y, x) [mm]
        target_spacing (tuple of float): 목표 voxel spacing (z, y, x) [mm]
            └ 기본값: (1.0, 1.0, 1.0)  # isotropic
        order (int): 보간 차수
            - 0: nearest (binary mask용)
            - 1: linear  (CT intensity용, 기본)
            - 3: cubic   (더 부드러운 보간이 필요할 때)

    Returns:
        np.ndarray: 리샘플된 3D 배열
    """
    # ① 축별 scale factor 계산
    factors = [o / t for o, t in zip(original_spacing, target_spacing)]

    # ② scipy.ndimage.zoom로 보간
    #    order 파라미터를 바꿔서 mask/label에는 0, CT에는 1 이상 사용
    resampled = zoom(volume, factors, order=order)

    return resampled


# ========================================
# 2️⃣ 3D 바이너리 마스크 리샘플링
# ========================================
def resample_mask(mask, original_spacing, target_spacing=(1.0, 1.0, 1.0)):
    """
    3D 바이너리 마스크를 원하는 spacing으로 nearest-neighbor 보간하여 리샘플링합니다.

    Parameters:
        mask (np.ndarray): 입력 3D 배열, dtype 이진 (0 또는 1)
        original_spacing (tuple of float): 원본 voxel spacing (z, y, x)
        target_spacing (tuple of float): 목표 voxel spacing

    Returns:
        np.ndarray: 리샘플된 3D 바이너리 마스크 (0/1)
    """
    # ① 정수형이어도 zoom은 float 반환 → scale factor 계산
    factors = [o / t for o, t in zip(original_spacing, target_spacing)]

    # ② nearest neighbor 보간 (order=0)
    resampled = zoom(mask, factors, order=0)

    # ③ float → binary로 다시 변환
    return (resampled >= 0.5).astype(np.uint8)

