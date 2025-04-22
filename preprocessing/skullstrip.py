import numpy as np
from scipy.ndimage import zoom

# ========================================
# 1️⃣ 뇌 마스크 적용 (skull stripping)
# ========================================
def apply_brain_mask(volume, brain_mask):
    """
    볼륨에 뇌 마스크를 곱해 skull 등 non-brain 영역을 제거

    Parameters:
        volume (np.ndarray): 3D 볼륨 배열 (D, H, W)
        brain_mask (np.ndarray): 동일한 shape의 binary 마스크 (0 or 1)

    Returns:
        np.ndarray: skull-stripped 볼륨
    """
    assert volume.shape == brain_mask.shape, "volume과 brain_mask의 shape이 일치해야 합니다."
    return volume * brain_mask


# ========================================
# 2️⃣ 3D 바이너리 마스크 리샘플 (order=0)
# ========================================
def resample_mask(mask, original_spacing, target_spacing=(1.0, 1.0, 1.0)):
    """
    3D 바이너리 마스크를 nearest-neighbor로 리샘플링

    Parameters:
        mask (np.ndarray): 입력 binary 마스크 (0/1), shape (D,H,W)
        original_spacing (tuple): 원본 spacing (z,y,x)
        target_spacing (tuple): 목표 spacing (z,y,x)

    Returns:
        np.ndarray: 리샘플된 binary 마스크
    """
    # ① scale factor 계산
    factors = [o / t for o, t in zip(original_spacing, target_spacing)]
    # ② nearest neighbor 보간으로 리샘플
    res = zoom(mask, factors, order=0)
    # ③ float → binary
    return (res >= 0.5).astype(np.uint8)
