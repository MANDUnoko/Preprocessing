import numpy as np

# ========================================
# 1️⃣ 모든 볼륨 축 순서 통일 함수
# ========================================
def to_standard_axis(volume):
    """
    volume의 axis 순서를 (Z, Y, X) 형태로 맞춥니다.
    - 예: 원본이 (X,Y,Z)일 때 (2,1,0) transpose 적용

    Parameters:
        volume (np.ndarray): 3D 배열 (어떤 순서든)

    Returns:
        np.ndarray: (Z, Y, X) 순서로 정렬된 3D 배열
    """
    # 만약 마지막 축 크기가 가장 작다면 (예: Z가 마지막), transpose 적용
    if volume.shape[2] < volume.shape[0]:
        volume = np.transpose(volume, (2, 1, 0))
    return volume


# ========================================
# 2️⃣ 3D 볼륨 pad or crop to target_shape
# ========================================
def pad_or_crop_3d(volume, target_shape=(140, 250, 250), mode='constant'):
    """
    3D 볼륨을 중앙 기준으로 pad 또는 crop하여 원하는 shape으로 만듭니다.

    Parameters:
        volume (np.ndarray): 입력 3D 배열 (D, H, W)
        target_shape (tuple of int): 원하는 출력 shape (D, H, W)
        mode (str): padding 모드 (np.pad의 mode 인자, 기본 'constant')

    Returns:
        np.ndarray: target_shape 크기의 3D 배열
    """
    vol = volume.copy()
    for axis in range(3):
        src, dst = vol.shape[axis], target_shape[axis]
        if src < dst:
            pad_before = (dst - src) // 2
            pad_after = dst - src - pad_before
            pad_cfg = [(0,0)]*3
            pad_cfg[axis] = (pad_before, pad_after)
            vol = np.pad(vol, pad_cfg, mode=mode)
        elif src > dst:
            crop_before = (src - dst) // 2
            slicer = [slice(None)]*3
            slicer[axis] = slice(crop_before, crop_before + dst)
            vol = vol[tuple(slicer)]
    return vol


# ========================================
# 3️⃣ 3D 중앙 Cropping 함수 (mask-based center crop)
# ========================================
def center_crop_3d(volume, crop_shape=(140, 250, 250)):
    """
    뇌 중심이나 마스크 중심을 기준으로 3D 중앙 Cropping을 수행합니다.

    Parameters:
        volume (np.ndarray): 입력 3D 배열 (D, H, W)
        crop_shape (tuple of int): 자를 영역 크기 (D, H, W)

    Returns:
        np.ndarray: crop_shape 크기의 중앙 Cropped 배열
    """
    d, h, w = volume.shape
    cd, ch, cw = crop_shape

    start_d = max((d - cd)//2, 0)
    start_h = max((h - ch)//2, 0)
    start_w = max((w - cw)//2, 0)

    return volume[
        start_d:start_d+cd,
        start_h:start_h+ch,
        start_w:start_w+cw
    ]
