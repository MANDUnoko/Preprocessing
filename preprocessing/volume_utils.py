import numpy as np

# ========================================
# 1️⃣ MIP (Maximum Intensity Projection)
# ========================================
def mip_projection(volume, axis=0):
    """
    주어진 3D 볼륨을 axis 방향으로 최대강도영상을 생성
    
    Parameters:
        volume (np.ndarray): 3D 볼륨 배열, shape (D, H, W)
        axis (int): 투영 축 (0=Z, 1=Y, 2=X), 기본값 0 (axial)

    Returns:
        np.ndarray: 2D MIP 이미지, shape(volume.shape[axes_except_axis])
    """
    # 💬 실험용으로 다른 축 투영해볼 수 있음
    return np.max(volume, axis=axis)


# ========================================
# 2️⃣ AIP (Average Intensity Projection)
# ========================================
def aip_projection(volume, axis=0):
    """
    주어진 3D 볼륨을 axis 방향으로 평균강도영상을 생성
    
    Parameters:
        volume (np.ndarray): 3D 볼륨 배열
        axis (int): 투영 축 (0=Z, 1=Y, 2=X), 기본값 0 (axial)

    Returns:
        np.ndarray: 2D AIP 이미지
    """
    # 💬 윈도우 적용 전/후 비교해볼 수도 있음
    return np.mean(volume, axis=axis)


# ========================================
# 3️⃣ Mid‑plane Projection
# ========================================
def mid_plane(volume, axis=0):
    """
    주어진 3D 볼륨에서 axis 방향의 중간 슬라이스를 추출
    
    Parameters:
        volume (np.ndarray): 3D 볼륨 배열
        axis (int): 추출 축 (0=Z, 1=Y, 2=X), 기본값 0 (axial)

    Returns:
        np.ndarray: 2D 중간 평면 이미지
    """
    idx = volume.shape[axis] // 2
    # np.take를 사용해 지정 axis의 중간 슬라이스 반환
    return np.take(volume, indices=idx, axis=axis)
