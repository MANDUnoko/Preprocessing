import numpy as np
import nibabel as nib

# ========================================
# 1️⃣ NIfTI 로드 & 방향 정렬
# ========================================
def load_nifti_as_array(path, reorient=True):
    """
    NIfTI 파일을 로드하여 NumPy 배열과 affine을 반환.
    - reorient=True: nibabel.as_closest_canonical로 RAS 방향으로 재정렬.

    Parameters:
        path (str): NIfTI 파일 경로
        reorient (bool): 방향 정렬 여부 (기본값: True)

    Returns:
        tuple:
            data (np.ndarray): 3D 볼륨 데이터, dtype=float32
            affine (np.ndarray): 4×4 affine 행렬
    """
    img = nib.load(path)
    if reorient:
        img = nib.as_closest_canonical(img)
    data = img.get_fdata().astype(np.float32)
    return data, img.affine


# ========================================
# 2️⃣ affine에서 voxel spacing 추출
# ========================================
def get_spacing_from_affine(affine):
    """
    affine 행렬에서 Z, Y, X 축의 voxel 크기를 추출.

    Parameters:
        affine (np.ndarray): 4×4 affine 행렬

    Returns:
        tuple of float: (spacing_z, spacing_y, spacing_x)
    """
    # affine[2,2], [1,1], [0,0]이 각각 Z, Y, X spacing
    spacing = np.abs([affine[2, 2], affine[1, 1], affine[0, 0]])
    return tuple(spacing)


# ========================================
# (Optional) DICOM → HU 변환
# ========================================
def dicom_to_hu(ds):
    """
    pydicom Dataset에서 pixel_array를 HU 단위로 변환.

    Parameters:
        ds: pydicom Dataset 객체 (RescaleSlope, RescaleIntercept 포함)

    Returns:
        np.ndarray: HU 변환된 이미지 배열 (float32)
    """
    image = ds.pixel_array.astype(np.float32)
    slope = float(ds.get('RescaleSlope', 1.0))
    intercept = float(ds.get('RescaleIntercept', 0.0))
    return image * slope + intercept