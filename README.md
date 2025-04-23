```
PREPROCESSING/
├── configs/
│   └── preprocessing.yaml       # 파라미터 일괄 관리
│
├── preprocessing/
│   ├── utils.py                 # reorient, HU utils
│   ├── resample_utils.py        # resample_volume, resample_mask
│   ├── windowing.py             # apply_window, normalize_volume
│   ├── skullstrip.py            # apply_brain_mask
│   ├── shape_utils.py           # pad_or_crop_safe
│   └── volume_utils.py          # mip_projection, aip_projection, mid_plane
│
├── scripts/
│   └── preprocess.py            # 전처리 + projection 생성 → .pt 저장
└── notebooks/
    └── preprocessing.ipynb      # 코랩기반 실험용
```
전체 3D 볼륨을 기반으로 한 SNR/CNR 지표는 충분히 괜찮게 나옴

다만 데이터 불균형(lesion 픽셀이 매우 적음)을 완화하기 위해
학습 시 patch sampling을 병변 위주로 조절하거나
3D augment(회전·스케일·좌우반전)를 적용
모델 입력으로 다채널 윈도우(+CLAHE, Gamma) 조합을 시험 등...

그리고 nii -> 전처리 -> nii
에서 post nii 를 뷰어로 보면 모양이 찌그러지는 문제 발생
어케 해결해야할지 모르겠음.. 패딩 문제인지
