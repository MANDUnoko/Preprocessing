STROKESEGMENTATION/
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
│   └── preprocess.py            # 전체 3D 전처리 + projection 생성 → .pt 저장
└── notebooks/
    └── 02_preprocessing_compare.ipynb  # slice 실험용

실험 계획
1. Slice 실험 (notebook)
병변 slice만 골라서 2D 전처리 비교 → parameter 튜닝

2. 5D 프로토타입
원본 중앙슬라이스±1 + Z‑MIP 1 채널 등 5채널 입력, 2D U‑Net 실험

3. 3D 모델 학습
최종 .pt로 ResUNet3D / SwinUNETR
실험군: window [0–80] vs [0–200], skull strip vs center crop vs 둘 다

4. 평가 지표
Dice, IoU, HD, CNR, SNR, 학습 속도(epochs to converge)