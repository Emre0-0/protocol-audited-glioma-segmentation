# Publication training and inference recipe

- Generated: 2026-08-11T04:52:38.805974+00:00
- nnU-Net: `2.7.0`
- Dataset: `Dataset503_BraTS2024Grouped4Ch` (1155 development scans)
- Input channels: `{'0': 't1n', '1': 't1c', '2': 't2w', '3': 't2f'}`
- Labels: `{'background': 0, 'label_1': 1, 'label_2': 2, 'label_3': 3, 'label_4': 4}`
- Configuration: `3d_fullres`
- Patch/batch: `[128, 192, 128]`, batch size `2`
- Spacing: `[1.0, 1.0, 1.0]` mm
- Architecture: `dynamic_network_architectures.architectures.unet.ResidualEncoderUNet`
- Features/stage: `[32, 64, 128, 256, 320, 320]`
- Epochs: `250`; iterations/epoch: `250`

## Optimization

- Optimizer: SGD(momentum=0.99, Nesterov=True, weight_decay=3e-5).
- Initial learning rate: 0.01; scheduler: PolyLRScheduler over 250 epochs.
- Loss: equal-weight cross-entropy plus memory-efficient soft Dice (background excluded, smooth=1e-5), wrapped with normalized deep-supervision weights proportional to 1/2^i; lowest output weight 0.
- Foreground oversampling: 33%.
- Deep supervision: True.

## Augmentation

- Spatial rotation +/-30 degrees (apply p=0.2) and scaling 0.7--1.4 (p=0.2; synchronized across axes); elastic deformation disabled.
- Gaussian noise variance 0--0.1 (p=0.1; synchronized channels).
- Gaussian blur sigma 0.5--1.0 (p=0.2; per-channel p=0.5).
- Multiplicative brightness 0.75--1.25 (p=0.15).
- Contrast 0.75--1.25 with range preservation (p=0.15).
- Simulated low resolution scale 0.5--1.0 (p=0.25; per-channel p=0.5).
- Inverted gamma 0.7--1.5 with retained statistics (p=0.1).
- Gamma 0.7--1.5 with retained statistics (p=0.3).
- Independent mirroring over all three spatial axes.
- Mask-based normalization for all four MRI channels; outside-mask values set to zero.

## Full-volume inference

- Sliding-window step: 0.5.
- Gaussian weighting: True.
- Mirroring/TTA axes: [0, 1, 2].
- Checkpoint: checkpoint_best.pth per grouped fold.
- Official reporting uses vendored BraTS 2024 lesion-wise metrics, not nnU-Net's patch pseudo-Dice.

## Reproducibility boundary

Python/NumPy/PyTorch seeds were set per fold, but cuDNN benchmarking and TF32 were enabled and deterministic algorithms were not enforced. The run is recipe-reproducible, not guaranteed bitwise deterministic.
