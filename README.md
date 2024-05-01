# IHC to H&E GAN
Modules are as follows:
- `train_cgan`: map segmentations to IHCs
- `train_gan`: trains CycleGAN to map H\&Es directly to IHCs
- `train_seg`: labels binary nuclei masks as being antigen positive or negative
