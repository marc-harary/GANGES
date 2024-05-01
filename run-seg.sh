wandb $1

python -m train_seg fit --seed_everything 123 \
                        --data.hdf5_path /mnt/disks/disk2/nanog-1000.h5 \
                        --data.batch_size 16 \
                        --trainer seg-trainer.yaml  \
                        --optimizer Adam \
                        --optimizer.lr 1e-3 \
                        --optimizer.weight_decay 0.00 \
                        --trainer.max_epochs 100 \
                        --trainer.accelerator auto \
                        --trainer.devices -1
                        # --data.crop_size null \
                        # --trainer.devices -1 \
                        # --trainer.strategy auto
