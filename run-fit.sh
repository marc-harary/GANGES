wandb $1

python -m train fit --seed_everything 123 \
                    --model.in_channels 3 \
                    --model.hidden_channels 32 \
                    --model.out_channels 3 \
                    --model.num_blocks 8 \
                    --model.kernel_size 3 \
                    --model.stride 1 \
                    --model.scaling_factor 1.0 \
                    --model.dropout_rate 0.5 \
                    --model.skip_scaling_factor 0.0 \
                    --model.negative_slope 0.01 \
                    --optimizer Adam \
                    --optimizer.lr 1e-3 \
                    --optimizer.weight_decay 0.00 \
                    --data.file_path /mnt/disks/raw-data/he_ihc_dataset.hdf5 \
                    --data.thres 0.03623734579162963 \
                    --data.train_patch_size 256 \
                    --data.train_batch_size 8 \
                    --data.val_patch_size 256 \
                    --data.val_batch_size 1 \
                    --data.num_workers 11 \
                    --trainer fit-trainer.yaml  \
                    --trainer.accelerator cpu \
                    --trainer.devices 1
                    # --data.crop_size null \
                    # --trainer.devices -1 \
                    # --trainer.strategy auto
