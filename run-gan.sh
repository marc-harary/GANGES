wandb $1

python -m train_gan fit --seed_everything 123 \
                        --model.input_channels 3 \
                        --model.output_channels 32 \
                        --model.lr=2e-4 \
                        --data.file_path files.lst \
                        --data.thres 0.03623734579162963 \
                        --data.train_patch_size 256 \
                        --data.train_batch_size 12 \
                        --data.val_patch_size 256 \
                        --data.val_batch_size 8 \
                        --data.epoch_length 1000 \
                        --data.num_workers 11 \
                        --trainer fit-trainer.yaml  \
                        --trainer.max_epochs 100 \
                        --trainer.accelerator auto \
                        --trainer.devices 1
                        # --data.crop_size null \
                        # --trainer.devices -1 \
                        # --trainer.strategy auto
