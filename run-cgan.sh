wandb $1

python -m train_cgan fit --seed_everything 123 \
                         --data.file_path /mnt/disks/raw-data/tmp.h5 \
                         --data.batch_size 16 \
                         --trainer cgan-trainer.yaml \
                         --model.lambda_pixel 100
                        # --optimizer Adam \
                        # --optimizer.lr 1e-3 \
                        # --optimizer.weight_decay 0.00 \
                        # --trainer.max_epochs 100 \
                        # --trainer.accelerator auto \
                        # --trainer.devices -1
                        # --data.crop_size null \
                        # --trainer.devices -1 \
                        # --trainer.strategy auto
