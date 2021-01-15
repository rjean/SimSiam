#!/bin/bash
python main.py \
--dataset objectron \
--image_size 96 \
--aug objectron_video \
--model simsiam \
--proj_layers 2 \
--backbone resnet18 \
--download \
--optimizer sgd \
--weight_decay 0.0005 \
--momentum 0.9 \
--warmup_epoch 10 \
--warmup_lr 0 \
--base_lr 0.3 \
--final_lr 0 \
--num_epochs 50 \
--stop_at_epoch 50 \
--batch_size 512 \
--eval_after_train "--base_lr float(30)
                    --weight_decay float(0)
                    --momentum float(0.9)
                    --warmup_epochs int(0)
                    --batch_size int(256)
                    --num_epochs int(30)
                    --optimizer str('sgd')" \
--head_tail_accuracy \
--output_dir outputs/objectron_96x96_experiment_video/ \
--data_dir datasets/objectron_96x96/ \
--resume_from_last \
#--hide_progress 
# --debug












