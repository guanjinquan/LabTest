# GPU Memory Used: 6G
python /mnt/home/Guanjq/BackupWork/LabTest/Code/main_train.py \
    --runs_id "003_resnet50_pathology" \
    --gpu_id "1" \
    --seed 109 \
    --weight_decay 6e-5 \
    --learning_rate 1e-6 \
    --backbone_lr 5e-7 \
    --acc_step 4 \
    --batch_size 4 \
    --split_filename "split_final_seed=2024.json" \
    --datainfo_file "pathology_info.json" \
    --img_size 512 \
    --num_epochs 200 \
    --model "resnet50_pathology"
