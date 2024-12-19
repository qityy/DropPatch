# shellcheck disable=SC2034
model="dm"
data="ECL"
root_path="./dataset/electricity/"
data_path="electricity.csv"
seq_len=512
patch_len=12
d_model=256
pred_len=96
enc_layers=4
n_heads=16
d_ff=256
batch_size=16
drop_ratio=0.75
mask_ratio=0.4
lr=1e-3
i=1

setting="$model"_"$i"_"$data"_sl"$seq_len"_dr"$drop_ratio"_mr"$mask_ratio"_patchl"$patch_len"_dmodel"$d_model"_layers"$enc_layers"_heads"$n_heads"_ff"$d_ff"_bs"$batch_size"_lr"$lr"

accelerate launch --multi_gpu --gpu_ids=0,1,2,3 --mixed_precision='no' --num_processes=4 --num_machines=1 --dynamo_backend='no' --main_process_port=29502 run_ddp.py \
  --model=$model \
  --data=$data \
  --root_path=$root_path \
  --data_path=$data_path \
  --seq_len=$seq_len \
  --patch_len=$patch_len \
  --d_model=$d_model \
  --pred_len=$pred_len \
  --drop_ratio=$drop_ratio \
  --mask_ratio=$mask_ratio \
  --pretrained_model_id="$i" \
  --setting=$setting \
  --n_epochs_pretrain=50 \
  --lr=$lr \
  --enc_layers=$enc_layers \
  --n_heads=$n_heads \
  --d_ff=$d_ff \
  --dropout=0.2 \
  --head_dropout=0.2 \
  --batch_size=$batch_size \
  --showcase=0 \
  --mode=pt

