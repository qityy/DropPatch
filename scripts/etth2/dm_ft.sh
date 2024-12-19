# shellcheck disable=SC2034
model="dm"
data="ETTh2"
root_path="./dataset/ETT/"
data_path="ETTh2.csv"
seq_len=512
patch_len=12
d_model=16
enc_layers=3
n_heads=4
d_ff=128
batch_size=16
i=1
drop_ratio=0.6
mask_ratio=0.4

pred_len=192

for lr in 1e-4 3e-5; do
setting="$model"_"$i"_"$data"_sl"$seq_len"_dr"$drop_ratio"_mr"$mask_ratio"_patchl"$patch_len"_dmodel"$d_model"_layers"$enc_layers"_heads"$n_heads"_ff"$d_ff"_bs64_lr1e-3
python run.py \
  --gpu=3 \
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
  --n_epochs_finetune=1 \
  --lr=$lr \
  --enc_layers=$enc_layers \
  --n_heads=$n_heads \
  --d_ff=$d_ff \
  --dropout=0.2 \
  --batch_size=$batch_size \
  --head_dropout=0.2 \
  --lradj="TST" \
  --mode=ft
done