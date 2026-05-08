python crownClustering_optimized_refactored.py \
  --input-mode multispectral \
  --data-root ~/Data/Talca2025/Registered/Darwin/ \
  --model-path ./data/pretrainedModels/FamNet_Save1.pth \
  --clustering-method random_walker \
  --rw-beta 600 \
  --output-dir ./output_multispectral_rw\
  --crop 720 1000 720 1000 --show --feature-reduction none

python crownClustering_optimized_refactored.py \
  --input-mode rgb \
  --rgb-image-path ~/Data/Arauco/Samples/MSanz/RGB/RGB_subset.tif \
  --seed-source manual_centers \
  --model-path ./data/pretrainedModels/FamNet_Save1.pth \
  --vegetation-mask-mode auto \
  --feature-reduction none \
  --clustering-method watershed \
  --fallback-to-manual-centers \
  --output-dir output_rgb_density \
  --crop 720 1000 720 1000 --show 


