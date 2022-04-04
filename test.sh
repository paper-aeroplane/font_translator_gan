set -ex
python test.py --dataroot ./datasets/font  --model font_translator_gan  --eval --name test_new_dataset --no_dropout --gpu_ids 0 --phase test_unknown_style --direction english2russian
