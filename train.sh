set -ex
python train.py --dataroot ./datasets/font --model font_translator_gan --name test_new_dataset --no_dropout --gpu_ids 0 --direction english2russian --batch_size 64 --print_freq 128
