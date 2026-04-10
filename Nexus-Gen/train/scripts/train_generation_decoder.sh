# download flux
python train/decoder/download_flux.py
# train generation decoder
PYTHONPATH=$(pwd) \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python train/decoder/generation_trainer.py --config train/configs/generation_decoder.yaml
