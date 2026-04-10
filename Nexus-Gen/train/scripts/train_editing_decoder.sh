# download flux
python train/decoder/download_flux.py
# download qwen2.5vl
modelscope download --model Qwen/Qwen2.5-VL-7B-Instruct --local_dir models/Qwen/Qwen2.5-VL-7B-Instruct
# train editing decoder
PYTHONPATH=$(pwd) \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python train/decoder/editing_trainer.py --config train/configs/editing_decoder.yaml
