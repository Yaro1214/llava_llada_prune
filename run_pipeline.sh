export LMMS_EVAL_PLUGINS=lmms_plugin
export PYTHONPATH=/home/ma-user/code/zyj/LLaDA-V/eval:/home/ma-user/code/zyj/LLaDA-V//train:$PYTHONPATH

# python evaluation_script.py --model llada_v --tasks ai2d --batch_size 1 \
#   --gen_kwargs='{"temperature":0,"cfg":0,"remasking":"low_confidence","gen_length":2,"block_length":2,"gen_steps":2,"think_mode":"think"}' \
#   --model_args pretrained=GSAI-ML/LLaDA-V,conv_template=llava_llada,model_name=llava_llada \
#   --output_path ./ai2d_log \
#   --log_samples \
#   --log_samples_suffix ai2d \
#   --device cuda:5 \
#   --limit 1

python evaluation_script.py --model llada_v --tasks ai2d --batch_size 1 \
  --gen_kwargs='{"temperature":0,"cfg":0,"remasking":"low_confidence","gen_length":2,"block_length":2,"gen_steps":2,"think_mode":"think"}' \
  --model_args "pretrained=GSAI-ML/LLaDA-V,conv_template=llava_llada,model_name=llava_llada,sparse=true,pruned_layer=2,image_token_start_index=47,image_token_length=1200,max_num_trunction=-1,reduction_ratio=0.5,pivot_image_token=4,pivot_text_token=4" \
  --output_path ./ai2d_log \
  --log_samples \
  --log_samples_suffix ai2d \
  --device cuda:5 \
  --limit 1