CUDA_VISIBLE_DEVICES=0 python main.py manipulate --model-name DC_Disentanglement --factor-name ethnicity --img input.png --output face_in_all_ethnicity.png
CUDA_VISIBLE_DEVICES=0 python main.py manipulate --model-name DC_Disentanglement --factor-name hair_color --img input.png --output face_in_all_hair_color.png
CUDA_VISIBLE_DEVICES=0 python main.py manipulate --model-name DC_Disentanglement --factor-name beard --img input.png --output face_in_all_beard.png
CUDA_VISIBLE_DEVICES=0 python main.py manipulate --model-name DC_Disentanglement --factor-name glasses --img input.png --output face_in_all_glasses.png
CUDA_VISIBLE_DEVICES=0 python main.py manipulate --model-name DC_Disentanglement --factor-name age --img input.png --output face_in_all_age.png
CUDA_VISIBLE_DEVICES=0 python main.py manipulate --model-name DC_Disentanglement --factor-name gender --img input.png --output face_in_all_gender.png
