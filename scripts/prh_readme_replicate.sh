cd platonic-rep

pip install -r requirements.txt


# extract all language model features and pool them along each block
# python extract_features.py --dataset minhuh/prh --subset wit_1024 --modelset val --modality language --pool avg --output_dir /workspace/results/emily/prh_features

# Extract last layer features of all vision models
python extract_features.py --dataset minhuh/prh --subset wit_1024 --modelset val --modality vision --pool cls --output_dir /workspace/results/emily/prh_features


# python measure_alignment.py --dataset minhuh/prh --subset wit_1024 --modelset val \
#         --modality_x language --pool_x avg --modality_y vision --pool_y cls --input_dir /workspace/results/emily/prh_features --output_dir /workspace/results/emily/prh_alignment
