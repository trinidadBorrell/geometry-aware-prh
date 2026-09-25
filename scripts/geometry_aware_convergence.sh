pip install -r platonic-rep/requirements.txt
pip install -r Aristotelian/requirements.txt

cd geometry_aware_convergence
# rm: lcs_knn svcca edit_distance_knn
# METRICS=(cycle_knn mutual_knn cka unbiased_cka cknna)

# for metric in "${METRICS[@]}"; do
#     python measure_alignment.py --modality_x all --modality_y all --metric "$metric" --null-calibrate
# done

# KVALS=(5 20 50 100)

# for k in "${KVALS[@]}"; do
#     # python measure_alignment.py --modality_x all --modality_y all --metric cknna --topk "$k"
#     python measure_alignment.py --modality_x all --modality_y all --metric cknna --null-calibrate --topk "$k"
# done


DISTS=(0.95 0.9 0.8 0.7 0.6)

for k in "${KVALS[@]}"; do
    python measure_alignment.py --modality_x all --modality_y all --metric cknda --dist "$dist"
    python measure_alignment.py --modality_x all --modality_y all --metric cknda --null-calibrate --topk "$dist"
done
