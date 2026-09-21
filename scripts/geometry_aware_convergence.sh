pip install -r platonic-rep/requirements.txt
pip install -r Aristotelian/requirements.txt

cd geometry_aware_convergence
# rm: lcs_knn svcca edit_distance_knn
METRICS=(cycle_knn mutual_knn cka unbiased_cka cknna)

for metric in "${METRICS[@]}"; do
    python measure_alignment.py --modality_x all --modality_y all --metric "$metric" --null-calibrate
done
