pip install -r platonic-rep/requirements.txt
pip install -r Aristotelian/requirements.txt

cd geometry_aware_convergence

METRICS=(cycle_knn mutual_knn lcs_knn cka unbiased_cka cknna svcca edit_distance_knn)

for metric in "${METRICS[@]}"; do
    python measure_alignment.py --modality_x all --modality_y all --metric "$metric"
done
