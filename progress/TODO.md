1. Check on progress of Groger et al README script
2. Implement Null-calibrated CKA/MKNN on PRH code.
3. Values that I need to compute:
	1. Pairwise model alignment metrics
		1. Model lists:
			1. Language: ["bigscience/bloomz-560m", "bigscience/bloomz-1b1", "bigscience/bloomz-1b7", "bigscience/bloomz-3b", "bigscience/bloomz-7b1", "openlm-research/open_llama_3b", "openlm-research/open_llama_7b", "openlm-research/open_llama_13b", "huggyllama/llama-7b","huggyllama/llama-13b",]
				1. Bloomz: 560M, 1B, 1B7, 3B, 7B1 (5 models)
				2. Open llama: 3b, 7b, 13b (3 models)
				3. Huggy llama: 7b, 13b (2 models)
				4. Total: 20 models
			2. Vision: ["vit_tiny_patch16_224.augreg_in21k", "vit_small_patch16_224.augreg_in21k", "vit_base_patch16_224.augreg_in21k", "vit_large_patch16_224.augreg_in21k", "vit_base_patch16_224.mae", "vit_large_patch16_224.mae", "vit_huge_patch14_224.mae", "vit_small_patch14_dinov2.lvd142m", "vit_base_patch14_dinov2.lvd142m", "vit_large_patch14_dinov2.lvd142m", "vit_giant_patch14_dinov2.lvd142m", "vit_base_patch16_clip_224.laion2b", "vit_large_patch14_clip_224.laion2b", "vit_huge_patch14_clip_224.laion2b", "vit_base_patch16_clip_224.laion2b_ft_in12k", "vit_large_patch14_clip_224.laion2b_ft_in12k", "vit_huge_patch14_clip_224.laion2b_ft_in12k",]
				1. 21K: Tiny, small, base, large (4 ViT)
				2. MAE: Base, Large, Huge (3 ViT)
				3. DinoV2: Small, Base, Large, Giant (4 ViT)
				4. Laion2B: Base, Large, Huge (3 ViT)
				5. Laion2B Ft In 12K: Base, Large, Huge (3 ViT)
				6. Total: 17 tasks
		2. Visual model: VTAB tasks
		3. Language model performance on Wikipedia caption dataset
		4. Language model performance on Hellaswag downstream
		5. Alignment metrics: ["cycle_knn", "mutual_knn", "lcs_knn", "cka", "unbiased_cka", "cknna", "svcca", "edit_distance_knn", "null_calibrated_cka", "null_calibrated_mknn"]