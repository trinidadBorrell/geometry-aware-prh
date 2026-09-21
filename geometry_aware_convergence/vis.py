for metric in ['cycle_knn', 'mutual_knn', 'cka', 'unbiased_cka', 'cknna']:
    # Step 1: Load files
    # Load .npy file from baseline and null-calibrated versions of the /workspace/results/emily/alignment directory.

    # Step 2: Plot language-language alignment

    llm_models = [
        "bigscience/bloomz-560m",
        "bigscience/bloomz-1b1",
        "bigscience/bloomz-1b7",
        "bigscience/bloomz-3b",
        "bigscience/bloomz-7b1",
        "openlm-research/open_llama_3b",
        "openlm-research/open_llama_7b",
        "openlm-research/open_llama_13b",
        "huggyllama/llama-7b",
        "huggyllama/llama-13b",
    ]

    n = len(llm_models)

    # Select the top n by n matrix for baseline, null-calibrated, AND the delta, and plot side-by-side as heat maps.

    # Step 3: Plot vision-vision alignment per task

    lvm_models = [
        "vit_tiny_patch16_224.augreg_in21k",
        "vit_small_patch16_224.augreg_in21k",
        "vit_base_patch16_224.augreg_in21k",
        "vit_large_patch16_224.augreg_in21k",
        "vit_base_patch16_224.mae",
        "vit_large_patch16_224.mae",
        "vit_huge_patch14_224.mae",
        "vit_small_patch14_dinov2.lvd142m",
        "vit_base_patch14_dinov2.lvd142m",
        "vit_large_patch14_dinov2.lvd142m",
        "vit_giant_patch14_dinov2.lvd142m",
        "vit_base_patch16_clip_224.laion2b",
        "vit_large_patch14_clip_224.laion2b",
        "vit_huge_patch14_clip_224.laion2b",
        "vit_base_patch16_clip_224.laion2b_ft_in12k",
        "vit_large_patch14_clip_224.laion2b_ft_in12k",
        "vit_huge_patch14_clip_224.laion2b_ft_in12k",
    ]

    # Split lvm_models by task (augreg_in21k, mae, lvd142m, laion2b, laion2b_ft_in12k), and plot model vs model heatmaps for the metric. Also for baseline, null-calib, and delta side by side.

    # Step 4: Using the same lvm models split, plot language vs vision alignment, similar setup.

    # Save all of these in /figures-0921