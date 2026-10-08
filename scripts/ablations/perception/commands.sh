uv run python scripts/ablations/perception/sam3_clip_rerank_eval.py --save-failure-samples --rerank-mode dino --max-demos-per-file 1 --out-dir tmp/perception/abl_dino_image
uv run python scripts/ablations/perception/sam3_clip_rerank_eval.py --save-failure-samples --max-demos-per-file 1 --out-dir tmp/perception/abl_clip_image
uv run python scripts/ablations/perception/sam3_clip_rerank_eval.py --save-failure-samples --max-demos-per-file 1  --rerank-mode text --out-dir tmp/perception/abl_clip_text
uv run python scripts/ablations/perception/dino_sam3_box_eval.py --save-failure-samples --max-demos-per-file 1 --out-dir tmp/perception/abl_dino
uv run python scripts/ablations/perception/sam3_segmentation_eval.py --save-failure-samples --max-demos-per-file 1 --out-dir tmp/perception/abl_baseline
uv run python scripts/ablations/perception/sam3_vlm.py --save-failure-samples --max-demos-per-file 1 --out-dir tmp/perception/abl_vlm