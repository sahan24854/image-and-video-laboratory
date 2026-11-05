# Reproducible image and video coding experiment

## Purpose

Measure how transform, quantization, chroma sampling and temporal prediction change actual compressed size, reconstruction quality and execution time. Distinguish correctness checks from content-dependent observations. No single setting is expected to win every comparison.

## Preparation and acceptance checks

1. Activate the project environment and install the UI: `python -m pip install -e '.[ui]'`.
2. Run `python -m unittest discover -s tests -v`. Every test must pass.
3. Launch `streamlit run app.py`. Open Validation bench and run all four self-checks; all must pass before experiments.
4. Record the Git revision, Python/package versions, operating system and processor. Keep inputs and their original dimensions fixed. Close other heavy applications when measuring time.
5. Create a dated folder for your CSV downloads and screenshots. Retain encoded files and original inputs. Repeat timing measurements three times and report the median; the first run may include warm-up overhead.

## Inputs

Begin with the built-in gradient, edges and seeded noise specimens. Then use three redistributable images: smooth/photo portrait, detailed foliage, and colored text/graphics. Use an odd-sized crop such as 127×95 for padding tests. Keep source files unchanged; note filename, dimensions and SHA-256 in the report.

Begin video with the built-in moving-square clip, which changes scene at frame 8. Then use a static-camera clip, a moving-object clip and a clip with a scene cut. Select 12 initial frames at maximum width 192 pixels. Record source FPS, decoded dimensions, actual frame count and any resizing. Metrics describe these analysis frames, not the original full-resolution clip. The app ignores audio and represents variable-rate input with average FPS.

## Baseline

Image: JPEG table, 8×8 DCT, Q=1, 4:4:4, edge padding, zlib.

Video: JPEG table, 8×8 DCT, Q=1, hybrid prediction, radius ±8 pixels, search step 2 pixels, SAD, I-frame interval 8 frames, zlib. Video currently codes RGB 4:4:4 residuals. Use a single FPS value for comparisons.

For every run press Run experiment, inspect the completed settings, then Add completed experiment to comparison. Controls can differ from the last completed result until you run again. Export comparison CSV at the end of each source; avoid mixing sources in a rate–distortion curve.

## Image experiments

| ID | Change only | Values | Measurements and observation |
|---|---|---|---|
| I1 | Q | 0.25, 0.5, 1, 4, 12 | Plot bits/pixel against PSNR and SSIM; inspect zeros, blocking and ringing |
| I2 | Quantization table | JPEG, weighted, uniform | Repeat I1 for each table; compare overlapping quality ranges rather than identical Q alone |
| I3 | Transform | DCT, identity | Repeat I1; smooth and noise inputs may behave differently |
| I4 | Chroma sampling | 444, 422, 420 | Same Q/input; inspect colored text and color bleeding |
| I5 | Padding | edge, constant | Odd dimensions; select final block in simulation and inspect AC energy and reconstructed edges |
| I6 | Entropy coder | zlib, bz2, lzma | Same source and quantization; reconstructions should be identical while sizes/times can differ |
| I7 | Block side | 4, 8, 16 pixels | Use weighted table (JPEG requires 8); compare artifacts and measured sizes |

The five-scale sweep button automates I1 for completed settings. The mechanism stage selector shows numeric processing; it is not a video of CPU execution. Inspect the same block before and after changes.

## Video experiments

| ID | Change only | Values | Measurements and observation |
|---|---|---|---|
| V1 | Prediction | previous, motion, hybrid | Per-frame PSNR/SSIM, whole-file bytes, residual appearance, DC counts |
| V2 | Search step | 1, 2, 4 pixels | Same radius/block; step 1 covers all candidates; cost should be no worse than sampled search on the same block/reference |
| V3 | Search radius | 0, 4, 8 pixels | Compare motion displacement coverage and elapsed time; size/quality need not improve monotonically |
| V4 | Matching cost | SAD, MSE | Compare selected vectors and outlier sensitivity; raw numerical costs have different units |
| V5 | I-frame interval | 1, 4, 8 frames | All-intra versus prediction; inspect frame types and scene-change residuals |
| V6 | Q | 0.25, 0.5, 1, 4, 12 | Per-frame distortion and complete-sequence rate–distortion curve |

Use the frame slider in Input → output to inspect frame 0, a motion frame, and frame 8. Compare the original, decoded frame and absolute error ×4. In Inside the codec inspect actual reconstructed-reference predictions. The separate temporal search simulation uses the input frame pair; do not equate its decisions with the actual encoded reference loop.

## Size-control experiments

For both image and video first encode at baseline and note B bytes. Request approximately 0.5B, B and 1.5B bytes with maximum mode. Record actual bytes, selected Q, within_budget, relative_error and evaluation trace. A candidate under budget passes the budget check. Failure to find one is a search limitation or infeasible target to investigate, not proof of a broken decoder. Also try 1 byte: it should report no feasible candidate. Whole-video targeting is not CBR or per-frame allocation.

## Recording template

Use one row per run: source ID/hash, media kind, dimensions/frame count/FPS, all settings, total bytes, bits/pixel, encode time, PSNR, SSIM, MSE, MAE, visual artifacts and pass/fail notes. Video adds inspected frame indices and DC-block counts. The UI comparison includes settings and mean scores; download full per-frame scores with the CLI analysis command when needed.

## Pass/fail versus hypotheses

Required: no crashes on valid supported inputs; decoded dimensions/frame count match; decoded outputs use uint8; exact identity self-check passes; entropy-coder changes preserve reconstructed pixels; metrics for identical inputs give MSE=0, SSIM=1 and infinite PSNR. Each reported byte count must equal the downloaded encoded file's size.

Investigate rather than fail automatically: a method produces a larger file, search fails to meet a target, SSIM and PSNR disagree, or higher search cost does not improve final quality. These depend on quantization, side information, content and heuristic choices. Never claim DCT, hybrid prediction or a specific compressor always wins.

## Final report

Include environment and input manifest; correctness results; one rate–distortion plot per source; method comparison tables; three original/reconstruction/error examples; video frame-level observations around the scene cut; timing medians; and explanations of unexpected results. Separate implemented mechanisms from theory-only extensions. Synthetic runs demonstrate behavior but do not establish general superiority over standard codecs.
