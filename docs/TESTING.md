# Hands-on laboratory checks

Launch with `.venv/bin/streamlit run app.py` after installing `python -m pip install -e '.[ui]'` in your environment.

1. Open **Validation bench · guided tests** and run the self-checks. All four should show PASS.
2. Image: choose the original gradient, JPEG table, Q=0.25. Run and add to comparison. Change only Q to 12, run and add again. Inspect increased error and reduced PSNR. File size generally decreases but is not strictly monotonic.
3. Choose the noise and edges specimens. Repeat the same Q values; observe that the same settings yield different errors and sizes on different content.
4. Expand the mechanism simulation. Traverse level shift, DCT, quantize and reconstruct. Inspect coefficients and zeros. Pixel levels use 0–255; transform coefficients are signed values.
5. Video: use the built-in clip, radius 8 pixels, step 2 pixels, hybrid prediction. Run. Inspect frame 8 (the scene change), DC counts and predictor/residual images.
6. In temporal simulation, choose a block and traverse candidate indices. Switch step to 1 and compare the best inter cost on the same block. More candidates cannot worsen the best cost, although measured run time varies.
7. Compare previous-frame, motion and hybrid methods on the same video/Q. Check file size, quality, residuals and cost rather than expecting one method to win every case.
8. Enable target bytes. Inspect within_budget and search trace. Try an extremely small target to observe an infeasible budget rather than assuming convergence.

Automated core checks: `python -m unittest discover -s tests -v`.

The app includes color/perception, transform, prediction, entropy, rate control and metric reference sections. Extensions labeled as theory are not additional implemented codecs.
