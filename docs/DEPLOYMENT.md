# Publish the interactive laboratory

The Python application runs on Streamlit Community Cloud; GitHub hosts its source. GitHub Pages alone cannot execute the codec.

1. Sign in at https://share.streamlit.io with your GitHub account and authorize repository access.
2. Click Create app, then deploy from GitHub.
3. Repository: `sahan24854/image-and-video-laboratory`.
4. Branch: `main`. Entrypoint: `app.py`.
5. In advanced settings choose Python 3.12. No secrets are required.
6. Choose an available subdomain, for example `image-and-video-laboratory`, then deploy.
7. Open the resulting URL and run Validation bench. Test a small image and the built-in video before sharing.

The actual public URL is assigned by Streamlit; the suggested subdomain is not reserved. Source updates on the selected branch redeploy the app. Start with small clips to limit CPU/memory use.

If dependency installation fails, check the deployment logs. The root requirements.txt installs the local package and UI extra, including PyAV.
