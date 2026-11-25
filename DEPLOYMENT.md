# Hosting notes

## Run locally

Install the dependencies and run the Streamlit server from the project root:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

## Netlify

This project cannot run as a native Netlify deployment in its current Streamlit form. Streamlit needs a Python server process and persistent WebSocket sessions. Netlify's current long-lived server runtime runs Node.js; its Python support is for build-time tooling, not a Python Streamlit app runtime.

For the simplest public demo, put this folder in a GitHub repository, open Streamlit Community Cloud, create an app, and select `app.py` as the entrypoint. The included `requirements.txt`, datasets, and model are in the repository for deployment.

In the deployment form, open **Advanced settings** and select **Python 3.13**. This matches the Python version used to build the included model artifact and pinned libraries.

After deployment, check the app logs once to confirm that all pinned packages installed and the model loaded. The app needs no secrets or API keys.

If Netlify is mandatory, it can host a separately built static frontend, but the prediction service must run on a Python-capable host and the frontend must call that service. This ZIP does not pretend Netlify can run `streamlit run app.py` directly.

The included datasets and model are local files, so the Streamlit-hosted app does not need a weather API or external dataset download at runtime. Dataset C, calibration tables, comparison tables, and reliability-curve data are included in the repository.

Official references:

- Streamlit architecture: https://docs.streamlit.io/develop/concepts/architecture/architecture
- Streamlit Community Cloud deployment: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- Netlify Server runtime: https://docs.netlify.com/build/server/get-started/
