# transformers v5.23.0 privacy-filter

Serves the `openai/privacy-filter` token-classification model over HTTP with FastAPI and the
transformers v5.23.0 pipeline, on the upstream `pytorch/pytorch` 2.14.1 CUDA 12.6 runtime image.

- `server.py`: the HTTP server, started by `uvicorn server:app` on port 8000. It loads the model at
  the revision pinned in the file.
- `requirements.txt`: hashed lock of the Python packages installed on top of the base, with
  `--no-deps`. Packages the base already provides at a compatible version, including torch and
  its CUDA libraries, are not listed and stay as the base ships them.
- `linux-libc-dev` is pinned to the noble-security version.
