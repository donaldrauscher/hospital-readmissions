Build container for model:
```bash
gcloud container builds submit --gcs-source-staging-dir=gs://djr-data/cloudbuild \
    --async --timeout 4h0m0s --config cloudbuild.yaml .
```

Generate predictions with model:
```bash
export PROJECT_ID=$(gcloud config get-value project -q)
export IMAGE_ID=gcr.io/${PROJECT_ID}/hospital-readmissions:latest
gcloud docker -- pull ${IMAGE_ID}
docker run -v $(pwd)/data:/usr/share/model/data ${IMAGE_ID} --input diabetic_data.csv
```

Shelling into image for testing:
```bash
docker run -it -v $(pwd)/data:/usr/share/model/data --entrypoint=ash ${IMAGE_ID}
```