Build container for model:
```bash
gcloud container builds submit --gcs-source-staging-dir=gs://djr-data/cloudbuild \
    --async --config cloudbuild.yaml .
```

Generate predictions with model:
```bash
docker pull gcr.io/$PROJECT_ID/hospital-readmissions:latest
docker run --mount source=data,target=/usr/share/model/data \
	hospital-readmission:latest diabetic_data.csv
```