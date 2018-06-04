## Hospital Readmission Model

This is a model for predicting hospital readmission among patients with diabetes.  Data from UCI:
[https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)

Model is built with Sklearn.  Python environment specified in `requirements.txt`.  I also created a Docker image which packages the .pkl as an executable.

| Model | AUC |
|--- | --- |
| LR+RF+XGB Stack | 0.6990824552912449 |
| LR+RF+XGB Avg | 0.6981398497127431 |
| XGB | 0.6956653497449965 |
| RF | 0.6952079165690574 |
| LR | 0.684611003872049 |

Build model (output is model.pkl):
```bash
python model.py
```

Build container for generating predictions with model:
```bash
gcloud container builds submit --gcs-source-staging-dir=gs://djr-data/cloudbuild \
    --async --timeout 4h0m0s --config cloudbuild.yaml .
```

NOTE: Container is built asyncronously with Google Container Builder and stored in Google Container Repository.

Generate predictions with container:
```bash
export PROJECT_ID=$(gcloud config get-value project -q)
export IMAGE_ID=gcr.io/${PROJECT_ID}/hospital-readmissions:latest
gcloud docker -- pull ${IMAGE_ID}
docker run -v $(pwd)/data:/usr/share/model/data ${IMAGE_ID} --input diabetic_data.csv
```

Shelling into container for testing:
```bash
docker run -it -v $(pwd)/data:/usr/share/model/data --entrypoint=ash ${IMAGE_ID}
```
