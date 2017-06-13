TRAIN_FILE=gs://cloudml-public/census/data/adult.data.csv
EVAL_FILE=gs://cloudml-public/census/data/adult.test.csv

export GCS_JOB_DIR=gs://analiticauniversal/LogsMLEngine
JOB_ID="census_gpu_${USER}_$(date +%Y%m%d_%H%M%S)"
export TRAIN_STEPS=2000

gcloud ml-engine jobs submit training "$JOB_ID" --stream-logs --runtime-version 1.0 --job-dir $GCS_JOB_DIR --module-name trainer.task --package-path trainer/ --region us-central1 --verbosity info -- --train-files $TRAIN_FILE --eval-files $EVAL_FILE --train-steps $TRAIN_STEPS --first-layer-size 5 --num-layers 1 --verbosity INFO > log.txt

mkdir "$JOB_ID"
mv log.txt "$JOB_ID"
gsutil -m cp -r gs://analiticauniversal/LogsMLEngine "$JOB_ID"
