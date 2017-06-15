TRAIN_FILE=gs://cloudml-public/census/data/adult.data.csv
EVAL_FILE=gs://cloudml-public/census/data/adult.test.csv

export GCS_JOB_DIR=gs://analiticauniversal/LogsMLEngine
JOB_ID="census_gpu_${USER}_$(date +%Y%m%d_%H%M%S)"
export TRAIN_STEPS=10

# Para el hyperparameter tuning añadir
# export HPTUNING_CONFIG=hptuning_config.yaml
# añadir flag --config $HPTUNING_CONFIG en el comando de ejecutar el mlengine

gcloud ml-engine jobs submit training "$JOB_ID" --stream-logs --runtime-version 1.0 --job-dir $GCS_JOB_DIR --module-name trainer.task --package-path trainer/ --scale-tier BASIC_GPU --region us-central1 -- --train-files $TRAIN_FILE --eval-files $EVAL_FILE --train-steps $TRAIN_STEPS --first-layer-size 3 --num-layers 1 > log.txt

mkdir $JOB_ID
mv log.txt $JOB_ID
gsutil -m mv gs://analiticauniversal/LogsMLEngine/* $JOB_ID
gsutil -m cp -r gs://analiticauniversal/LogsMLEngine/* $JOB_ID
gsutil -m rm gs://analiticauniversal/LogsMLEngine/**

ZIP_FILE=$JOB_ID$".zip"
zip -r $ZIP_FILE $JOB_ID
gsutil -m cp -r $ZIP_FILE gs://analiticauniversal/ResultadosMLEngine
rm -f $ZIP_FILE
rm -rd $JOB_ID
