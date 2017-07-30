TRAIN_FILE=gs://analiticauniversal/creditcard.data.csv
EVAL_FILE=gs://analiticauniversal/creditcard.test.csv

export GCS_JOB_DIR=gs://analiticauniversal/LogsMLEngine
JOB_ID="ccf_gpu_${USER}_$(date +%Y%m%d_%H%M%S)"
export TRAIN_STEPS=500
export EV_STEPS=20

# Para el hyperparameter tuning añadir esto
export HPTUNING_CONFIG=hptuning_config.yaml
# añadir flag --config $HPTUNING_CONFIG en el comando de ejecutar el mlengine

# Añadiendo  --scale-tier $SCALE_TIER antes de region se puede poner BASIC_GPU para entrenar con una máquina con GPU o STANDARD_1 para entrenar con un cluster de 10 máquinas
export SCALE_TIER=BASIC_GPU

# Si no realizamos tuneo de hiperparámetros
gcloud ml-engine jobs submit training "$JOB_ID"  --stream-logs --runtime-version 1.2 --job-dir $GCS_JOB_DIR --module-name trainer.task --package-path trainer/ --scale-tier $SCALE_TIER --region us-central1 -- --train-files $TRAIN_FILE  --train-steps $TRAIN_STEPS --eval-files $EVAL_FILE --first-layer-size 25 --num-layers 2 --verbosity DEBUG --eval-steps $EV_STEPS  > log.txt

# Para el tuneo de hiperparámetros
#gcloud ml-engine jobs submit training "$JOB_ID"  --stream-logs --runtime-version 1.2 --config $HPTUNING_CONFIG --job-dir $GCS_JOB_DIR --module-name trainer.task --package-path trainer/ --scale-tier $SCALE_TIER --region us-central1 -- --train-files $TRAIN_FILE  --train-steps $TRAIN_STEPS --eval-files $EVAL_FILE --verbosity DEBUG --eval-steps $EV_STEPS  > log.txt

mkdir $JOB_ID
mv log.txt $JOB_ID
#gsutil -m mv gs://analiticauniversal/LogsMLEngine/* $JOB_ID
gsutil -m cp -r gs://analiticauniversal/LogsMLEngine/* $JOB_ID
gsutil -m rm gs://analiticauniversal/LogsMLEngine/**

ZIP_FILE=$JOB_ID$".zip"
zip -r $ZIP_FILE $JOB_ID
gsutil -m cp -r $ZIP_FILE gs://analiticauniversal/ResultadosMLEngine
rm -rf $JOB_ID
