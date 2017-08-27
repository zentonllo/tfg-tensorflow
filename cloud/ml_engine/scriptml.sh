# Use these commands if the dataset hasn't been uploaded (manually, via Cloud Console) to the Cloud Storage
# gsutil -m cp creditcard.data.csv gs://analiticauniversal/creditcard.data.csv
# gsutil -m cp creditcard.test.csv gs://analiticauniversal/creditcard.test.csv

# Cloud Storage paths for the train and validation (eval) sets for ML Engine
TRAIN_FILE=gs://analiticauniversal/creditcard.data.csv
EVAL_FILE=gs://analiticauniversal/creditcard.test.csv

# Generate a job (job names cannot be repeated)
export GCS_JOB_DIR=gs://analiticauniversal/LogsMLEngine
JOB_ID="ccf_gpu_${USER}_$(date +%Y%m%d_%H%M%S)"

# 500 epochs for training and 20 for evaluation
export TRAIN_STEPS=500
export EV_STEPS=20

# Flag used for hyperparameter tuning
# Add flag --config $HPTUNING_CONFIG in the ML Engine command (before '--' argument)
export HPTUNING_CONFIG=hptuning_config.yaml


# We could train using a GPU (BASIC_GPU) or a 10 machines cluster (STANDARD_1)
export SCALE_TIER=BASIC_GPU

# Training ML Engine command
gcloud ml-engine jobs submit training "$JOB_ID"  --stream-logs --runtime-version 1.2 --job-dir $GCS_JOB_DIR --module-name trainer.task --package-path trainer/ --scale-tier $SCALE_TIER --region us-central1 -- --train-files $TRAIN_FILE  --train-steps $TRAIN_STEPS --eval-files $EVAL_FILE --first-layer-size 25 --num-layers 2 --verbosity DEBUG --eval-steps $EV_STEPS  > log.txt

# Hyperparameter ML Engine command
#gcloud ml-engine jobs submit training "$JOB_ID"  --stream-logs --runtime-version 1.2 --config $HPTUNING_CONFIG --job-dir $GCS_JOB_DIR --module-name trainer.task --package-path trainer/ --scale-tier $SCALE_TIER --region us-central1 -- --train-files $TRAIN_FILE  --train-steps $TRAIN_STEPS --eval-files $EVAL_FILE --verbosity DEBUG --eval-steps $EV_STEPS  > log.txt

# Create local folder to download ML Engine results
mkdir $JOB_ID
mv log.txt $JOB_ID
gsutil -m cp -r gs://analiticauniversal/LogsMLEngine/* $JOB_ID
gsutil -m rm gs://analiticauniversal/LogsMLEngine/**

# Create a zip with the results and upload them back to Storage
ZIP_FILE=$JOB_ID$".zip"
zip -r $ZIP_FILE $JOB_ID
gsutil -m cp -r $ZIP_FILE gs://analiticauniversal/ResultadosMLEngine
rm -rf $JOB_ID
