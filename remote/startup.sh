#!/usr/bin/env bash

git clone -b skorch-nn https://dgreis:$GIT_TOKEN@github.com/dgreis/ML.git

aws s3 cp s3://$S3_BUCKET/$CURRENT_PROJECT/global_settings.yaml /$REPO_LOC/global_settings.yaml
cd $REPO_LOC
python ./remote/pullfiles.py
python -u ./run.py