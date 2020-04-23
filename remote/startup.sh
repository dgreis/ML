#!/usr/bin/env bash

aws s3 cp s3://$S3_BUCKET/$CURRENT_PROJECT/global_settings.yaml /$REPO_LOC/global_settings.yaml
python ./remote/pullfiles.py
python ./run.py