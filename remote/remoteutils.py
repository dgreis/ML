from utils import *
from ruamel.yaml import YAML
import boto3
import importlib

def exists_remote(sftp_client, path):
    try:
        sftp_client.stat(path)
    except FileNotFoundError:
        return False
    else:
        return True

def get_environment(env_mode_abbrev):
    env_module = importlib.import_module('remote.environment')
    env_class_name= env_mode_abbrev + 'Env'
    env_class = getattr(env_module, env_class_name)
    return env_class

def send_file_to_s3(bucket_name, current_project, f):
    file_name = f.split('/')[-1]
    if 'processed' in f:
        remote_loc = 'processed/' + file_name
    else:
        remote_loc = file_name
    boto3.client("s3").upload_file(f, bucket_name, current_project + '/' + remote_loc)

def check_for_missing_files(bucket_name, project_settings):
    bucket_objects = boto3.client("s3").list_objects(Bucket=bucket_name)
    current_project = project_settings['current_project']
    if 'Contents' in bucket_objects:
        bucket_keys = [x['Key'].strip('/') for x in bucket_objects['Contents']]
    else:
        bucket_keys = list()
    s3_project_dir = current_project + '/'
    if current_project not in bucket_keys:
        boto3.client("s3").put_object(Bucket=bucket_name, Body='', Key=s3_project_dir)
        boto3.client("s3").put_object(Bucket=bucket_name, Body='', Key=s3_project_dir + 'processed/')

    processed_file_manifest = project_settings['clean_input_files'].values()
    files_to_upload = list()
    data_dir = find_data_dir(project_settings)
    for pf in processed_file_manifest:
        pf_suffix = pf.split('/')[-1]
        if pf_suffix in [x.split('/')[-1] for x in bucket_keys]:
            pass
        else:
            files_to_upload.append(data_dir + pf)
    return files_to_upload

def write_remote_global_settings_file(project_settings):
    yaml = YAML(typ='safe')
    yaml.preserve_quotes = True
    yaml.boolean_representation = ['False', 'True']
    with open('./global_settings.yaml') as f:
        doc = yaml.load(f)
        doc['repo_loc'] = project_settings['remote_settings']['remote_repo_loc']
        doc['remote_settings']['remote_deploy'] = False
    with open('./remote/global_settings.yaml', 'w') as f:
        yaml.dump(doc, f)
