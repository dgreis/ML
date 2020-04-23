from utils import *
from ruamel.yaml import YAML
import boto3
import os
import tarfile
import docker

def handle_remote(project_settings):
    bucket_name = project_settings['remote_settings']['s3_bucket']
    files_to_export = check_for_missing_files(bucket_name, project_settings)

    # 1.1 Export global settings file
    global_settings = yaml.safe_load(open('./global_settings.yaml'))
    current_project = global_settings['current_project']
    write_remote_global_settings_file(project_settings)
    files_to_export.append('./remote/global_settings.yaml')

    #Other essential files:
    files_to_export.append('./projects/' + current_project + '/src/project_settings.yaml')
    files_to_export.append('./projects/' + current_project + '/src/models.yaml')

    for f in files_to_export:
        send_file_to_s3(bucket_name, current_project, f)

    # 2. TODO: Deploy instance
    dc = docker.from_env()
    active_containers = dc.containers.list()
    if len(active_containers) > 0:
        c = active_containers[0]
    else:
        c = dc.containers.run('ssh_test_image:latest',
                          environment={
                              'CURRENT_PROJECT': project_settings['current_project'],
                              'S3_BUCKET': project_settings['remote_settings']['s3_bucket'],
                              'REPO_LOC': project_settings['remote_settings']['remote_repo_loc']
                          },
                          detach=True)

    #Add AWS credentials
    working_dir = os.getcwd()
    src = os.path.expanduser('~') + '/.aws'
    os.chdir(os.path.dirname(src))
    dst = '/root/'
    srcname = os.path.basename(src)
    tar = tarfile.open('./test' + '.tar', mode='w')
    try:
        tar.add(srcname)
    finally:
        tar.close()

    data = open('./test' + '.tar', 'rb').read()
    c.put_archive(dst, data)
    os.chdir(working_dir)

    cmds = ["/bin/sh", "-c", 'cd $REPO_LOC && bash startup.sh']
    apic = docker.APIClient()
    exe = apic.exec_create(container=c.id, cmd=cmds)
    exe_start = apic.exec_start(exec_id=exe, stream=True)
    for val in exe_start:
        print(val.strip())


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
