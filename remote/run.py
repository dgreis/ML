
import yaml
from .remoteutils import *
from utils import *

credentials = yaml.safe_load(open('./remote/credentials.yaml'))

def remote_run(project_settings):

    print("Program running in remote mode")
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

    env_mode = project_settings['remote_settings']['environment_mode']
    env_class = get_environment(env_mode)
    env = env_class()

    project_settings = update(project_settings, credentials)

    workspace = dict()
    i = 0
    steps = env.steps
    len_steps = len(steps)
    for s in env.steps:
        print("Step " + str(i+1) + "/" + str(len_steps) + ": " + s)
        workspace = env.do(s, workspace, project_settings)
        i += 1