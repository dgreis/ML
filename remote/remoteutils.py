from utils import *
from ruamel.yaml import YAML
import boto3
import os
import tarfile
import docker
import time
import paramiko
from subprocess import Popen,PIPE

credentials = yaml.safe_load(open('./remote/credentials.yaml'))

def handle_remote(project_settings):

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

    #2.0 spawn ec2 instance
    ec2 = boto3.resource('ec2', region_name='eu-west-1')
    instances_init = list(ec2.instances.filter(
        Filters=[{'Name': 'instance-state-name', 'Values': ['running']}]))
    print("Checking AWS for instances that are already running")
    if len(instances_init) > 0:
        pass
        #TODO: write logic
    else:
        print("No running instance found. Spawning a new instance instead")
        ec2.create_instances(ImageId='ami-0f2ed58082cb08a4d'
                             ,MinCount=1, MaxCount=1
                             ,InstanceType='t2.large'
                             ,KeyName='remote-ml'
                             ,SecurityGroups=['remote-ml-sg']
                             ,BlockDeviceMappings=[
                                {'Ebs': {'VolumeSize': 16},
                                 'DeviceName': '/dev/sda1'
                                }])
        init_time = 30
        print("Sleep " + str(init_time) + " seconds while instance initiates")
        time.sleep(init_time)
        print("Sleep time over; get to work!")
    instance = list(ec2.instances.filter(
        Filters=[{'Name': 'instance-state-name', 'Values': ['running']}]))[0]

    #Connect to SSH Client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    privkey = paramiko.RSAKey.from_private_key_file('./remote/remote-ml.pem')
    print("connecting")
    ssh.connect(instance.public_dns_name,username='ubuntu',pkey=privkey)
    print("connected")

    #Uploading essential files from local
    sftp = ssh.open_sftp()
    if not exists_remote(sftp, '/home/ubuntu/.aws'):
        ssh.exec_command('mkdir -p /home/ubuntu/.aws')
    for file in ['config','credentials']:
        sftp.put( os.path.expanduser('~') + '/.aws/' + file, '/home/ubuntu/.aws/' + file)
    sftp.put('./remote/ec2bootup.sh','/home/ubuntu/ec2bootup.sh')
    sftp.close()
    print("Essential credentials and ec2 boot-up scripts added via SFTP")

    #Running SSH Commands
    stdin, stdout, stderr = ssh.exec_command('bash ec2bootup.sh')
    #stdin, stdout, stderr = ssh.exec_command('curl -fsSL https://get.docker.com -o get-docker.sh')
    #stdin, stdout, stderr = ssh.exec_command('sudo sh get-docker.sh')

    #Docker port forwarding
    bashCommand = "ssh -o StrictHostKeyChecking=no -i ./remote/remote-ml.pem  -N -L 2375:/var/run/docker.sock ubuntu@" + instance.public_dns_name
    #process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    forwarding_process = Popen(bashCommand.split(), stdout=PIPE)
    print('Begin local port forwarding so this computer can run remote docker server. Spawned forwarding process, id: ' + str(forwarding_process.pid))
    #output, error = process.communicate()
    #os.environ["DOCKER_HOST"] = "tcp://localhost:2375"
    time.sleep(5)

    # assert 1 == 0

    # #stdin.flush()
    # data = stdout.read().splitlines()
    # err = stderr.read().splitlines()
    # for line in err:
    #     x = line.decode()
    #     #print(line.decode())
    #     print(x)
    # ssh.close()
    # assert 1 == 0
    # 2.1 create docker and run script


    #apic = docker.APIClient()
    apic = docker.APIClient(base_url='tcp://localhost:2375')
    Err = True
    Max_tries = 5
    i = 0
    interval = 10
    print("Attempt Max: " + str(Max_tries) + " times to establish tunnel docker client")
    while Err and i <= Max_tries:
        try:
            dc = docker.from_env(environment={'DOCKER_HOST': 'tcp://localhost:2375'})
            active_containers = dc.containers.list()
            Err = False
        except Exception:
            print("Attempt #" + str(i) + " to instantiate tunnel docker client failed. Try again in "+ str(interval) +  " seconds")
            time.sleep(10)
            i += 1
    print("Client connection established after " + str(i) + "attempts. Now checking for active docker containers")
    if len(active_containers) > 0:
        c = active_containers[0]
    else:
        print("No active containers found.")
        # resp = dc.login(username=credentials['DOCKER_USERNAME'],
        #          password=credentials['DOCKER_PASSWORD']
        #         )
        progress = apic.pull('dgreis/ml', 'latest', stream=True,
                  auth_config={'username': credentials['DOCKER_USERNAME']
                             , 'password': credentials['DOCKER_PASSWORD']})
        for val in progress:
            print(val.strip())
        #dc.images.pull('dgreis/ml','latest')
        #dc = docker.from_env()
        print("Start new docker container")
        c = dc.containers.run('dgreis/ml:latest',
                          environment={
                              'CURRENT_PROJECT': project_settings['current_project'],
                              'S3_BUCKET': project_settings['remote_settings']['s3_bucket'],
                              'REPO_LOC': project_settings['remote_settings']['remote_repo_loc']
                          },
                          detach=True)
        # c = apic.create_container('dgreis/ml:latest',
        #                           environment={
        #                             'CURRENT_PROJECT': project_settings['current_project'],
        #                             'S3_BUCKET': project_settings['remote_settings']['s3_bucket'],
        #                             'REPO_LOC': project_settings['remote_settings']['remote_repo_loc']
        #                             })

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
    print("credentials loaded into docker container. Now run program")

    cmds = ["/bin/sh", "-c", 'cd $REPO_LOC && bash startup.sh']
    exe = apic.exec_create(container=c.id, cmd=cmds)
    exe_start = apic.exec_start(exec_id=exe, stream=True)
    for val in exe_start:
        print(val.strip())
    forwarding_process.kill()

def exists_remote(sftp_client, path):
    try:
        sftp_client.stat(path)
    except FileNotFoundError:
        return False
    else:
        return True


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
