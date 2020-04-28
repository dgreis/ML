import boto3
import time
import paramiko
import os
import docker
import tarfile

from subprocess import Popen,PIPE
from .remoteutils import exists_remote
from utils import get_args

class Environment(object):

    def __init__(self):
        steps = self.steps
        method_steps = list()
        for s in steps:
            method_name = s.lower().replace(' ','_')
            method_steps.append(method_name)
        self.method_steps = method_steps

    def do(self, s, workspace, project_settings):
        method_name = s.lower().replace(" ", "_")
        method = getattr(self,method_name)
        kwarg_keys = get_args(self, method_name)
        kwargs = dict()
        for k in kwarg_keys:
            kwargs[k] = eval(k)
        return method(**kwargs)

    def connect_to_docker_server(self, workspace):
        base_url = self.base_url
        docker_host = self.docker_host
        apic = docker.APIClient(base_url=base_url)
        Err = True
        Max_tries = 5
        i = 0
        interval = 10
        print("Attempt Max: " + str(Max_tries) + " times to establish tunnel docker client")
        while Err and i <= Max_tries:
            try:
                dc = docker.from_env(environment={'DOCKER_HOST': docker_host })
                active_containers = dc.containers.list()
                Err = False
            except Exception:
                print("Attempt #" + str(i+1) + " to instantiate tunnel docker client failed. Try again in " + str(
                    interval) + " seconds")
                time.sleep(10)
                i += 1
        print("Client connection established after " + str(
                    i + 1) + " attempts.")
        workspace['active_containers'] = active_containers
        workspace['docker_client'] = dc
        workspace['api_client'] = apic
        return workspace

    def pull_and_run_docker_container(self, workspace, project_settings):
        active_containers = workspace['active_containers']
        apic = workspace['api_client']
        dc = workspace['docker_client']
        print("Now checking for active docker containers")
        if len(active_containers) > 0:
            c = active_containers[0]
        else:
            print("No active containers found.")
            progress = apic.pull('dgreis/ml', 'latest', stream=True,
                                 auth_config={'username': project_settings['DOCKER_USERNAME']
                                     , 'password': project_settings['DOCKER_PASSWORD']})
            for val in progress:
                print(val.strip())
            print("Start new docker container")
            c = dc.containers.run('dgreis/ml:latest',
                                  environment={
                                      'CURRENT_PROJECT': project_settings['current_project'],
                                      'S3_BUCKET': project_settings['remote_settings']['s3_bucket'],
                                      'REPO_LOC': project_settings['remote_settings']['remote_repo_loc']
                                  },
                                  detach=True)
        workspace['container'] = c
        return workspace

    def upload_credentials_on_to_container(self, workspace):
        # Add AWS credentials
        c = workspace['container']
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
        return workspace

    def run_program(self, workspace):
        c = workspace['container']
        apic = workspace['api_client']
        cmds = ["/bin/sh", "-c", 'cd $REPO_LOC && bash startup.sh']
        exe = apic.exec_create(container=c.id, cmd=cmds)
        exe_start = apic.exec_start(exec_id=exe, stream=True)
        for val in exe_start:
            print(val.strip())
        if 'forwarding_process' in workspace:
            forwarding_process = workspace['forwarding_process']
            forwarding_process.kill()

class EC2Env(Environment):

    def __init__(self):
        self.steps = [
             "Provision EC2 Resource"
            ,"Upload Essential Files to EC2"
            ,"Enable Local Port Forwarding"
            ,"Connect to Docker Server"
            ,"Pull and Run Docker Container"
            ,"Upload Credentials on to Container"
            ,"Run Program"
        ]
        self.base_url = 'tcp://localhost:2375'
        self.docker_host = 'tcp://localhost:2375'
        super().__init__()

    def provision_ec2_resource(self, workspace, project_settings):
        print("Checking AWS for instances that are already running...")
        ec2 = boto3.resource('ec2', region_name='eu-west-1')
        instances_init = list(ec2.instances.filter(
            Filters=[{'Name': 'instance-state-name', 'Values': ['running']}]))
        if len(instances_init) > 0:
            print("Existing instance found.")
        else:
            instance_type = project_settings['remote_settings']['aws']['instance_type']
            print("No running instance found. Spawning a new " + instance_type + " instead")
            ec2.create_instances(ImageId='ami-0f2ed58082cb08a4d'
                                 , MinCount=1, MaxCount=1
                                 , InstanceType=instance_type
                                 , KeyName='remote-ml'
                                 , SecurityGroups=['remote-ml-sg']
                                 , BlockDeviceMappings=[
                                        {'Ebs': {'VolumeSize': 16},
                                         'DeviceName': '/dev/sda1'
                                        }
                                    ]
                                 )
            init_time = 30
            print("Sleep " + str(init_time) + " seconds while instance initiates")
            time.sleep(init_time)
            print("Sleep time over; get to work!")
        instance = list(ec2.instances.filter(
            Filters=[{'Name': 'instance-state-name', 'Values': ['running']}]))[0]
        workspace['instance'] = instance
        return workspace

    def upload_essential_files_to_ec2(self, workspace):
        # Connect to SSH Client
        print("Attempting connection to EC2 instance via SSH")
        instance = workspace['instance']
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        privkey = paramiko.RSAKey.from_private_key_file('./remote/remote-ml.pem')
        ssh.connect(instance.public_dns_name, username='ubuntu', pkey=privkey)
        print("Success")
        # Uploading essential files from local
        sftp = ssh.open_sftp()
        if not exists_remote(sftp, '/home/ubuntu/.aws'):
            ssh.exec_command('mkdir -p /home/ubuntu/.aws')
        for file in ['config', 'credentials']:
            sftp.put(os.path.expanduser('~') + '/.aws/' + file, '/home/ubuntu/.aws/' + file)
        sftp.put('./remote/ec2bootup.sh', '/home/ubuntu/ec2bootup.sh')
        sftp.close()
        print("Essential credentials and ec2 boot-up scripts added via SFTP")

        # Running SSH Commands
        ssh.exec_command('bash ec2bootup.sh')
        print("Running Docker install script on instance")
        return workspace

    def enable_local_port_forwarding(self, workspace):
        # Docker port forwarding
        bashCommand = "ssh -o StrictHostKeyChecking=no -i ./remote/remote-ml.pem  -N -L 2375:/var/run/docker.sock ubuntu@" + instance.public_dns_name
        forwarding_process = Popen(bashCommand.split(), stdout=PIPE)
        print('Begin local port forwarding so this computer can run remote docker server. Spawned forwarding process, id: ' + str(
                forwarding_process.pid))
        time.sleep(5)
        workspace['forwarding_process'] = forwarding_process
        return workspace

class LocalDockerEnv(Environment):

    def __init__(self):
        self.steps = [
             "Connect to Docker Server"
            ,"Pull and Run Docker Container"
            ,"Upload Credentials on to Container"
            ,"Run Program"
        ]
        self.base_url = None
        self.docker_host = None