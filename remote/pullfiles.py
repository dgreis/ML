import boto3
import yaml
import os


def main():
    current_dir = os.getcwd()
    global_settings = yaml.safe_load(open(current_dir + '/global_settings.yaml'))

    repo_loc = global_settings['repo_loc']

    s3 = boto3.client('s3')
    s3r = boto3.resource('s3')

    bucket_name = global_settings['remote_settings']['s3_bucket']
    bucket = s3r.Bucket(bucket_name)

    bucket_objects = s3.list_objects(Bucket=bucket_name)
    assert 'Contents' in bucket_objects
    all_bucket_keys = [x['Key'].strip('/') for x in bucket_objects['Contents']]

    project_bucket_keys = list(filter(lambda x: global_settings['current_project'] in x, all_bucket_keys))
    processed_data_dir = repo_loc + 'projects/' + global_settings['current_project'] + '/data/processed/'
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
    project_src_dir = repo_loc + 'projects/' + global_settings['current_project'] + '/src/'
    if not os.path.exists(project_src_dir):
        os.makedirs(project_src_dir)
    for key in project_bucket_keys:
        if 'global_settings' in key:
            bucket.download_file(key, global_settings['repo_loc'] +'/global_settings.yaml')
        elif '.txt' in key:
            file_name = key.split('/')[-1]
            bucket.download_file(key, processed_data_dir + '/' +  file_name)
        elif any( x in key for x in ['models.yaml','project_settings.yaml']):
            file_name = key.split('/')[-1]
            bucket.download_file(key, project_src_dir + '/' + file_name)
        else:
            pass

if __name__ == "__main__":
    main()