import boto3
import yaml
import os


def main():
    global_settings = yaml.safe_load(open('./global_settings.yaml'))
    credentials = yaml.safe_load(open('./remote/credentials.yaml'))

    repo_loc = global_settings['repo_loc']

    s3 = boto3.client('s3',
         aws_access_key_id=credentials['ACCESS_ID'],
         aws_secret_access_key= credentials['ACCESS_KEY'])

    s3r = boto3.resource('s3',
         aws_access_key_id=credentials['ACCESS_ID'],
         aws_secret_access_key= credentials['ACCESS_KEY'])

    bucket_name = global_settings['s3_bucket']
    bucket = s3r.Bucket(bucket_name)

    bucket_objects = s3.list_objects(Bucket=bucket_name)
    assert 'Contents' in bucket_objects
    all_bucket_keys = [x['Key'].strip('/') for x in bucket_objects['Contents']]

    project_bucket_keys = list(filter(lambda x: global_settings['current_project'] in x, all_bucket_keys))
    processed_data_dir = repo_loc + 'projects/' + global_settings['current_project'] + '/data/processed/'
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
    for key in project_bucket_keys:
        if 'global_settings' in key:
            pass
        elif '.txt' in key:
            file_name = key.split('/')[-1]
            bucket.download_file(key, processed_data_dir + '/' +  file_name)
        else:
            pass

if __name__ == "__main__":
    main()