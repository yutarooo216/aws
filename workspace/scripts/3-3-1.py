import boto3
import sagemaker

session = sagemaker.Session(default_bucket="sagemaker-ap-northeast-1-980921727789")
bucket = session.default_bucket()
role = 'arn:aws:iam::980921727789:role/service-role/AmazonSageMaker-ExecutionRole-20241204T070530'
region = boto3.Session().region_name

sm = boto3.Session().client(service_name='sagemaker',
                            region_name=region)
max_candidate = 3

job_config = {
        'CompletionCriteria': {
            'MaxRuntimePerTrainingJobInSeconds': 600,
            'MaxCandidates': max_candidate,
            'MaxAutoMLJobRuntimeInSeconds': 3600
            },
        }

input_data_config = [
        {
            'DataSource': {
                'S3DataSource': {
                'S3DataType': 'S3Prefix',
                'S3Uri': 's3://sagemaker-ap-northeast-1-980921727789/inputs/anime_info.csv'
                }
            },
        'TargetAttributeName': 'rating'
            }
        ]

output_data_config = {
        'S3OutputPath': 's3://sagemaker-ap-northeast-1-980921727789/outputs/'
        }

from time import gmtime, strftime, sleep
timestamp_suffix = strftime('%d-%H-%M-%S', gmtime())

auto_ml_job_name = 'automl-dm-' + timestamp_suffix

# model
sm.create_auto_ml_job(AutoMLJobName=auto_ml_job_name,
                      InputDataConfig=input_data_config,
                      OutputDataConfig=output_data_config,
                      AutoMLJobConfig=job_config,
                      RoleArn=role)
job = sm.describe_auto_ml_job(AutoMLJobName=auto_ml_job_name)
job_status = job['AutoMLJobStatus']
job_sec_status = job['AutoMLJobSecondaryStatus']

if job_status not in ('Stopped', 'Failed'):
    while job_status in ('InProgress') and job_sec_status in ('AnalyzingData'):
        job = sm.describe_auto_ml_job(AutoMLJobName=auto_ml_job_name)
        job_status = job['AutoMLJobStatus']
        job_sec_status = job['AutoMLJobSecondaryStatus']
        print(job_status, job_sec_status)
        sleep(100)
    print("Data analysis complete")

print(job)
