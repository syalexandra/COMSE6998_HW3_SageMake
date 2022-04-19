from sagemaker import get_execution_role
import pandas
import boto3

from sagemaker.mxnet import MXNet



bucket_name = 'spam-detection-hw3-bucket'

role = get_execution_role()
bucket_key_prefix = 'sms-spam-classifier'
vocabulary_length = 9013

print(role)

client = boto3.client('sagemaker')
try:
    client.delete_endpoint(
        EndpointName='sms-spam-classifier-mxnet'
    )
except:
    pass



s3 = boto3.resource('s3')
target_bucket = s3.Bucket(bucket_name)
"""
with open('dataset/sms_train_set.gz', 'rb') as data:
    target_bucket.upload_fileobj(data, '{0}/train/sms_train_set.gz'.format(bucket_key_prefix))
    
with open('dataset/sms_val_set.gz', 'rb') as data:
    target_bucket.upload_fileobj(data, '{0}/val/sms_val_set.gz'.format(bucket_key_prefix))
"""

output_path = 's3://{0}/{1}/output'.format(bucket_name, bucket_key_prefix)
code_location = 's3://{0}/{1}/code'.format(bucket_name, bucket_key_prefix)

m = MXNet('/home/ec2-user/SageMaker/COMSE6998_HW3_SageMake/sms_spam_classifier_mxnet_script.py',
          role=role,
          train_instance_count=1,
          instance_type='ml.c5.2xlarge',
          output_path=output_path,
          base_job_name='sms-spam-classifier-mxnet',
          framework_version="1.2",
          py_version="py3",
          model_uri=output_path,
          code_location = code_location,
          hyperparameters={'batch_size': 100,
                         'epochs': 20,
                         'learning_rate': 0.01})

inputs = {'train': 's3://{0}/{1}/train/'.format(bucket_name, bucket_key_prefix),
 'val': 's3://{0}/{1}/val/'.format(bucket_name, bucket_key_prefix)}

m.fit(inputs)


mxnet_pred = m.deploy(initial_instance_count=1,
                      instance_type='ml.m5.large',
                      endpoint_name='sms-spam-classifier-mxnet')



client.stop_notebook_instance(NotebookInstanceName='spam-detection-instance')
