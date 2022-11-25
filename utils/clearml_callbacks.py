import os

import utils.s3


def get_prefix(model_info):
    storage_key_prefix = model_info.task.metrics_manager.storage_key_prefix
    task_project = os.path.dirname(os.path.dirname(storage_key_prefix))
    full_task_name = model_info.task.name
    task_name = full_task_name.split(':')[0]
    task_id = model_info.task.id
    prefix = os.path.join(task_project, f'{task_name}.{task_id}', 'models')
    return prefix


def get_local_models(model_info):
    local_path = os.path.dirname(model_info.local_model_path)
    local_models = os.listdir(local_path)
    return local_models


def get_remote_models(bucket, prefix):
    remote_models = []
    for object in bucket.objects.filter(Prefix=prefix):
        model_name = object.key.replace(prefix + '/', '')
        remote_models.append(model_name)
    return remote_models


def delete_extra_models(resource, bucket_name, prefix, local_models, remote_models):
    for remote_model in remote_models:
        if remote_model not in local_models:
            delete_key = os.path.join(prefix, remote_model)
            resource.Object(bucket_name=bucket_name, key=delete_key).delete()


def checkpoint_save_callback_function(operation_type, model_info):
    assert operation_type in ('load', 'save')
    prefix = get_prefix(model_info=model_info)
    local_models = get_local_models(model_info=model_info)
    remote_models = get_remote_models(bucket=custom_utils.s3.s3_resource.bucket, prefix=prefix)
    delete_extra_models(resource=custom_utils.s3.s3_resource.resource,
                        bucket_name=custom_utils.s3.s3_resource.bucket_name,
                        prefix=prefix, local_models=local_models, remote_models=remote_models)
    return model_info
