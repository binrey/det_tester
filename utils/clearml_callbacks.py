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
