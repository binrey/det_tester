from clearml import Task, OutputModel
from utils.common import get_data_from_yaml

from utils.s3 import initialize_s3_resource


def initialize_clearml_task(project_name, task_name, tags, task_type):
    global clearml_task
    global clearml_logger
    clearml_task = Task.init(project_name=project_name, task_name=task_name,
                             task_type=task_type, tags=tags, reuse_last_task_id=False,
                             auto_resource_monitoring=True,
                             auto_connect_streams=False,
                             auto_connect_frameworks=False,
                             auto_connect_arg_parser=True)
    clearml_logger = clearml_task.get_logger()
    # initialize_s3_resource(s3_config)

    
def clearml_init_task(opt):
    # s3config = get_data_from_yaml(opt.s3config)
    initialize_clearml_task(project_name=opt.project, 
                            task_name=opt.name, 
                            tags=opt.tags, 
                            #s3_config=s3config, 
                            task_type=Task.TaskTypes.testing)
    # clearml_task.connect(s3config.to_dict())
    clearml_task.connect(opt)
    if opt.weights is not None:
        output_model = OutputModel(task=clearml_task)
        output_model.update_weights(register_uri=opt.weights)

