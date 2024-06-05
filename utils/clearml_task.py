from clearml import Task


def clearml_init_task(project_name, task_name, tags, description=None): #TODO
    global clearml_logger
    task = Task.init(project_name=project_name, 
                             task_name=task_name,
                             task_type=Task.TaskTypes.testing, 
                             tags=tags, 
                             reuse_last_task_id=False,
                             auto_resource_monitoring=True,
                             auto_connect_streams=False,
                             auto_connect_frameworks=False,
                             auto_connect_arg_parser=False)
    clearml_logger = task.get_logger()
