from clearml import Task

from utils.s3 import initialize_s3_resource


def initialize_clearml_task(clearml_arguments, s3_config, remotely=False, no_queue=False,
                            task_type=Task.TaskTypes.application):
    global clearml_task
    global clearml_logger
    clearml_task = Task.init(project_name=clearml_arguments.project_name, task_name=clearml_arguments.task_name,
                             task_type=task_type, tags=clearml_arguments.tag, reuse_last_task_id=False,
                             auto_resource_monitoring=True,
                             auto_connect_streams=False,
                             auto_connect_frameworks=False,
                             auto_connect_arg_parser=True)
    clearml_logger = clearml_task.get_logger()
    initialize_s3_resource(s3_config)
    if remotely:
        clearml_task.set_base_docker(docker_image=clearml_arguments.docker_image,
                                     docker_arguments=clearml_arguments.docker_args)
        if no_queue:
            clearml_task.execute_remotely()
        else:
            clearml_task.execute_remotely(queue_name=clearml_arguments.queue_name)
