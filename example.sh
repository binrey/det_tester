python run.py \
    --grounds=/mnt/work_share/DWH/DataSets/Detection/11/1/labels/ann_det11.1_test.json \
    --predicts=/mnt/work_share/developers/rybin/models/ObjectDetection/yolov7-1280-cls17/test/annotation.json \
    --images=/mnt/work_share/DWH/DataSets/Detection/11/0/yolo/images/test \
    --weights=smb://10.10.50.26/work_share/developers/rybin/models/ObjectDetection/yolov7-1280-cls17/weights/best.pt \
    --project=Testing \
    --name=yolov7 \
    --tags="1280x1280 det11" \
    --s3config=clearml_config.yaml \
    --batch-size=9 \
    --nplots=5 \
    --data-stats \
    --tide \
    --false-negatives \
    --clearml \
    --exist-ok \
    