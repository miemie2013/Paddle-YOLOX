安装依赖
cd ~/w*
pip install -r requirements.txt


# 解压预训练模型
nvidia-smi
cd ~/w*
cp ../data/data64338/dygraph_yolov4_2x.pdparams ./dygraph_yolov4_2x.pdparams

(或者下载yolov4.pt
链接：https://pan.baidu.com/s/152poRrQW9Na_C8rkhNEh3g
提取码：09ou
将它放在项目根目录下。然后运行1_yolov4_2x_2paddle.py
)



下载预训练模型ppyolo.pdparams
cd ~/w*
wget https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams
python 1_ppyolo_2x_2paddle.py
rm -f ppyolo.pdparams



下载预训练模型ResNet50_vd_ssld_pretrained.tar
cd ~/w*
wget https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_ssld_pretrained.tar
tar -xf ResNet50_vd_ssld_pretrained.tar
python 1_r50vd_ssld_2paddle.py
rm -f ResNet50_vd_ssld_pretrained.tar
rm -rf ResNet50_vd_ssld_pretrained



下载预训练模型ResNet50_cos_pretrained.tar
cd ~/w*
wget https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_cos_pretrained.tar
tar -xf ResNet50_cos_pretrained.tar
python 1_r50vb_cos_2paddle.py
rm -f ResNet50_cos_pretrained.tar
rm -rf ResNet50_cos_pretrained



下载预训练模型ppyolo_r18vd.pdparams
cd ~/w*
wget https://paddlemodels.bj.bcebos.com/object_detection/ppyolo_r18vd.pdparams
python 1_ppyolo_r18vd_2paddle.py
rm -f ppyolo_r18vd.pdparams






# 解压COCO2017数据集
nvidia-smi
cd ~
cd data
cd data7122
unzip ann*.zip
unzip val*.zip
unzip tes*.zip
unzip image_info*.zip
unzip train*.zip
cd ~/w*



# 解压voc数据集
nvidia-smi
cd ~
cd data
cd data4379
unzip pascalvoc.zip
cd ~/w*



cd ~/w*
rm -rf log*.txt


cd ~/w*
unzip P*.zip


-------------------------------- PPYOLO --------------------------------
parser.add_argument('-c', '--config', type=int, default=0,
                    choices=[0, 1, 2, 3, 4, 5],
                    help=textwrap.dedent('''\
                    select one of these config files:
                    0 -- ppyolo_2x.py
                    1 -- yolov4_2x.py
                    2 -- ppyolo_r18vd.py
                    3 -- ppyolo_mobilenet_v3_large.py
                    4 -- ppyolo_mobilenet_v3_small.py
                    5 -- ppyolo_mdf_2x.py
                    6 -- ppyolo_large_2x.py'''))

训练
cd ~/w*
nohup python train_yolo.py --config=0 > nohup.log 2>&1 &


cd ~/w*
python train_yolo.py --config=0

cd ~/w*
python train_yolo.py --config=1

cd ~/w*
python train_yolo.py --config=2

cd ~/w*
python train_yolo.py --config=3

cd ~/w*
python train_yolo.py --config=4

cd ~/w*
python train_yolo.py --config=5

cd ~/w*
python train_yolo.py --config=6




预测
cd ~/w*
python demo_yolo.py --config=0

cd ~/w*
python demo_yolo.py --config=1

cd ~/w*
python demo_yolo.py --config=2

cd ~/w*
python demo_yolo.py --config=3

cd ~/w*
python demo_yolo.py --config=4

cd ~/w*
python demo_yolo.py --config=5

cd ~/w*
python demo_yolo.py --config=6





验证
cd ~/w*
python eval_yolo.py --config=0

cd ~/w*
python eval_yolo.py --config=1

cd ~/w*
python eval_yolo.py --config=2

cd ~/w*
python eval_yolo.py --config=3

cd ~/w*
python eval_yolo.py --config=4

cd ~/w*
python eval_yolo.py --config=5

cd ~/w*
python eval_yolo.py --config=6




跑test_dev
cd ~/w*
python test_dev_yolo.py --config=0

cd ~/w*
python test_dev_yolo.py --config=1

cd ~/w*
python test_dev_yolo.py --config=2

cd ~/w*
python test_dev_yolo.py --config=3

cd ~/w*
python test_dev_yolo.py --config=4

cd ~/w*
python test_dev_yolo.py --config=5

cd ~/w*
python test_dev_yolo.py --config=6







导出后的预测
python tools/export_model_yolo.py  -c configs/ppyolo/ppyolo.yml --output_dir=./inference_model -o weights=output/ppyolo/model_final
python deploy/python/infer.py --model_dir=inference_model/ppyolo_2x --image_file=./images/test/000000000019.jpg --use_gpu=True

python deploy/python/infer.py --model_dir=inference_model/ppyolo --image_file=./images/test/000000000019.jpg --use_gpu=True


TensrRT FP32
python tools/export_model.py  -c configs/ppyolo/ppyolo.yml --output_dir=./inference_model -o weights=output/ppyolo/model_final --exclude_nms
CUDA_VISIBLE_DEVICES=0
python deploy/python/infer.py --model_dir=inference_model/ppyolo_2x --image_file=./images/test/000000000019.jpg --use_gpu=True --run_benchmark=True


TensrRT FP16
python tools/export_model.py  -c configs/ppyolo/ppyolo.yml --output_dir=./inference_model -o weights=output/ppyolo/model_final --exclude_nms
CUDA_VISIBLE_DEVICES=0
python deploy/python/infer.py --model_dir=inference_model/ppyolo_2x --image_file=./images/test/000000000019.jpg --use_gpu=True --run_benchmark=True --run_mode=trt_fp16

















