安装依赖
cd ~/w*
pip install -r requirements.txt





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


-------------------------------- YOLOX --------------------------------
parser.add_argument('-c', '--config', type=int, default=0,
                    choices=[0, 1, 2, 3],
                    help=textwrap.dedent('''\
                    select one of these config files:
                    0 -- yolox_s.py
                    1 -- yolox_m.py
                    2 -- yolox_l.py
                    3 -- yolox_x.py'''))

分布式训练
(有疑问请参考文档https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/06_distributed_training/cluster_quick_start_cn.html)
假设要运行2卡的任务，那么只需在命令行中执行:
python -m paddle.distributed.launch --gpus=0,1 train_yolox.py --config=0

从单机多卡到多机多卡训练，在代码上不需要做任何改动，只需再额外指定ips参数即可。其内容为多机的ip列表，命令如下所示：
python -m paddle.distributed.launch --ips="xx.xx.xx.xx,yy.yy.yy.yy" --gpus 0,1,2,3,4,5,6,7 train_yolox.py --config=0



训练
cd ~/w*
nohup python train_yolox.py --config=0 > nohup.log 2>&1 &


cd ~/w*
python train_yolox.py --config=0

cd ~/w*
python train_yolox.py --config=1

cd ~/w*
python train_yolox.py --config=2

cd ~/w*
python train_yolox.py --config=3



预测
cd ~/w*
python demo_yolox.py --config=0

cd ~/w*
python demo_yolox.py --config=1

cd ~/w*
python demo_yolox.py --config=2

cd ~/w*
python demo_yolox.py --config=3





验证
cd ~/w*
python eval_yolox.py --config=0

cd ~/w*
python eval_yolox.py --config=1

cd ~/w*
python eval_yolox.py --config=2

cd ~/w*
python eval_yolox.py --config=3



跑test_dev
cd ~/w*
python test_dev_yolox.py --config=0

cd ~/w*
python test_dev_yolox.py --config=1

cd ~/w*
python test_dev_yolox.py --config=2

cd ~/w*
python test_dev_yolox.py --config=3





导出后的预测

















