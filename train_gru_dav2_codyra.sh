#!/bin/bash
#SBATCH --job-name=dav2-gru          # 作业名
#SBATCH --output=pure_%j.log        # 输出日志文件，%j是作业ID
#SBATCH --error=pure_%j.log          # 错误日志文件，%j是作业ID
#SBATCH --open-mode=append            # 立即打开日志文件
#SBATCH --ntasks=1                    # 任务数
#SBATCH --cpus-per-task=8            # 使用节点的全部64个CPU核心
#SBATCH --gres=gpu:1                  # 请求2张GPU
#SBATCH --time=7-00:00:00             # 设置作业的最大运行时间为7天
#SBATCH --mem=20G                     # 分配内存大小
#SBATCH --nodelist=hades             # 指定运行在erinyes节点上

# 在脚本开始处添加
export PYTHONUNBUFFERED=1             # 禁用Python输出缓冲
stdbuf -oL -eL                        # 禁用标准输出和错误的行缓冲

# 进入提交目录
cd $SLURM_SUBMIT_DIR

# 初始化并激活conda环境
# 不要直接使用conda init (它是用于修改shell配置文件的)
# 而是直接source conda.sh文件
source /home/ywan0794/miniconda3/etc/profile.d/conda.sh

# 激活gpupytorch环境
conda activate stereoadapter

# 确认conda环境已激活
echo "Activated conda environment: $CONDA_DEFAULT_ENV"

# 设置Python路径，使其能够找到Depth-Anything-V2中的模块
export PYTHONPATH=$PYTHONPATH:$SLURM_SUBMIT_DIR
echo "Added current directory to PYTHONPATH: $PYTHONPATH"

# 显示GPU信息
echo "=== GPU ==="
nvidia-smi

# 检查CUDA是否可用
python -c "import torch; print('CUDA是否可用:', torch.cuda.is_available()); print('可用的GPU数量:', torch.cuda.device_count()); print('当前GPU型号:', torch.cuda.get_device_name(0))" 2>&1 | tee cuda_check.log

# 创建日志目录
LOG_DIR="./melo_logs/flsea-all"
mkdir -p $LOG_DIR

# 获取当前时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 打印一些信息到日志
echo "Starting GPU sleep job on $(hostname) at $(date)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

# 运行训练脚本并保存日志
python train_dist_2.py \
    --name GRU-DAv2-codyra-tartanair-occ \
    --exp_opts options/TiO-Depth/train/gru-dav2_codyra-tartanair.yaml \
    --batch_size 4 \
    --metric_source rawdepth sdepth \
    --save_freq 5 \
    --visual_freq 10 \
    --is_codyra True \
    --step_epochs 20 30 \
    2>&1 | tee "${LOG_DIR}/train_dav2_flsea_${TIMESTAMP}.log"

echo "Sleep job completed at $(date)"