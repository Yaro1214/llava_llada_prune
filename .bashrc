export PIP_INDEX_URL=https://pypi.org/simple
export PYTHONPATH=/home/ma-user/code/zyj/LLaDA-V/eval
export PYTHONPATH=/home/ma-user/code/zyj/LLaDA-V/train

proxy_on
conda activate llada_zyj
echo "[zyj_bash] Loaded private environment config."

git config core.sshCommand "ssh -i ../.ssh/key -F /dev/null"#需要-F忽略全局设置
git config  user.name "yaro1214"
git config user.email "15934071030@163.com"