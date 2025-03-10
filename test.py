import swanlab

# 创建1个实验
run = swanlab.init()

for i in range(10):
  # 将指标loss，上传到这个实验中
  run.log({"loss": i})