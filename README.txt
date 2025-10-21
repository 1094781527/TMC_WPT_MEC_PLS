MEC-WPT-PLS 算法对比平台

一个集成了9种深度强化学习算法的移动边缘计算仿真平台。

快速开始

安装依赖：
pip install torch numpy scipy matplotlib pandas tensorflow

运行程序：
python all_comparison.py

包含算法：
- OURs（我们的算法）
- PG（策略梯度）
- AC（Actor-Critic）
- DDPG（深度确定性策略梯度）
- DROO（深度强化学习优化）
- H2AC（混合Actor-Critic）
- A3C（异步优势Actor-Critic）
- PPO（近端策略优化）
- SAC（软Actor-Critic）

基本用法：
使用默认参数运行：python all_comparison.py
自定义训练步数：python all_comparison.py --time-steps 10000
指定数据文件：python all_comparison.py --file-path your_data.mat

输出结果：
- 控制台性能对比报告
- 算法奖励曲线图

参数配置：
在代码中修改Config类调整系统参数：
- N=15（设备数量）
- 学习率、批大小等超参数

详细文档请查看代码注释