# RFM_Helmholtz
汇总Random Feature Method求解Helmholtz方程的Code。

本篇主要解决两个问题：
1、RFM求解Helmholtz内问题
直接用Python运行，2d和3d

2、RFM求解Helmholtz外问题(将外问题转化为内问题)
直接用Python运行，在上边的基础上先加一步外问题求解

3、RFM最低消耗的算法
在RFM Efficient中，使用下面的命令调用
python RFM_Helmholtz/RFM_efficient/complex_1.py --basis 3000 --k 100 --multi 1.0 --Boundary_len 300
