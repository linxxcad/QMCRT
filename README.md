# QMCRT

Two rounds of tests were progressed to compare and verify the results from QMCRT and six optical modelling tools in Concentrating Solar Power (CSP) research, including: SolTrace, Tonatiuh, Tracer, Solstice, Heliosim and SolarPILOT. This repositories contains the results of QMCRT in two rounds tests.

The tests are based on the related experiments as described in (Wang et al., 2020) to investigate the effect of sun shape and surface slope error and verify the correctness of the QMCRT.

The details are recorded and shared in this Github repository for people who are interested. The content includs:.

Buie_sunshape
     --data
           This folder contains six files. There are three files results of three CSR(0.01,0.02,0.03) tests in QMCRT ,other three files are results of CSR(0.01,0.02,0.03) with the polynomial calibration in QMCRT.
     --plots
           --energy  /    differen不同仿真工具接收器接受的总能量
           --flux_diff    /     csr=0.03 的不同仿真工具接收器接收能量密度的比较
           --flux_map     /     不同仿真工具在csr为0.01，0.02 ，0.03的接收到的能量密度的热力图(做过csr修正,修正方式参考Tonatiuh）
           --radiance    /    Buie sunshape(QMCRT）的normalised radiance 结果；
                                                                                  Buie sunshape(QMCRT）的normalised radiance 结果(做过csr修正,修正方式参考Tonatiuh）                                                                                   
Normal_slope_error
                          --data
                                           三个文件，分别为normal slope error为1 mrad，2 mrad ，3 mrad 的实验结果
                          --plots
                                           四级目录--energy  /    不同仿真工具接收器接受的总能量
                                           四级目录--flux_diff    /     normal slope error=1 mrad 的不同仿真工具接收器接收能量密度的比较
                                           四级目录--flux_map     /     不同仿真工具在normal slope error为1 mrad，2 mrad ，3 mrad 的接收到的能量密度的热力图
                                           四级目录--radiance    /     normal slope error（QMCRT）的normalised radiance 结果；
                                                                                   不同仿真工具（normal slope error）normalised radiance 结果
                                                                                   
