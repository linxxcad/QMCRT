# QMCRT
说明： 
    本文件夹存放本次围绕QMCRT实验特定镜场的结果。
    实验分为buie sunshape 和 normal slope error 两组实验。
    
    
    一级目录--result
           二级目录--Buie_sunshape
                           三级目录--data（实验CSV结果文件）
                                          六个文件，分别为csr=0.01,0.02,0.03的实验结果以及对csr=0.01,0.02,0.03做修正的实验结果（修正方式参考Tonatiuh）
                           三级目录--plots（实验可视化图形结果文件）
                                           四级目录--energy  /    不同仿真工具接收器接受的总能量
                                           四级目录--flux_diff    /     csr=0.01 的不同仿真工具接收器接收能量密度的比较
                                           四级目录--flux_map     /     不同仿真工具在csr为0.01，0.02 ，0.03的接收到的能量密度的热力图(做过csr修正,修正方式参考Tonatiuh）
                                           四级目录--radiance    /    Buie sunshape(QMCRT）的normalised radiance 结果；
                                                                                  Buie sunshape(QMCRT）的normalised radiance 结果(做过csr修正,修正方式参考Tonatiuh）                                                                                   
           二级目录--Normal_slope_error
                           三级目录--data（实验CSV结果文件）
                                           三个文件，分别为normal slope error为1 mrad，2 mrad ，3 mrad 的实验结果
                           三级目录--plots（实验可视化图形结果文件）
                                           四级目录--energy  /    不同仿真工具接收器接受的总能量
                                           四级目录--flux_diff    /     normal slope error=1 mrad 的不同仿真工具接收器接收能量密度的比较
                                           四级目录--flux_map     /     不同仿真工具在normal slope error为1 mrad，2 mrad ，3 mrad 的接收到的能量密度的热力图
                                           四级目录--radiance    /     normal slope error（QMCRT）的normalised radiance 结果；
                                                                                   不同仿真工具（normal slope error）normalised radiance 结果
                                                                                   
