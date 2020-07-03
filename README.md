# QMCRT

Two rounds of tests were progressed to compare and verify the results from QMCRT and six optical modelling tools in Concentrating Solar Power (CSP) research, including: SolTrace, Tonatiuh, Tracer, Solstice, Heliosim and SolarPILOT. This repositories contains the results of QMCRT in two rounds tests.<br>

The tests are based on the related experiments as described in (Wang et al., 2020) to investigate the effect of sun shape and surface slope error and verify the correctness of the QMCRT.<br>

The details are recorded and shared in this Github repository for people who are interested. The content includs:<br>

## Buie_sunshape
### data(data files)
  This folder contains three files. The three files are results of three tests in QMCRT with Tonatiuh's polynomial calibration for CSR(0.01,0.02,0.03) .
### plots(figures and comparison)
   #### energy<br>       /    The total energy power on the target in different optical modelling tools.
   #### flux_diff<br>    /    The flux density distribution on the target in differnet optical modelling tools compared with Tonatiuh in csr=0.03.
   #### flux_map<br>     /    The flux density distribution on the target in differnet optical modelling tools showed in flux map with CSR calibration.
   #### radiance<br>     /     The normalised radiance distribution on the target          
   
## Normal_slope_error
### data(data files)
   三个文件，分别为normal slope error为1 mrad，2 mrad ，3 mrad 的实验结果
### plots(figures and comparison)
   #### energy <br>      /    不同仿真工具接收器接受的总能量
   #### flux_diff<br>    /     normal slope error=1 mrad 的不同仿真工具接收器接收能量密度的比较
   #### flux_map <br>    /     不同仿真工具在normal slope error为1 mrad，2 mrad ，3 mrad 的接收到的能量密度的热力图
   #### radiance <br>    /     normal slope error（QMCRT）的normalised radiance 结果；不同仿真工具（normal slope error）normalised radiance 结果
                                                                                   
