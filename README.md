# QMCRT

Two rounds of tests were progressed to compare and verify the results from QMCRT and six optical modelling tools in Concentrating Solar Power (CSP) research, including: SolTrace, Tonatiuh, Tracer, Solstice, Heliosim and SolarPILOT. This repositories contains the results of QMCRT in two rounds tests.<br>

The tests are based on the related experiments as described in (Wang et al., 2020) to investigate the effect of sun shape and surface slope error and verify the correctness of the QMCRT.<br>

The details are recorded and shared in this Github repository for people who are interested. The content includs:<br>

* Buie_sunshape<br>
  * data(data files)<br>
  This folder contains three files. The three files are results of three tests in QMCRT with Tonatiuh's polynomial calibration for CSR(0.01,0.02,0.03) .
  * plots(figures and comparison)<br>
    * energy<br>      The total energy power on the target in different optical modelling tools.
    * flux_diff<br>    The flux density distribution on the target in differnet optical modelling tools compared with Tonatiuh in csr=0.03.
    * flux_map <br>   The flux density distribution on the target in differnet optical modelling tools showed in flux map with CSR calibration.
    * radiance<br>    The normalised radiance distribution on the target in tests of QMCRT with different csr; The normalised radiance distribution on the target in tests of QMCRT with polynomial calibration for CSR(0.01,0.02,0.03);The normalised radiance distribution on the target in tests of QMCRT compare with other six optical modelling tools with polynomial calibration for CSR(0.01,0.02,0.03)
   
* Normal_slope_error<br>
  * data(data files)<br>
   This folder contains three files. The three files are results of three tests in QMCRT with different slope error(1,2,3mrad) .
  * plots(figures and comparison)<br>
    * energy<br>        The total energy power on the target in different optical modelling tools.
    * flux_diff<br>     The flux density distribution on the target in differnet optical modelling tools compared with Tonatiuh in slope error=1mrad.
    * flux_map <br>     The flux density distribution on the target in differnet optical modelling tools showed in flux map with different slope error.
    * radiance<br>      The normalised radiance distribution on the target in tests of normal slope error(1,2,3mrad).
                                                                                   
