# QMCRT

This repositories contains the results from QMCRT and six optical modelling tools in Concentrating Solar Power (CSP) research.The six other modelling tools are SolTrace, Tonatiuh, Tracer, Solstice, Heliosim and SolarPILOT.<br>

The experiments consists of two rounds of tests.They are based on the related experiments described in (Wang et al., 2020) which aims at investigating the effect of sunshape and surface slope error.<br>

The content includs:<br>

* Buie_sunshape<br>
  * data(data files)<br>
  This folder contains six files. <br> The first half files are results of three tests in QMCRT for CSR(0.01,0.02,0.03)<br>The other three files are results of three tests in QMCRT with Tonatiuh's polynomial calibrated CSR(0.01,0.02,0.03)
  * plots(figures and comparison)<br>
    * energy<br>      The total energy received on the target in different optical modelling tools.
    * flux_diff<br>    The flux density distribution on the target in different optical modelling tools compared to Tonatiuh in csr=0.03.
    * flux_map <br>   The flux density distribution on the target in different optical modelling tools with CSR calibration.
    * radiance<br>    The normalised radiance distribution on the target in tests of QMCRT with different csr(0.01,0.02,0.03);<br> The normalised radiance distribution on the target in tests of QMCRT with polynomial calibrated CSR(0.01,0.02,0.03);<br>The normalised radiance distribution on the target in tests of QMCRT compare to other six optical modelling tools with polynomial calibrated CSR(0.01,0.02,0.03).
   
* Normal_slope_error<br>
  * data(data files)<br>
   This folder contains three files corresponding to experments with three slope errors(1,2,3mrad). 
  * plots(figures and comparison)<br>
    * energy<br>        The total energy received on the target in different optical modelling tools.
    * flux_diff<br>     The flux density distribution on the target in different optical modelling tools compared to Tonatiuh in slope error=1mrad.
    * flux_map <br>     The flux density distribution on the target in different optical modelling tools with different slope error.
    * radiance<br>      The normalised radiance distribution on the target under three slope errors(1,2,3mrad).
                                                                                   
