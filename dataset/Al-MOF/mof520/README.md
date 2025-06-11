## Update of the cif file

The original cif file (i.e, RSM4131_corrected_without_charge.cif) does not contain the DDEC charge and are used to calculate the results about binding energy and isotherms of MOF-520 in the paper. In the paper, you may notice the simulated isotherm of MOF-520 is slightly lower than the experimental isotherm in low pressure region and its predicted binding energy is slightly lower than DFT binding energy. It is caused by the wrong cif without charge information. Because this error is lower than the error bar so we do not find it in publication of the paper. 

But if you use the correct cif (i.e, RSM4131_corrected_charge.cif) to simulate the result, you can find that our refined force field can reproduce more consistent result with DFT and experimental isotherm of MOF-520, which actually support the effectiveness of the refined force field. 
