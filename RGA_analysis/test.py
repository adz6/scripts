import RGA
import numpy
import math
import matplotlib.pyplot as plt

mass_2 = RGA.data("/Users/ziegler/scripts/RGA_analysis/RGA_spectrum_files/2019-07-17_gas_comp",2)
mass_3 = RGA.data("/Users/ziegler/scripts/RGA_analysis/RGA_spectrum_files/2019-07-17_gas_comp",3)
mass_4 = RGA.data("/Users/ziegler/scripts/RGA_analysis/RGA_spectrum_files/2019-07-17_gas_comp",4)
mass_6 = RGA.data("/Users/ziegler/scripts/RGA_analysis/RGA_spectrum_files/2019-07-17_gas_comp",6)

#for i in range(len(mass_2.massdata[:,2])):
    #mass_2.massdata[i,2] = math.log(mass_2.massdata[i,2])

#for i in range(len(mass_3.massdata[:,2])):
    #mass_3.massdata[i,2] = math.log(mass_3.massdata[i,2])

#for i in range(len(mass_4.massdata[:,2])):
    #mass_4.massdata[i,2] = math.log(mass_4.massdata[i,2])

#for i in range(len(mass_6.massdata[:,2])):
    #mass_6.massdata[i,2] = math.log(mass_6.massdata[i,2])

plt.plot(mass_2.massdata[:,0],mass_2.massdata[:,2], 'm', label = "Mass 2")
plt.plot(mass_3.massdata[:,0],mass_3.massdata[:,2],label = 'Mass 3')
plt.plot(mass_4.massdata[:,0],mass_4.massdata[:,2],'r', label = 'Mass 4')
plt.plot(mass_6.massdata[:,0],mass_6.massdata[:,2],'g', label = 'Mass 6')
plt.xlabel('Minutes after starting RGA')
plt.ylabel('RGA Partial Pressure')
plt.title('Tritium Getter Gas Composition 7/17/2019')
axes = plt.gca()
axes.set_yscale('log')
#axes.set_ylim([0,9.e-8])
plt.legend(loc = 'center left', bbox_to_anchor=(1,0.5))
plt.tight_layout(pad = 2)
plt.savefig('/Users/ziegler/plots/2019-07-17_gas_comp_logy.png')

plt.show()
