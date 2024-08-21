Here I construct the initial force field Lenard-Jone potential and Coulume potential from UFF and MIL-120 from aiida-plugin

The force field file in RASPA: epsilon units (Kelvin); sigma(Angstrom)

Al\_     LENNARD\_JONES   254.152 4.0082

C\_      LENNARD\_JONES   52.8435 3.4309

H\_      LENNARD\_JONES   22.1439 2.5711

O\_      LENNARD\_JONES   30.1963 3.1181

O\_co2   LENNARD\_JONES   79.0    3.05

C\_co2   LENNARD\_JONES   27.0    2.80

The force field file in OpenMM: epsilon units (KJ/mol); sigma(nm)

Al\_     LENNARD\_JONES   2.1019    0.40082

C\_      LENNARD\_JONES   0.4400    0.34309

H\_      LENNARD\_JONES   0.1841    0.25711

O\_      LENNARD\_JONES   0.2513    0.31181

O\_co2   LENNARD\_JONES   0.6577    0.30500

C\_co2   LENNARD\_JONES   0.2249    0.28000

Some prepared files to run the simulation: (I put all of them in data directory)

"""

It is not easy to generate these files. Bullshit PDB and xml files. I save some childish methods in refine.ipynb

"""

Structural file of MOF:

atoms in pdb file should have a special name

gas.pdb: CO2 molecule predicted by UFF force field

MIL-120.pdb: The rigid framework

A good force field:

custom\_force\_field.xml, containing charges information from CIF file