################################################################################
#
# Copyright (c) 2010 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this 
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
from __future__ import absolute_import
import string


mg4_template = string.Template("""\
#*********************************************************************
#                        MadGraph/MadEvent                           *
#                   http://madgraph.hep.uiuc.edu                     *
#                                                                    *
#                          proc_card.dat                             *
#*********************************************************************
#                                                                    *
#            This Files is generated by MADGRAPH 5                   *
#                                                                    *
# WARNING: This Files is generated for MADEVENT (compatibility issue)*
#          This files is NOT a valid MG4 proc_card.dat               *
#          Running this in MG4 will NEVER reproduce the result of MG5*
#                                                                    *
#*********************************************************************
#*********************************************************************
# Process(es) requested : mg2 input                                  *
#*********************************************************************
# Begin PROCESS # This is TAG. Do not modify this line
$process
done               # this tells MG there are no more procs
# End PROCESS  # This is TAG. Do not modify this line
#*********************************************************************
# Model information                                                  *
#*********************************************************************
# Begin MODEL  # This is TAG. Do not modify this line
$model
# End   MODEL  # This is TAG. Do not modify this line
#*********************************************************************
# Start multiparticle definitions                                    *
#*********************************************************************
# Begin MULTIPARTICLES # This is TAG. Do not modify this line
$multiparticle
# End  MULTIPARTICLES # This is TAG. Do not modify this line
""") 

process_template = string.Template("""\
$process           #Process
# Be carefull the coupling are here in MG5 convention
$coupling          
end_coup           # End the couplings input
""")


