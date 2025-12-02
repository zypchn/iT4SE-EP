import numpy as np
import re
def readToMatrix(filename,name):
    f=open(filename,"rt")
#
    if name in ['pssm','psfm','pssmAndLabels','psfmAndLabels']:
        pssm=[]
        for j,line in enumerate(f.readlines()):
            if j > 2:
                line=line.strip()
                overall_vec = re.split(r" +",line)
                if len(overall_vec)<44:
                    break
                else:
                    pssm.append(overall_vec[:42])
        pssm=np.array(pssm)
        if name == 'pssm':
            return pssm[:, 2:22].astype(np.float64)  #
        elif name == 'psfm':
            return pssm[:, 22:42].astype(np.float64)  #
        elif name == 'pssmAndLabels':
            return pssm[:, 2:22].astype(np.float64), pssm[:, 1]  #
        elif name == 'psfmAndLabels':
            return pssm[:, 22:42].astype(np.float64), pssm[:, 1]  #

#
def autoNorm(matrix,name):
    if name=="pssm":
        matrix=matrix.astype(np.float64)
        matrix = 1 / (1 + np.exp(0 - matrix))
    elif name=="psfm":
        matrix=matrix.astype(np.float64)
        matrix = matrix / 100
    return matrix






