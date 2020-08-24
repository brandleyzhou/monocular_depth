import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator
def lin_interp(shape,xyd):

    # taken from https://github.com/hunse/kitti
    m,n = shape
    ij,d = xyd[:,1::-1],xyd[:,2]
    f = LinearNDInterpolator(ij,d,fill_value=0)
    J,I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disp = f(IJ).reshape(shape)
    
    return disp

def main():
    lin_interp()
if  __name__ == '__main__':
    main()
