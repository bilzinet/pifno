"""
Created on Sun Nov 15 2023
@author: Bilal Thonnam Thodi (btt1@nyu.edu)

Classical traffic state estimation problem using data-assimilation technique:
    Traffic flow model: Lighthill-Withams-Richards
    Transition model (process model): Discrete Godunov numerical scheme
    Assimilation method: Extended Kalman Filter
    Measurement type: Initial conditions and vehicle trajectory measurements 
    Reference paper: Localized Extended Kalman Filter for Scalable Real-Time Traffic State Estimation
        (https://ieeexplore.ieee.org/abstract/document/6105572/)

"""