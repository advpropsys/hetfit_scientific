import numpy as np
import seaborn as sns
import pandas as pd

def B_field_norm(Bmax:float,L:float,k:int=16,plot=True) -> np.array:
    """ Returns vec B_z for MS config

    Args:
        Bmax (any): maximum B in thruster
        L - channel length
        k - magnetic field profile number
    """
    z = np.linspace(0,L*1.4,200)
    B = Bmax * np.exp(-k * (z/(1.2*L) - 1)**2)
    if plot:
        sns.lineplot(x=z,y=B)
    return B

def PUdesign(P:float,U:float) -> pd.DataFrame:
    """Computes design via numerical model, uses fits from PINNs

    Args:
        P (float): _description_
        U (float): _description_

    Returns:
        _type_: _description_
    """
    d = np.sqrt(P/(635*U))
    h = 0.245*d
    m_a = 0.0025*h*d
    T = 890 * m_a * np.sqrt(U)
    j = P/(np.pi*d*h)
    Isp = T/(m_a*9.81) 
    nu_t = T*Isp*9.81/(2*P)
    df = pd.DataFrame([[d,h,m_a,T,j,nu_t,Isp]],columns=['d','h','m_a','T','j','nu_t','Isp'])
    g = sns.barplot(df,facecolor='gray')
    g.set_yscale("log")
    return df
    
def cathode_erosion():
    pass