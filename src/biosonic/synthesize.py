def ripple_sound():
    """
    Generate a ripple signal ad defined in [2]:

    Amplitude envelope of each frequency band:
    .. math::
        S_i(t,f)=A_icos(2\pi\omega_{t,i}t+2\pi\omega_{f,i}f+\phi_i)
    
    Spectrogram: 
    .. math::
        S(t,f)=A_0+\sum_i S_i(t,f)

    2. Singh NC, Theunissen FE. 2003 Modulation spectra of natural sounds and ethological theories of auditory processing. 
    The Journal of the Acoustical Society of America 114, 3394–3411. (doi:10.1121/1.1624067) 
    
    """
    pass