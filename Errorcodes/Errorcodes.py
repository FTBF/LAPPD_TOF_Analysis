from enum import Enum

class Station_Error(Enum):
    
    # peak time find errors
    PEAK_FIND_FAIL = 6010
    # peak_ch identification errors
    IMPROPER_PEAK_CH = 7010

    # Sync channel fit errors
    SIN_FIT_VALUE_ERROR = 8010
    SIN_FIT_FAIL = 8020
    NOT_TWO_PEAKS = 8030

    # Position reconstruction errors
    POS_RECON_FAIL = 9010