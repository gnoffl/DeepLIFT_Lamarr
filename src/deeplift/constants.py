# Description: Constants used in Deeplift

# prevents numerical instability in non-linear layers since values need to be divided there. Sets lower bound for
# absolute value of divisor.
NUMERICAL_INSTABILITY_THRESHOLD = 1e-7
# threshold under which a value is considered to be zero. Important, since for the propagation of positive and negative
# multipliers, zero values in the inputs are treated differently than positive and negative values.
ZERO_MASK_THRESHOLD = 1e-7
