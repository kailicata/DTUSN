NEURON {
    SUFFIX ca
    USEION ca READ eca WRITE ica
    RANGE gca, ica
}

PARAMETER {
    gca = 0.0001 (S/cm2)  : Maximum conductance
    eca = 120 (mV)        : Reversal potential for Ca
}

ASSIGNED {
    v (mV)
    ica (mA/cm2)
}

BREAKPOINT {
    ica = gca * (v - eca)
}
