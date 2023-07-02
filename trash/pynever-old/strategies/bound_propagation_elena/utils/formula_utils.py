from venus.utils.formula import ConjFormula, ENextFormula, ANextFormula, DisjFormula


def get_eg_formula(n, phi, psi):
    formula = psi
    for i in range(n):
        formula = ConjFormula(phi, ENextFormula(1, formula))
    return formula


def get_ag_formula(n, phi, psi):
    formula = psi
    for i in range(n):
        formula = ConjFormula(phi, ANextFormula(1, formula))
    return formula


def get_ef_formula(n, phi, psi):
    formula = psi
    for i in range(n):
        formula = DisjFormula(phi, ENextFormula(1,formula))
    return formula


# obsolete with the new NAry Conj and Disj classes
def get_conjunction(atoms):
    formula = atoms[0]
    for i in range(1,len(atoms)):
        formula = ConjFormula(atoms[i], formula)
    return formula


def get_disjunction(atoms):
    formula = atoms[0]
    for i in range(1,len(atoms)):
        formula = DisjFormula(atoms[i], formula)
    return formula