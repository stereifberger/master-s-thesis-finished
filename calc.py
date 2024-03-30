# Import libraries
import numpy as np
import random as rd
from random import seed, randint, choice, shuffle

# Intuitionistic propositional logic
## Symbols
t_nu = 9

## Generate random well-formed-formulas
def rd_f(t_nu):
    return rd.randint(1,t_nu)

## Define numerical values for symbols:
symb = {"DE": [t_nu + 1],                               # DERIVES
        "BR": [t_nu + 2],                               # BRACKET
        "NO": [t_nu + 3],                               # NOT
        "TH": [t_nu + 4],                               # THEN
        "OR": [t_nu + 5],                               # OR
        "AN": [t_nu + 6],                               # AND
        "FA": [t_nu + 7]}                               # FALSUM

## Definition of rules of intuitionistic propositional logic (INPL):
fa_e = lambda prem: rd_f()                              # FALSUM_ELIMINATION
no_e = lambda prem: symb["FA"]                          # NOT_ELIMINATION
n_ia = lambda prem: symb["NO"] + prem[1]                # NOT_INTRODUCTION_A
n_ib = lambda prem: symb["NO"] + prem[0]                # NOT_INTRODUCTION_B 
an_i = lambda prem: prem[0]    + symb["AN"] + prem[1]   # AND_INTRODUCTION
a_ea = lambda prem: prem[0]                             # AND_ELIMINATION_A
a_eb = lambda prem: prem[1]                             # AND_ELIMINATION_B
t_ea = lambda prem: prem[0][2]                          # THEN_ELIMINATION_A
t_eb = lambda prem: prem[1][2]                          # THEN_ELIMINATION_B
th_i = lambda prem: prem[0]    + symb["TH"] + prem[1]   # THEN_INTRODUCTION
o_ia = lambda prem: prem[0]    + symb["OR"] + rd_f()    # OR_INTRODUCTION_A
o_ib = lambda prem: rd_d()     + symb["OR"] + prem[0]   # OR_INTRODUCTION_B

## List the rules of INPL
inpl = [fa_e, no_e, n_ia, n_ib, an_i, a_ea,
       a_eb, t_ea, t_eb, th_i, o_ia, o_ib]

## Define check for applicability of rule to premises via syntactic conditions:
def check(rule, prem = list):
    if len(prem) == 1:
        if rule == fa_e:                                # FALSUM_ELIMINATION
            if symb["FA"] == prem[0]:
                return True
        if rule == a_ea or rule == a_eb:                # AND_ELIMINATION
            if len(prem[0]) == 3:
                if prem[0][1] == symb["AN"]:
                    return True
        if rule == o_ia or rule == o_ib:                # OR_INTRODUCTION_A and B
            return True
    if len(prem) == 2:
        if symb["FA"] == prem[0] or symb["FA"] == prem[1]:
            if rule == n_ia:                            # NOT_INTRODUCTION_A
                if prem[0] == symb["FA"]:
                    if prem[1] != symb["FA"]:
                        return True
            if rule == n_ib                             # NOT_INTRODUCTION_B
                if prem[1] == symb["FA"]:
                    if prem[0] != symb["FA"]:
                return True
        else:
            if rule == no_e:                            # NOT_ELIMINATION
                if len(prem[0]) == 2:
                    if len(prem[1]) == 1:
                        if len(prem[0][0]) == 1:
                            if prem[0][1] == prem[1]:
                                if prem[0][0] == symb["NO"]:
                                    return True
                if len(prem[1]) == 2:
                    if len(prem[0]) == 1:
                        if len(prem[1][0]) == 1:
                            if prem[1][1] == prem[0]:
                                if prem[1][0] == symb["NO"]:
                                    return True
            if rule == an_i:                            # AND_INTRODUCTION
                return True
            if rule == t_ea:                            # THEN_ELIMINATION_A
                if len(prem[0]) == 3:
                    if len(prem[1]) == 1:
                        if prem[0][1] == symb["TH"]:
                            if prem[0][0] == prem[1]:
                                return True
            if rule == t_eb:                            # THEN_ELIMINATION_B
                if len(prem[1]) == 3:
                    if prem[1][1] == symb["TH"]:
                        if prem[1][0] == prem[0]:
                            return True
            if rule == th_i:                            # THEN_INTRODUCTION
                return True