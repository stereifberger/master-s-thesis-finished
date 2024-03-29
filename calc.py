import numpy as np

# Calculus 1
## Symbols
term_num = 0

symb = {"DE": [term_num + 1],
        "BR": [term_num + 2],
        "NO": [term_num + 3],
        "TH": [term_num + 4],
        "OR": [term_num + 5],
        "AN": [term_num + 6],
        "FA": [term_num + 7]}

## Definition of rules 
def fa_e(ex1):
  return(rand_formula())

def no_e(ex1,ex2):
  return(symb["FALS"])

def no_i_a(ex1, ex2):
  return(symb["NOT"] + ex1)

def no_i_b(ex1, ex2):
  return(symb["NOT"] + ex2)

def an_i(ex1,ex2):
  return(ex1 + symb["AND"] + ex2)

def an_e_a(ex1,ex2):
  return(str(ex1) + str(ex2))

def an_e_b(ex1,ex2):
  return(str(ex1) + str(ex2))

def th_e(ex1,ex2):
  return(str(ex1) + str(ex2))

def th_i(ex1,ex2):
  return(str(ex1) + str(ex2))

def or_i(ex1,ex2):
  return(str(ex1) + str(ex2))

def or_e(ex1,ex2):
  return(str(ex1) + str(ex2))

## List the rules
calc_1 = [fals_e, not_e, not_i_a, not_i_b, and_i]
#calc_1 = [fals_e, not_e, not_i,
#          and_i, and_e_a, and_e_b,
#          then_e, then_i, or_i, or_e]

## Check whether rules is applicable to premises
def app_check(rule,
              term_num = int,
              prem = list):
  if rule == fa_e and prem == [symb["FA"]]:
      return True
  if rule == no_e and len(prem) == 2:
    if len(prem[0]) == 2 and len(prem[1]) == 1 and len(prem[0][0]) == 1:
      if prem[0][1] == prem[1] and prem[0][0] == symb["NO"]:
          return True
    if len(prem[1]) == 2 and len(prem[0]) == 1 and len(prem[1][0]) == 1:
      if prem[1][1] == prem[0] and prem[1][0] == symb["NO"]:
          return True
  if rule == no_i_a and prem[0] == symb["FA"] and prem[1] != symb["FA"]:
    return True
  if rule == no_i_b and prem[1] == symb["FA"] and prem[0] != symb["FA"]:
    return True
  if rule == an_i and len(prem) == 2:
      return True
  if rule == an_e_a or rule == an_e_b:
    if len(prem) == 1 and len(prem[0]) == 3:
      if prem[0][1] = symb["AN"]:
        return True
  if rule == th_e and len(prem) = 2:
    if 
    return True
  if rule == th_i and COND:
    return True
  if rule == or_i and COND:
    return True
  if rule == or_e and COND:
    return True
