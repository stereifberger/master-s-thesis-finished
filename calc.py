import numpy as np
import random as rd
from random import seed, randint, choice, shuffle

# Calculus 1
## Symbols
term_num = 9

def rd_f(term_num):
  return rd.randint(1,term_num)

symb = {"DE": [term_num + 1],
        "BR": [term_num + 2],
        "NO": [term_num + 3],
        "TH": [term_num + 4],
        "OR": [term_num + 5],
        "AN": [term_num + 6],
        "FA": [term_num + 7]}

## Definition of rules 
fa_e   = lambda prem: rd_f()
no_e   = lambda prem: symb["FA"]
no_i_a = lambda prem: symb["NO"] + prem[1]
no_i_b = lambda prem: symb["NO"] + prem[0]
an_i   = lambda prem: prem[0]    + symb["AN"] + prem[1]
an_e_a = lambda prem: prem[0]
an_e_b = lambda prem: prem[1]
th_e_a = lambda prem: prem[0][2]
th_e_b = lambda prem: prem[1][2]
th_i   = lambda prem: prem[0]    + symb["TH"] + prem[1]
or_i_a = lambda prem: prem[0]    + symb["OR"] + rd_f()
or_i_b = lambda prem: rd_d()     + symb["OR"] + prem[0] 

## List the rules
ipl = [fa_e, no_e, no_i_a, no_i_b, an_i, an_e_a,
       an_e_b, th_e_a, th_e_b, th_i, or_i_a, or_i_b]

## Check whether rules is applicable to premises
def app_check(rule,
              term_num = int,
              prem = list):
  if len(prem) == 1:
    if rule == fa_e and symb["FA"] == prem[0]:
      return True
    if rule == an_e_a or rule == an_e_b:
      if len(prem[0]) == 3:
        if prem[0][1] == symb["AN"]:
          return True
    if rule == or_i_a or rule == or_i_b:
      return True
  if len(prem) == 2:
    if symb["FA"] == prem[0] or symb["FA"] == prem[1]:
      print("a")
      if rule == no_i_a and prem[0] == symb["FA"] and prem[1] != symb["FA"]:
        return True
      if rule == no_i_b and prem[1] == symb["FA"] and prem[0] != symb["FA"]:
        return True
    else:
      if rule == no_e:
        if len(prem[0]) == 2 and len(prem[1]) == 1 and len(prem[0][0]) == 1:
          if prem[0][1] == prem[1] and prem[0][0] == symb["NO"]:
              return True
        if len(prem[1]) == 2 and len(prem[0]) == 1 and len(prem[1][0]) == 1:
          if prem[1][1] == prem[0] and prem[1][0] == symb["NO"]:
              return True
      if rule == an_i:
          return True
      if rule == th_e_a:
        if len(prem[0]) == 3 and len(prem[1]) == 1:
          if prem[0][1] == symb["TH"] and prem[0][0] == prem[1]:
            return True
      if rule == th_e_b:
        if len(prem[1]) == 3:
          if prem[1][1] == symb["TH"] and prem[1][0] == prem[0]:
            return True
      if rule == th_i:
        return True
