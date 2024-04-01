# Import libraries
from libs import *

t_nu = 9

"""When used with the Derivation method it returns:
    - X: The input to a network of the form [PREMISES, DELIMITER, CONCLUSION]
    - y: The output to a network of the form
         [PREMISES, DERIVATION STEP 1,.recursion_task.., DERIVATION STEP N, CONCLUSION]"""

# Define the data generation process. If the nettworks are trained on the class of all 
# correct derivations then I only need the input data. Also in a future version at each
# iteratio of applying rules a new input could be defined on it, making the function faster.
def dat_b(itra = list, s_si = int):
    t_nu = 9
    inpt, outp = [], []
    gbar = tqdm(total=s_si, desc="Generating example derivations")
    while len(inpt) < s_si:
        drva = []
        for p in range(rd.randint(1, 3)):
            drva.append(calc.rd_f(t_nu))
        in_i = copy.copy(drva)
        for it in range(randint(itra[0], itra[1])):
            if len(drva) >= 2:
                pr_n = randint(1, 2)
            else:
                pr_n = 1
            prem = sample(drva, pr_n)
            cand = []
            for rule in calc.inpl:
                if calc.check(rule, prem):
                    cand.append(rule(prem))
            drva.append(choice(cand))
        in_i.append(calc.symb["DE"])
        in_i.append(drva[-1])
        if in_i not in inpt:
            inpt.append(in_i)
            outp.append(drva)
            gbar.update(1)
    gbar.close()
    return inpt, outp