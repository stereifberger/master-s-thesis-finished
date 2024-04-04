# Import libraries
from libs import *

t_nu = 5

outp_dict = {}


"""When used with the Derivation method it returns:
    - X: The input to a network of the form [PREMISES, DELIMITER, CONCLUSION]
    - y: The output to a network of the form
         [PREMISES, DERIVATION STEP 1,.recursion_task.., DERIVATION STEP N, CONCLUSION]"""
# Define the data generation process. If the networks are trained on the class of all 
# correct derivations then I only need the input data. Also in a future version at each
# iteratio of applying rules a new input could be defined on it, making the function faster.
def dat_b(itra = list, s_si = int):
    t_nu = 9
    number_examples = 0
    inpt, outp = [], []
    gbar = tqdm(total=s_si, desc="Generating example derivations")
    while number_examples < s_si:
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
        if len(in_i) > 0:
            comp_list = [s[1:] for s in inpt] # GPT-4
            if in_i not in comp_list:
                inpt.append([number_examples] + in_i)
                outp_dict[number_examples] = []
                gbar.update(1)
                if drva not in outp:
                    outp.append(drva)
                    outp_dict[number_examples].append(drva)
                number_examples+=1
            else:
                if drva not in outp:
                    position = comp_list.index(in_i)
                    outp_dict[position].append(drva)
                    outp.append(drva)

    gbar.close()
    return inpt, outp, outp_dict