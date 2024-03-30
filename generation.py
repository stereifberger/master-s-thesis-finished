import calc

"""When used with the Derivation method it returns:
    - X: The input to a network of the form [PREMISES, DELIMITER, CONCLUSION]
    - y: The output to a network of the form
         [PREMISES, DERIVATION STEP 1,.recursion_task.., DERIVATION STEP N, CONCLUSION]"""

def dat_b(itra = list, s_si = int):
    inpt, outp = [], []
    gbar = tqdm(total=s_si, desc="Generating example derivations")
    while len(X) < s_si:
        drva = []
        for p in rd.randint(1, 3):
            drva.append(calc.rd_f())
        drva.append.(calc.symb["DE"])
        outp = drva
        for it in randint(itra):
            pr_n = rd.randint(1, 2)
            prem = rd.choice(drva, pr_n)
            cand = []
            for rule in calc.inpl:
                if calc.check(rule, prem):
                    cand.append(rule)
            used = rd.choice(cand)
            drva.append(used())
        inpt.append(drva)
        outp.append(drva[-1])
        gbar.update(1)
    gbar.close()
    return inpt, outp