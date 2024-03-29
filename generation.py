"""Generates the premises."""
def premises_gen(self):
  arg_num = rd.randint(*span_arg_num)
  arguments = rd.sample(terms, arg_num)
  # Optionally some logic for building well formed formulas.
  return arguments

"""When used with the Derivation method it returns:
  - X: The input to a network of the form [PREMISES, DELIMITER, CONCLUSION]
  - y: The output to a network of the form
    [PREMISES, DERIVATION STEP 1,.recursion_task.., DERIVATION STEP N, CONCLUSION]"""
def dataset_builder(iter_range = list,
                    sample_size = int):
  X, y = [], []
  bar = tqdm(total=sample_size, desc="Generating example derivations")
  while len(X) < sample_size:
    derivation = premises_gen()
    if len(iter_range) == 2:
      iterations = range(rd.randint(*iter_range))
    else:
      iterations = iter_range
    premise = copy.copy(derivation)
    for iteration in iterations:
      if len(flatten(derivation)) >= 2:
        args = rd.sample(flatten(derivation), rd.randint(1,2))
      else:
        args = rd.sample(flatten(derivation), 1)
      derived, max_num = apply_nonterms(term_num, args)
      derivation.append(derived)
    conclusion = derivation[-1]
    X_new = to_int(flatten([premise, term_num + 1, conclusion]))
    #print(f"X_new:{X_new}")
    y.append(to_int(flatten(derivation)))

    X.append(X_new)
    bar.update(1)
  bar.close()
  return X, y, max_num