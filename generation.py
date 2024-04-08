# Import libraries
from imports import *
outp_dict = {}

def create_dataset(iterations = list, calculus = list):
    prem = gen_prem()
    drvas = generate_derivations(iterations = iterations, calculus = calculus, prem = prem)
    sample_conclusions = get_conclusions(premises = prem,
                                         max_iterations = iterations[1],
                                         calculus = calculus)
    drvas = [l for l in tqdm(drvas, desc =  "Checked derivations for sample conclusions") if l[-1] in sample_conclusions]
    max_y_train = torch.tensor(util.lflt(max(drvas, key=len)))
    max_y_train_len = 14 * len(max_y_train)
    inpt, outp_dict = gen_optimized(drvas)
    inpt = [util.lflt(i) for i in inpt]
    inpt = np.array(inpt, dtype=object)
    maxl = max(len(sub) for sub in inpt)
    inpt = np.array([sub + [0] * (maxl - len(sub)) for sub in tqdm(inpt, desc =  "Padded x_train entries")])
    inpt = torch.tensor(inpt)

    # Create padding tensor
    padding = torch.zeros(t_nu + 8)

    # Perform F.one_hot and concatenation operations in a single loop
    inpt_processed = []
    for i in inpt:
        ione = F.one_hot(i[1:], num_classes=t_nu + 9)
        index = torch.cat((i[0].unsqueeze(0), padding))
        conc = torch.cat((index.unsqueeze(0), ione), dim=0)
        inpt_processed.append(conc)

    inpt = np.array(inpt_processed)
    inpt = torch.tensor(inpt, dtype= torch.float32)

    inpt_2d = inpt.view(inpt.size(0),-1)
    inpt_2d = inpt_2d[:,1:]
    inpt_2d = torch.cat((inpt_2d[:,:1],inpt_2d[:,14:]), dim = 1)

    inpt_3d = inpt[:,1:]
    inpt_3d = torch.cat((inpt_3d[:,:1],inpt_3d[:,14:]), dim = 1)

    # Print results
    print(f"Number x_train examples: {len(inpt)}")
    print(f"Average number ground truth examples/x_train example: {len(drvas)/len(outp_dict)}")
    return inpt_2d, inpt_3d, outp_dict, max_y_train_len

def get_conclusions(premises, max_iterations, calculus):
    drvas = []
    for prem in premises:
        subsets = util.subsets_with_length(prem, 1) + util.subsets_with_length(prem, 2)
        for i in subsets:
            for rule in calculus:
                if calculi.check(rule, i):
                    new = i + [rule(i)]
                    drvas.append(new)
    iterations = 2
    prev = 0
    while iterations <= max_iterations:
        sub_drvas = drvas[prev:]#sample(drvas[prev:], int(round(len(drvas[prev:])/(iterations))))
        prev = len(drvas)
        for drva in tqdm(sub_drvas, desc = f"Processed premises for sample conclusions at iteration {iterations}"):
            subsets = util.subsets_with_length(drva, 1) + util.subsets_with_length(drva, 2)
            for i in subsets:
                cand = []
                for rule in calculus:
                    if calculi.check(rule, i):
                        cand.append(rule(i))
            new = drva + [choice(cand)]
            drvas.append(new)
        iterations += 1
    conclusions = [x[-1] for x in drvas]
    conclusions_proxy = []
    conclusions = [conclusions_proxy.append(x) for x in conclusions if x not in conclusions_proxy]
    return conclusions_proxy

def generate_derivations(iterations = list, calculus = list, prem = list):
    iter = 1
    number_examples = 0
    inpt = []
    drvas = prem


    start_point_next_it = 0
    with Pool() as pool:
        while iter <= iterations[1]:
            args = [(i, iter) for i in drvas[start_point_next_it:]]
            drvas_proxy = []

            # partial function implementation by GPT-4
            process_subsets_with_calculus = partial(process_subsets, calculus=calculus)
            for result in tqdm(pool.imap(process_subsets_with_calculus, args),
                            desc=f"Processed at iteration {iter}",
                            total=len(args),
                            miniters=1,
                            mininterval=0.1):
                drvas_proxy.extend(result)

            start_point_next_it = len(drvas)
            if iter == 1:
                drvas = drvas_proxy
            else:
                drvas += drvas_proxy
            iter += 1
            
    print(f"Number ground truth examples in y_tdict: {len(drvas)}")
    print(f"DRVAS:{drvas}")
    return drvas

def process_subsets(args, calculus):
    i, iterations = args
    processed_data = []
    if len(i) > 1:
        subsets = util.subsets_with_length(i, 1) + util.subsets_with_length(i, 2)
    else:
        subsets = [i]
    if iterations > 3:
        subsets = sample(subsets, int(round(len(subsets)/(iterations))))
    for d in subsets: 
        for rule in calculi.ipl:
            if calculi.check(rule, d):
                new = i + [rule(d)]
                if new not in processed_data:
                    processed_data.append(new)
    return processed_data

def gen_prem():
    prem, prem_pool = [], []
    while len(prem_pool) < 10000:   
        form = calculi.gen_wff(randint(1,t_nu), depth=0)
        prem_pool.append(form)
    while len(prem) < 100:
        new = sample(prem_pool, 2)
        if new not in prem:
            prem.append(new)
    return prem

def to_immutable(obj):
    """
    Recursively converts lists (and nested lists) into tuples
    to make them hashable.
    """
    if isinstance(obj, list):
        return tuple(to_immutable(item) for item in obj)
    return obj

def gen_optimized(drvas):
    symb_DE = calculi.symb["DE"]
    unique_inputs = {}
    outp_dict = {}
    n = len(drvas)

    for i, drv in tqdm(enumerate(drvas), desc = "Processed entries for x_train and y_tdict",total = n):
        # Making sure every element is converted to a hashable type
        in_i = (
            to_immutable(drv[:2]),    # Convert the list and any nested lists into tuples
            to_immutable(symb_DE),    # Convert symb_DE into a tuple if it's a list
            to_immutable(drv[-1])     # Convert the last element into a tuple if it's a list
        )
        
        # Use the immutable representation as a key
        if in_i not in unique_inputs:
            position = len(unique_inputs)
            unique_inputs[in_i] = position
            outp_dict[position] = [drv]
        else:
            position = unique_inputs[in_i]
            outp_dict[position].append(drv)

    # Convert the unique_inputs dict keys back to list format if required
    inpt_with_keys = [[pos] + recursively_convert_to_list(key) for key, pos in unique_inputs.items()]

    return inpt_with_keys, outp_dict

def recursively_convert_to_list(obj):
    """
    Recursively convert tuples (and nested tuples) back into lists.
    This is the reverse operation of the to_immutable function.
    """
    if isinstance(obj, tuple):
        return [recursively_convert_to_list(item) for item in obj]
    return obj
