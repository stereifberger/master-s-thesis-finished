# Import libraries
from imports import *
outp_dict = {}

def create_dataset(iterations = list, calculus = list):
    prem = gen_prem()
    drvas = generate_derivations(iterations = iterations, calculus = calculus, prem = prem)
    sample_conclusions = get_conclusions(premises = prem,
                                         max_iterations = iterations[1],
                                         calculus = calculus)
    print(len(drvas))
    drvas = [l for l in tqdm(drvas, desc =  "Checked derivations for sample conclusions") if l[-1] in sample_conclusions]
    print(len(drvas))
    max_y_train = torch.tensor(util.lflt(max(drvas, key=len)))
    #max_y_train_len = 14 * len(max_y_train)

    inpt, y_train_ordered, max_y_train_len = gen_optimized(drvas)
    print(f"LENINPT: {len(inpt)}")
    print(f"LENy_t: {len(y_train_ordered)}")

    inpt = np.array(inpt)
    inpt = torch.tensor(inpt, dtype= torch.float32)


    inpt_2d = inpt.view(inpt.size(0),-1)
    inpt_2d = torch.cat((inpt_2d[:,:1],inpt_2d[:,14:]), dim = 1)

    inpt_3d = inpt
    #inpt_3d = torch.cat((inpt_3d[:,:1],inpt_3d[:,14:]), dim = 1)

    # Print results
    print(f"Number x_train examples: {len(inpt)}")
    print(f"Average number ground truth examples/x_train example: {len(drvas)/len(y_train_ordered)}")
    return inpt_2d, inpt_3d, y_train_ordered, max_y_train_len

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
        sub_drvas = drvas[prev:]
        prev = len(drvas)
        for drva in tqdm(sub_drvas, desc = f"Processed premises for sample conclusions at iteration {iterations}"):
            subsets = util.subsets_with_length(drva, 1) + util.subsets_with_length(drva, 2)
            subsets = sample(subsets, int(round(len(subsets)/(iterations*iterations))))
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

            """
            Description: Implementation of multprocessing for my existing code, via new subfunction "process_subsets".
            Generated by: GPT-4
            Date: 2020-04-05
            URL of Service: https://platform.openai.com/playground
            """
            # Partial function implementation by GPT-4
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
            
    print(f"Number ground truth examples: {len(drvas)}")
    return drvas

def process_subsets(args, calculus):
    i, iterations = args
    processed_data = []
    if len(i) > 1:
        subsets = util.subsets_with_length(i, 1) + util.subsets_with_length(i, 2)
    else:
        subsets = [i]
    if iterations > 3:
        subsets = sample(subsets, int(round(len(subsets)/(iterations*iterations))))
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
    while len(prem) < 30:
        new = sample(prem_pool, 2)
        if new not in prem:
            prem.append(new)
    return prem

"""
Description: Optimization of my exiting "gen" function tu run fast, via making lists immutable tuples
Generated by: GPT-4
Date: 2020-04-05
URL of Service: https://platform.openai.com/playground
"""
def to_immutable(obj):
    """
    Recursively converts lists (and nested lists) into tuples
    to make them hashable.
    """
    if isinstance(obj, list):
        return tuple(to_immutable(item) for item in obj)
    return obj


def gen_optimized(drvas):
    y_train_ordered = []
    symb_DE = calculi.symb["DE"]
    unique_inputs = {}
    outp_dict = {}
    onehot_drvas  = [util.lflt(i) for i in drvas]
    onehot_drvas  = np.array(onehot_drvas, dtype=object)
    maxl = max(len(sub) for sub in onehot_drvas)
    onehot_drvas = np.array([sub + [0] * (maxl - len(sub)) for sub in tqdm(onehot_drvas, desc =  "Padded x_train entries")])
    onehot_drvas = torch.tensor(onehot_drvas) 
    print(1)
    onehot_drvas = [torch.cat([F.one_hot(i[j], num_classes=t_nu + 9) for j in range(len(i))], dim=-1) for i in tqdm(onehot_drvas)]
    #onehot_drvas = parallel_pad(onehot_drvas)
    #onehot_drvas = [F.one_hot(i[1:], num_classes=t_nu + 9) for i in onehot_drvas]
    #onehot_drvas = torch.cat(onehot_drvas, dim=-1)
    max_l = len(onehot_drvas[0])
    n = len(drvas)


    for i, (drv, one_drv) in tqdm(enumerate(zip(drvas, onehot_drvas)), desc = "Processed entries for x_train and y_tdict",total = n):
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
            y_train_ordered.append([one_drv])
        else:
            position = unique_inputs[in_i]
            y_train_ordered[position].append(one_drv)
    

    # Convert the unique_inputs dict keys back to list format if required
    inpt = [[pos] + recursively_convert_to_list(key) for key, pos in unique_inputs.items()]
    print(2)
    max_y = max(len(sub) for sub in y_train_ordered)
    print(f"Second dim length: {max_y}")
    y_padding = [0] * max_l
    max_y_train_len = len(y_train_ordered[0][0])
    print(3)
    #y_train_ordered = [sub + [y_padding] * (max_y - len(sub)) for sub in tqdm(y_train_ordered, desc = "Padded y_train_ordered")]
    #y_train_ordered = torch.tensor(y_train_ordered)
    padded_y_train_ordered = [sub + [torch.tensor(y_padding)] * (max_y - len(sub)) for sub in tqdm(y_train_ordered, desc="Padded y_train_ordered")]
    y_train_ordered = torch.stack([torch.stack(sub) for sub in padded_y_train_ordered])
    inpt = list_to_onehot(inpt)
    return inpt, y_train_ordered, max_y_train_len

def list_to_onehot(list):
    inpt = [util.lflt(i) for i in list]
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
    return inpt_processed

def recursively_convert_to_list(obj):
    """
    Recursively convert tuples (and nested tuples) back into lists.
    This is the reverse operation of the to_immutable function.
    """
    if isinstance(obj, tuple):
        return [recursively_convert_to_list(item) for item in obj]
    return obj

# GPT
def process_item(item, t_nu):
    return torch.cat([F.one_hot(i, num_classes=t_nu + 9) for i in item], dim=-1)

def parallel_pad(input):
    num_cores = cpu_count()  # Or set this to a fixed number of cores if desired

    with Pool(processes=num_cores) as pool:
        # Use partial to set the constant parameter `t_nu`
        func = partial(process_item, t_nu=t_nu)
        
        # Distribute the work across the pool and collect results
        # The tqdm wrapper allows for a progress bar
        results = []
        for result in tqdm(pool.imap_unordered(func, input), total=len(input)):
            results.append(result)
    return results
 
