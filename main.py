import sys
import time

text = sys.argv[1] # input text
llm_path = sys.argv[2] # hf models

# load model
from vllm import LLM, SamplingParams
llm = LLM(model=llm_path, swap_space=8)
def llm_gen(queries: list, temperature=1.0):
    B_INST, E_INST = "[INST]", "[/INST]"
    def wrap_query(prompt):
        return f"{B_INST} {prompt} {E_INST}"
    sampling_params = SamplingParams(temperature=temperature, max_tokens=1024)
    results = llm.generate([wrap_query(query) for query in queries], sampling_params)
    return [result.outputs[0].text.strip() for result in results]

# step 0: baseline
start_time = time.time()
responses = llm_gen([text])
total_time = time.time() - start_time

print(f"** Total time: {total_time:.2f}s")

# step 1 : sample N responses
N = 10
start_time = time.time()
responses = llm_gen([text] * N)
responses = [{"text": response} for response in responses]

# step 2 : extract facts from responses
from string import Template
instruct_template = Template("List all facts from the text below in numerical order. Each fact should be a self-contained sentence.\n\n${text}")
prompts =  [instruct_template.substitute(text=response["text"]) for response in responses]
_responses = llm_gen(prompts, temperature=0)

import re
def split_text(text):
    facts = []
    curr_fact = None
    started = False
    for _text in text.split("\n"):
        _text = _text.strip()
        # recognize whether "_text" startswith a number (use regex)
        if re.match(r"^\d+", _text):
            # clean the start number
            _text = re.sub(r"^\d+\.?", "", _text).strip()
            if curr_fact:
                facts.append(curr_fact)
            curr_fact = _text
            started = True
        elif started:
            curr_fact += f"\n{_text}"
    if curr_fact:
        facts.append(curr_fact)
    return facts
for response, _response in zip(responses, _responses):
    facts = split_text(_response)
    response["facts"] = facts


# step 3 : calculate endorsement scores
endorse_template = Template('''
Take the following as truth: 
$premise

Then the following statement: "$hypothesis" is true, false, or inconclusive?
'''.strip())
ph_pairs = []
anchors = []
for response in responses:
    response["facts"] = [{"text": fact, "endorse": []} for fact in response["facts"]]
regu_fn = lambda x: re.sub("\n+", "\n", x).strip()
for i, prem_response in enumerate(responses):
    for j, hypo_response in enumerate(responses):    
        for fact in prem_response["facts"]:
            ph_pairs.append({"premise": regu_fn(hypo_response["text"]), "hypothesis": regu_fn(fact["text"]),})
            anchors.append(fact)
ph_pairs = [endorse_template.substitute(premise=pair["premise"], hypothesis=pair["hypothesis"]) for pair in ph_pairs]
_responses = llm_gen(ph_pairs, temperature=0)
for anchor, _response in zip(anchors, _responses):
    match = re.search(r'\btrue\b|\bfalse\b|\binconclusive\b', _response, re.I)
    if match:
        pred_term = match.group(0).lower()
    else:
        pred_term = "inconclusive"
    anchor["endorse"].append(pred_term)


# step 4 : generate final response
# first select high-quality facts with alpha
from nltk import word_tokenize
from nltk.corpus import stopwords
STOPWORDS=stopwords.words('english')
from string import punctuation

def strict_tokenize(text):
    word_list = word_tokenize(text.lower())
    word_list = [word for word in word_list if word not in STOPWORDS]
    word_list = [word for word in word_list if word not in punctuation]
    return word_list

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
def kmean_selection(X, n_clusters=100):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X)
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
    return closest

alpha = 1.0 - 1e-3
def get_endorse_score(fact):
    endorse_types = fact["endorse"]
    x = len([i for i in endorse_types if i == "true"])
    y = len(endorse_types)
    if y == 0:
        return 0
    return x / y

total_fact_num, total_select_fact_num = 0, 0
collected_facts = set()
for response in responses:
    selected_facts = {fact["text"] for fact in response["facts"] if get_endorse_score(fact) >= alpha}
    collected_facts.update(selected_facts)
    total_fact_num += len(response["facts"])
    total_select_fact_num += len(selected_facts)
collected_facts = list(collected_facts)
avg_fact_num = total_fact_num // len(responses)
avg_select_fact_num = total_select_fact_num // len(responses)

# one-hot vector
import numpy as np
word_set = set()
for fact in collected_facts:
    words = strict_tokenize(fact)
    word_set.update(set(words))
word_list = list(word_set)
vectors = []
for fact in collected_facts:
    words = strict_tokenize(fact)
    vector = np.zeros(len(word_list))
    for word in words:
        vector[word_list.index(word)] = 1.
    vectors.append(vector)

# kmean
class_num = min(avg_fact_num, avg_select_fact_num * 2)
if len(vectors) < class_num:
    selected_facts = collected_facts
else:
    indices = kmean_selection(vectors, n_clusters=class_num)
    selected_facts = [collected_facts[indice] for indice in indices]

# next, generate given selected facts. This template is used for biographies and may not be suitable for different tasks.
from string import Template
generate_template = Template(
'''Knowledge from other sources:
${facts}

Given materials above, ${question}
'''.strip())
# each fact startswith an indice
selected_facts_str = [f"{i+1}. {fact}" for i, fact in enumerate(selected_facts)]
prompts = [generate_template.substitute(facts=selected_facts_str, question=text)]
_responses = llm_gen(prompts)
final_response = _responses[0]

total_time = time.time() - start_time

print(f"** Input:\n{text}")
print(f"** Selected facts:\n{selected_facts}")
print(f"** Final response:\n{final_response}")

print(f"** Total time: {total_time:.2f}s")