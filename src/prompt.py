import json

one_shot_rag_qa_docs = (
    """Wikipedia Title: The Last Horse\nThe Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film directed by Edgar Neville starring Fernando Fernán Gómez.\n"""
    """Wikipedia Title: Southampton\nThe University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students. The university is ranked in the top 100 research universities in the world in the Academic Ranking of World Universities 2010. In 2010, the THES - QS World University Rankings positioned the University of Southampton in the top 80 universities in the world. The university considers itself one of the top 5 research universities in the UK. The university has a global reputation for research into engineering sciences, oceanography, chemistry, cancer sciences, sound and vibration research, computer science and electronics, optoelectronics and textile conservation at the Textile Conservation Centre (which is due to close in October 2009.) It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of Natural Environment Research Council-funded marine research.\n"""
    """Wikipedia Title: Stanton Township, Champaign County, Illinois\nStanton Township is a township in Champaign County, Illinois, USA. As of the 2010 census, its population was 505 and it contained 202 housing units.\n"""
    """Wikipedia Title: Neville A. Stanton\nNeville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). He has written and edited over a forty books and over three hundered peer-reviewed journal papers on applications of the subject. Stanton is a Fellow of the British Psychological Society, a Fellow of The Institute of Ergonomics and Human Factors and a member of the Institution of Engineering and Technology. He has been published in academic journals including "Nature". He has also helped organisations design new human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.\n"""
    """Wikipedia Title: Finding Nemo\nFinding Nemo Theatrical release poster Directed by Andrew Stanton Produced by Graham Walters Screenplay by Andrew Stanton Bob Peterson David Reynolds Story by Andrew Stanton Starring Albert Brooks Ellen DeGeneres Alexander Gould Willem Dafoe Music by Thomas Newman Cinematography Sharon Calahan Jeremy Lasky Edited by David Ian Salter Production company Walt Disney Pictures Pixar Animation Studios Distributed by Buena Vista Pictures Distribution Release date May 30, 2003 (2003 - 05 - 30) Running time 100 minutes Country United States Language English Budget $$94 million Box office $$940.3 million"""
)

rag_qa_system = (
    'As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. '
    'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
    'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations. For multiple answers, separate them by [ANS].'
)

one_shot_rag_qa_input = (
    f"{one_shot_rag_qa_docs}"
    "\n\nQuestion: "
    "When was Neville A. Stanton's employer founded?"
    '\nThought: '
)

one_shot_rag_qa_output = (
    "The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. "
    "\nAnswer: 1862."
)

ircot_rag_qa_system = (
    'As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. '
    'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
    'If you reach what you believe to be the final step, conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.'
)

ner_system = """Your task is to extract named entities from the given paragraph. 
Respond with a JSON list of entities.
"""

one_shot_ner_paragraph = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""

one_shot_ner_output = """{"named_entities":
    ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]
}
"""

ner_conditioned_re_system = """Your task is to construct an RDF (Resource Description Framework) graph from the given doc and named entity lists. 
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph. 

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each doc.
- Clearly resolve pronouns to their specific names to maintain clarity.

"""

ner_conditioned_re_frame = """Convert the doc into a JSON dict, it has a named entity list and a triple list.
Doc:
```
{doc}
```

{named_entity_json}
"""

ner_conditioned_re_input = ner_conditioned_re_frame.format(doc=one_shot_ner_paragraph, named_entity_json=one_shot_ner_output)

ner_conditioned_re_output = """{"triples": [
            ["Radio City", "located in", "India"],
            ["Radio City", "is", "private FM radio station"],
            ["Radio City", "started on", "3 July 2001"],
            ["Radio City", "plays songs in", "Hindi"],
            ["Radio City", "plays songs in", "English"],
            ["Radio City", "forayed into", "New Media"],
            ["Radio City", "launched", "PlanetRadiocity.com"],
            ["PlanetRadiocity.com", "launched in", "May 2008"],
            ["PlanetRadiocity.com", "is", "music portal"],
            ["PlanetRadiocity.com", "offers", "news"],
            ["PlanetRadiocity.com", "offers", "videos"],
            ["PlanetRadiocity.com", "offers", "songs"]
    ]
}
"""

routing_sys_prompt = """As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. Your response starts after "Thought: ".

1) Examine whether advanced knowledge-graph-based retrieval given the question and retrieved passages is needed. If so, conclude with "REQUIRES_KG".
2) Otherwise, methodically break down your reasoning process, illustrating how you arrive at conclusions. Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations. For multiple answers, separate them by [ANS]."""

one_input_template = """[[ ## question ## ]]\n{question}\n\n[[ ## fact_before_filter ## ]]\n{fact_before_filter}\n\nRespond with the corresponding output fields, starting with the field `[[ ## fact_after_filter ## ]]` (must be formatted as a valid Python Fact), and then ending with the marker for `[[ ## completed ## ]]`."""

one_output_template = """[[ ## fact_after_filter ## ]]\n{fact_after_filter}\n\n[[ ## completed ## ]]"""

class PromptTemplateManager:
    def render_routing(self, top_doc_list, question):
        docs_text = ""
        for passage in top_doc_list:
            docs_text += f"Article: {passage}\n\n"
    
        input_prompt = docs_text + 'Question: ' + question + '\nThought: '

        prompt = [
            {"role": "system", "content": routing_sys_prompt},
            {"role": "user", "content": input_prompt},
        ]
        
        return prompt
    
    def render_qa(self, top_doc_list, q):
        prompt_user = ''
        for passage in top_doc_list:
            prompt_user += f'Article: {passage}\n\n'
        prompt_user += 'Question: ' + q + '\nThought: '

        prompt = [
            {"role": "system", "content": rag_qa_system},
            {"role": "user", "content": one_shot_rag_qa_input},
            {"role": "assistant", "content": one_shot_rag_qa_output},
        ]
        prompt.append({
            "role": "user",
            "content": prompt_user
        })
        
        return prompt
    
    def render_qa_ircot(self, top_doc_list, q, thoughts):
        prompt_user = ''
        for passage in top_doc_list:
            prompt_user += f'Article: {passage}\n\n'
        prompt_user += 'Question: ' + q + '\nThought: ' + ' '.join(thoughts)

        prompt = [
            {"role": "system", "content": rag_qa_system},
            {"role": "user", "content": one_shot_rag_qa_input},
            {"role": "assistant", "content": one_shot_rag_qa_output},
        ]
        prompt.append({
            "role": "user",
            "content": prompt_user
        })
        
        return prompt

    def render_qa_ircot_no_demo(self, top_doc_list, q, thoughts):
        prompt_user = ''
        for passage in top_doc_list:
            prompt_user += f'Article: {passage}\n\n'
        prompt_user += 'Question: ' + q + '\nThought: ' + ' '.join(thoughts)

        prompt = [
            {"role": "system", "content": rag_qa_system},
        ]
        prompt.append({
            "role": "user",
            "content": prompt_user
        })
        
        return prompt
    
    def render_ner(self, doc):
        prompt = [
            {"role": "system", "content": ner_system},
            {"role": "user", "content": one_shot_ner_paragraph},
            {"role": "assistant", "content": one_shot_ner_output},
            {"role": "user", "content": doc}
        ]
        
        return prompt
    
    def render_triple(self, doc, ne_json):
        prompt = [
            {"role": "system", "content": ner_conditioned_re_system},
            {"role": "user", "content": ner_conditioned_re_input},
            {"role": "assistant", "content": ner_conditioned_re_output},
            {"role": "user", "content": ner_conditioned_re_frame.format(doc=doc, named_entity_json=ne_json)}
        ]
        
        return prompt

    def render_filter_triple(self, q, candidate_facts):
        with open('./src/hipporag2_prompts/filter_llama3.3-70B-Instruct.json', 'r') as f:
            dspy_saved = json.load(f)
        
        prompt = [
            {"role": "system", "content": dspy_saved['prog']['system']}
        ]

        demos = dspy_saved["prog"]["demos"]
        for demo in demos:
            prompt.append({"role": "user", "content": one_input_template.format(
                question=demo["question"], fact_before_filter=demo["fact_before_filter"])})
            prompt.append({"role": "assistant", "content": one_output_template.format(
                fact_after_filter=demo["fact_after_filter"])})
        
        prompt.append({"role": "user", "content": one_input_template.format(
            question=q, fact_before_filter=json.dumps(
                {"fact": candidate_facts}
            ))})
        
        return prompt
