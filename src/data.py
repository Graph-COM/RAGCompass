import json
import os

from huggingface_hub import hf_hub_download


def download_if_not_exists(file_path):
    if not os.path.exists(file_path):
        print(f"File not found at {file_path}, downloading...")
        hf_hub_download(
            repo_id="osunlp/HippoRAG_2",
            repo_type="dataset",
            filename=file_path.split('/')[-1],
            local_dir="dataset",
        )


def _iter_json_array_items(file_path, chunk_size=1024 * 1024):
    decoder = json.JSONDecoder()
    with open(file_path, "r", encoding="utf-8") as f:
        buffer = ""
        array_started = False
        array_finished = False

        while True:
            chunk = f.read(chunk_size)
            if chunk:
                buffer += chunk
            elif not buffer.strip():
                break

            while True:
                buffer = buffer.lstrip()
                if not buffer:
                    break

                if not array_started:
                    if buffer[0] != "[":
                        raise ValueError(f"Expected top-level JSON array in {file_path}")
                    array_started = True
                    buffer = buffer[1:]
                    continue

                buffer = buffer.lstrip()
                if not buffer:
                    break

                if buffer[0] == "]":
                    array_finished = True
                    buffer = buffer[1:]
                    break

                if buffer[0] == ",":
                    buffer = buffer[1:]
                    continue

                try:
                    item, end_idx = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    if chunk:
                        break
                    raise

                yield item
                buffer = buffer[end_idx:]

            if not chunk:
                break

        trailing = buffer.strip()
        if trailing and trailing != "]":
            raise ValueError(f"Unexpected trailing content in {file_path}: {trailing[:200]!r}")
        if not array_started or not array_finished:
            raise ValueError(f"Did not find a complete JSON array in {file_path}")


def iter_corpus(dataset):
    corpus_path = f"dataset/{dataset}_corpus.json"
    download_if_not_exists(corpus_path)

    for doc in _iter_json_array_items(corpus_path):
        yield f"{doc['title']}\n{doc['text']}"


def load_corpus(dataset):
    return list(iter_corpus(dataset))

def get_gold_answers(sample):
    gold_ans = None
    
    if 'answer' in sample or 'gold_ans' in sample:
        gold_ans = sample['answer'] if 'answer' in sample else sample['gold_ans']
    elif 'reference' in sample:
        gold_ans = sample['reference']
    elif 'obj' in sample:
        gold_ans = set(
            [sample['obj']] + [sample['possible_answers']] + [sample['o_wiki_title']] + [sample['o_aliases']])
        gold_ans = list(gold_ans)
    assert gold_ans is not None
    if isinstance(gold_ans, str):
        gold_ans = [gold_ans]
    assert isinstance(gold_ans, list)
    gold_ans = set(gold_ans)
    if 'answer_aliases' in sample:
        gold_ans.update(sample['answer_aliases'])
    
    return gold_ans

def get_gold_docs(sample, dataset_name):
    if 'supporting_facts' in sample:  # hotpotqa, 2wikimultihopqa
        gold_title = set([item[0] for item in sample['supporting_facts']])
        gold_title_and_content_list = [item for item in sample['context'] if item[0] in gold_title]
        if dataset_name.startswith('hotpotqa'):
            gold_doc = [item[0] + '\n' + ''.join(item[1]) for item in gold_title_and_content_list]
        else:
            gold_doc = [item[0] + '\n' + ' '.join(item[1]) for item in gold_title_and_content_list]
    elif 'contexts' in sample:
        gold_doc = [item['title'] + '\n' + item['text'] for item in sample['contexts'] if item['is_supporting']]
    else:
        assert 'paragraphs' in sample, "`paragraphs` should be in sample, or consider the setting not to evaluate retrieval"
        gold_paragraphs = []
        for item in sample['paragraphs']:
            if 'is_supporting' in item and item['is_supporting'] is False:
                continue
            gold_paragraphs.append(item)
        gold_doc = [item['title'] + '\n' + (item['text'] if 'text' in item else item['paragraph_text']) for item in gold_paragraphs]

    gold_doc = list(set(gold_doc))
    return gold_doc

def load_qa(dataset):
    sample_path = f"dataset/{dataset}.json"
    download_if_not_exists(sample_path)
    with open(sample_path, 'r') as f:
        samples_ = json.load(f)

    samples = []
    id_seen = set()
    for i, sample_i_ in enumerate(samples_):
        try:
            gold_doc_list = get_gold_docs(sample_i_, dataset)
        except:
            gold_doc_list = None
        
        if 'id' in sample_i_:
            sample_id = sample_i_['id']
        else:
            sample_id = f'{dataset}_{i}'
        assert sample_id not in id_seen
        id_seen.add(sample_id)
        
        sample_i = {
            'id': sample_id,
            'question': sample_i_['question'],
            'answers': get_gold_answers(sample_i_),
            'gold_doc_list': gold_doc_list
        }
        if 'excluded_ids' in sample_i_:
            sample_i['excluded_ids'] = sample_i_['excluded_ids']
        samples.append(sample_i)
    
    return samples

def filter_cached(out_file, full_samples):
    ids_done = set()
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            ids_done = set([
                json.loads(line)['id'] for line in f
            ])
    
    samples = []
    for sample_i in full_samples:
        if sample_i['id'] not in ids_done:
            samples.append(sample_i)
    
    return samples
