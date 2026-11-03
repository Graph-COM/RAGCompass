import json
import numpy as np

def eval(out_file):
    recalls = []
    ndcgs = []
    f1s = []
    latencies = []
    
    with open(out_file, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line.strip())
            recalls.append(result['recall'])
            if 'ndcg' in result:
                ndcgs.append(result['ndcg'])
            f1s.append(result['f1'])
            if 'latency' in result:
                latencies.append(result['latency'])
    
    assert len(recalls) == len(f1s)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1s)
    
    print(f"Average Recall: {avg_recall:.4f}")
    if ndcgs:
        avg_ndcg = np.mean(ndcgs)
        print(f"Average NDCG: {avg_ndcg:.4f}")
    print(f"Average F1: {avg_f1:.4f}")
    if latencies:
        avg_latency = np.mean(latencies)
        print(f"Average Latency: {avg_latency:.4f}")
    print(f"Number of samples: {len(recalls)}")
    
    return avg_recall, avg_f1
