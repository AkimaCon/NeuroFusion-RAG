from __future__ import annotations
import argparse, json

def reciprocal_rank(ranked_ids, relevant_ids):
    rel=set(relevant_ids)
    for i,x in enumerate(ranked_ids,1):
        if x in rel: return 1/i
    return 0.0

def main():
    ap=argparse.ArgumentParser(description='Evaluate retrieval runs saved as JSONL: {ranked_ids:[], relevant_ids:[]}')
    ap.add_argument('jsonl')
    args=ap.parse_args()
    rows=[json.loads(l) for l in open(args.jsonl, encoding='utf-8')]
    mrr=sum(reciprocal_rank(r['ranked_ids'], r['relevant_ids']) for r in rows)/max(1,len(rows))
    print({'queries':len(rows),'MRR':round(mrr,4)})
if __name__=='__main__': main()
