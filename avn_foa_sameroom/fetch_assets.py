"""Cache only official TRAIN episode definitions; never fetch benchmark test audio."""
import argparse, concurrent.futures, gzip, hashlib, json, pathlib, urllib.request


def main():
    p=argparse.ArgumentParser(); p.add_argument('--index',type=pathlib.Path,default=pathlib.Path(__file__).with_name('official_train_index.json'))
    p.add_argument('--output',type=pathlib.Path,required=True); a=p.parse_args()
    rows=[r for r in json.loads(a.index.read_text()) if '/train/content/' in r['path']]
    a.output.mkdir(parents=True,exist_ok=True)
    def fetch(r):
        path=a.output/pathlib.Path(r['path']).name
        if not path.exists():
            url='https://drive.google.com/uc?export=download&id='+r['id']
            with urllib.request.urlopen(url,timeout=90) as f: blob=f.read()
            doc=json.loads(gzip.decompress(blob))
            assert 'episodes' in doc, (r['path'], list(doc)); print('downloaded',r['path'],len(doc['episodes']),flush=True)
            tmp=path.with_suffix('.part'); tmp.write_bytes(blob); tmp.replace(path)
        blob=path.read_bytes(); doc=json.loads(gzip.decompress(blob))
        return dict(**r,sha256=hashlib.sha256(blob).hexdigest(),episodes=len(doc['episodes']),cached_path=str(path))
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        results=list(pool.map(fetch,rows))
    (a.output/'download_manifest.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({'files':len(results),'episodes':sum(r['episodes'] for r in results)}))
if __name__=='__main__':main()
