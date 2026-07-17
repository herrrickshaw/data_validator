import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, ".bt_bin")
import backtest as bt, pandas as pd, json, time
out={}
for mkt in ["US","IN","JP","KR","CN"]:
    t=time.time()
    try:
        df=bt.run_all(mkt, horizon=21, verbose=False)
        out[mkt]=df.to_dict("records")
        print(f"{mkt}: {len(df)} screens in {time.time()-t:.0f}s", flush=True)
        print(df.to_string(index=False), flush=True)
    except Exception as e:
        print(f"{mkt}: ERR {str(e)[:80]}", flush=True)
json.dump(out, open("results/backtest_deep.json","w"), default=str)
print("\nDONE — results/backtest_deep.json", flush=True)
