import warnings; warnings.filterwarnings("ignore")
import market_memory as mm, fundamental_metrics as fm
d = mm.read("IN", tier="stm") or mm.read("IN", tier="ltm")
scored=sorted(((s,float((df["Close"]*df["Volume"]).tail(60).mean())) for s,df in d.items() if df is not None and len(df)>=20), key=lambda x:-x[1])
syms=[s for s,_ in scored[:400]]
print(f"[IN] fetching up to 300 fundamentals (screener.in) from top-400 liquid ...", flush=True)
df=fm.load_fundamentals("IN", limit=300, symbols=syms, verbose=True)
print(f"[IN] cache now: {len(df)} symbols", flush=True)
