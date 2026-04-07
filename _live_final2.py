import sys, logging, urllib.request, json, ssl, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="ascii", errors="replace")
logging.disable(logging.CRITICAL)
ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
BASE="https://tharunrajr-cyber-battle-env.hf.space"
def post(path, body):
    req=urllib.request.Request(BASE+path,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    r=urllib.request.urlopen(req,context=ctx,timeout=30)
    return json.loads(r.read()),r.status
p=0;f=0
def ck(name,ok,val=""):
    global p,f
    if ok: print("[PASS]",name); p+=1
    else:  print("[FAIL]",name,str(val)[:50]); f+=1
print("=== FINAL DUAL-ROLE LIVE TESTS ==="); print()
for task in ["easy","medium","hard"]:
    print("-- TASK:",task.upper(),"--")
    d,s=post("/reset",{"task":task,"role":"attacker","seed":42})
    ck(task+"/attacker reset OK", s==200 and d.get("attacker_position")==0)
    d,s=post("/step",{"action_type":"scan","target_node":1,"role":"attacker","last_task":task})
    ck(task+"/attacker scan reward>0", s==200 and d.get("reward",0)>0)
    d,s=post("/reset",{"task":task,"role":"defender","seed":42})
    ck(task+"/defender reset OK", s==200 and "DEFENDER" in d.get("last_action_message",""))
    d,s=post("/step",{"action_type":"monitor","target_node":1,"role":"defender","last_task":task})
    ck(task+"/defender step [ATT]+[DEF]", s==200 and "[ATT]" in d.get("last_action_message","") and "[DEF]" in d.get("last_action_message",""))
    ck(task+"/defender reward in range", -0.5<=d.get("reward",99)<=0.7)
    print()
print("=== RESULTS: %d/%d PASSED ===" % (p,p+f))
