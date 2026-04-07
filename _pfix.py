txt = open("inference.py", encoding="utf-8").read()
old = '"action_type": atype, "target_node": target},'
new = '"action_type": atype, "target_node": target, "role": role, "last_task": task},'
if old in txt:
    open("inference.py","w",encoding="utf-8").write(txt.replace(old, new, 1))
    print("DONE - role+task added to /step")
else:
    print("NOT FOUND")
    # Show context
    idx = txt.find('"action_type": atype')
    print("Context:", repr(txt[idx:idx+80]))
