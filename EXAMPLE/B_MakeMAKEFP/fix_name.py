import os

efps = [fl for fl in os.listdir(os.getcwd()) if fl.endswith('.efp')]
for efp in efps:
	fragname = efp.replace('.efp', '')
	with open(efp, 'r') as fl:
		efp_lns = fl.readlines()
	out_efp = open(efp, 'w+')
	for ln in efp_lns:
		if ' $FRAGNAME' in ln:
			new_ln = ln.replace('FRAGNAME', f'{fragname}')
			out_efp.write(new_ln)
		else:
			out_efp.write(ln)
	out_efp.close()
