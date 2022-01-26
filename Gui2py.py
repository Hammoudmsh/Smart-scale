
import os,  sys, datetime

ps=""#"C:/Python/Python 3.7.4/Scripts/"
ui2py=ps+"pyuic5.exe"
res2py=ps+"pyrcc5.exe"


def DO(root_dir):	
	for root,dirr,filename in os.walk(root_dir):
		for fn in filename:
			if ".ui" in fn:
				f,e=fn.split(".")
				cmd=ui2py+" -x "+fn+" -o "+f+".py"
				print(cmd,'\n')
				os.system(cmd)

			elif ".qrc" in fn:
				f,e=fn.split(".")
				cmd=res2py+" "+fn+" -o "+f+"_rc.py"
				print(cmd,'\n')
				os.system(cmd)
				#os.system("pause")

DO(os.getcwd())
