import os                                         
import sys                                        
                                                  
sys.path.append('/Users/akivalipshitz/Developer/netrain/src')              
import Constants as const                         
tasknum = int(os.environ['SLURM_ARRAY_TASK_ID'])  
fd = open('/Users/akivalipshitz/Developer/netrain/trainings/1:22.01.2017/trainscripts/thing-awal-train_script.txt', 'r')                  
lines = fd.readlines()                            
print tasknum, len(lines)                         
fd.close()                                        
task = lines[tasknum-1]                           
args = map(lambda a: a, task.split(' '))[2:-1]    
print 'Running Task {}'.format(args )             
from montecarlotrain import run                   
                                                  
run(args)                                         
                                                  
