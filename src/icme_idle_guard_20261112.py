"""Read-only, privacy-filtered whole-machine training scan for the requested idle benchmark."""
from pathlib import Path
import os,re,time
PATTERN=re.compile(r'^(?:train|pretrain|finetune|fine_tune)(?:[._-]).*',re.I)
def training_processes():
 found=[];unreadable=0
 for d in Path('/proc').iterdir():
  if not d.name.isdigit():continue
  try:
   raw=(d/'cmdline').read_bytes();argv=[x.decode(errors='replace') for x in raw.split(b'\0') if x];names=[Path(x).name for x in argv]
   matches=[n for n in names if (n.endswith('.py') and PATTERN.match(n)) or n in ['torchrun','deepspeed']]
   if not matches and any(x in argv for x in ['--do_train','--train']):matches=['explicit_training_flag']
   if not matches and any(x in argv for x in ['torch.distributed.run','accelerate.commands.launch']):matches=['distributed_training_launcher']
   if matches:found.append({'pid':int(d.name),'entrypoints':matches})
  except (OSError,ProcessLookupError):unreadable+=1
 return {'matches':found,'unreadable_process_count':unreadable,'checked_unix':time.time()}
def assert_no_training():
 a=training_processes()
 if a['matches']:raise RuntimeError('Trainingprocessesdetected:'+str(a['matches']))
 return a
