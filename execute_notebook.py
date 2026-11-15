"""Restart and execute notebook; optional in-process kernel avoids socket needs."""
from pathlib import Path
import hashlib,json,os,sys
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parent
path=ROOT/'optimization_engine.ipynb';csv=ROOT/'data/candidate_plans.csv'
before=hashlib.sha256(csv.read_bytes()).hexdigest() if csv.exists() else None
nb=nbformat.read(path,as_version=4)
if '--in-process' in sys.argv:
    from ipykernel.inprocess.manager import InProcessKernelManager
    os.chdir(ROOT)
    manager=InProcessKernelManager();manager.start_kernel()
    # Explicit restart discards all prior user state before executing cells.
    manager.restart_kernel();client=manager.client();client.start_channels()
    for cell in nb.cells:
        if cell.cell_type!='code':continue
        cell.outputs=[];client.execute(cell.source)
        while client.iopub_channel.msg_ready():
            msg=client.get_iopub_msg();kind=msg['msg_type']
            if kind=='execute_input':cell.execution_count=msg['content']['execution_count']
            if kind in ['stream','display_data','execute_result','error']:
                cell.outputs.append(nbformat.v4.output_from_msg(msg))
                if kind=='error':raise RuntimeError(msg['content']['traceback'])
    client.stop_channels();manager.shutdown_kernel();kernel_type='restarted ipykernel InProcessKernel in a fresh Python process (no sockets)'
else:
    nb=NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    kernel_type='fresh external Python kernel'
nbformat.write(nb,path)
after=hashlib.sha256(csv.read_bytes()).hexdigest()
if before is not None and before!=after:raise RuntimeError('Candidate CSV changed across rerun')
manifest=json.loads((ROOT/'results/reproducibility.json').read_text())
manifest.update({'fresh_kernel_verified':True,'kernel_type':kernel_type,'stable_candidate_sha256':after,'same_candidate_csv_on_repeat':before==after})
(ROOT/'results/reproducibility.json').write_text(json.dumps(manifest,indent=2))
print('Fresh-kernel run passed; candidate CSV unchanged:',after)
