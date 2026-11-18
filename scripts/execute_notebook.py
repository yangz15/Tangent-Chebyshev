"""Execute main.ipynb and preserve cell outputs, including failure diagnostics."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root=Path(__file__).resolve().parents[1]
notebook=nbformat.read(root/'main.ipynb',as_version=4)
try:
    NotebookClient(notebook,timeout=21600,kernel_name='python3',
        resources={'metadata':{'path':str(root)}}).execute()
finally:
    nbformat.write(notebook,root/'main.ipynb')
print('Notebook executed:',root/'main.ipynb')
