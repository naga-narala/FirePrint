#!/usr/bin/env python3
import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
import sys

def convert_file(py_file):
    with open(py_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    cells = []
    current_cell = []
    cell_type = 'code'
    
    for line in lines:
        if line.startswith('# %% [markdown]'):
            if current_cell:
                cells.append(new_markdown_cell('\n'.join(current_cell)) if cell_type == 'markdown' else new_code_cell('\n'.join(current_cell)))
                current_cell = []
            cell_type = 'markdown'
        elif line.startswith('# %%'):
            if current_cell:
                cells.append(new_markdown_cell('\n'.join(current_cell)) if cell_type == 'markdown' else new_code_cell('\n'.join(current_cell)))
                current_cell = []
            cell_type = 'code'
        else:
            if cell_type == 'markdown' and line.startswith('# '):
                current_cell.append(line[2:])
            else:
                current_cell.append(line)
    
    if current_cell:
        cells.append(new_markdown_cell('\n'.join(current_cell)) if cell_type == 'markdown' else new_code_cell('\n'.join(current_cell)))
    
    nb = new_notebook()
    nb.cells = cells
    nb.metadata = {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'codemirror_mode': {'name': 'ipython', 'version': 3},
            'file_extension': '.py',
            'mimetype': 'text/x-python',
            'name': 'python',
            'nbconvert_exporter': 'python',
            'pygments_lexer': 'ipython3',
            'version': '3.8.5'
        }
    }
    
    notebook_file = py_file.replace('.py', '.ipynb')
    with open(notebook_file, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print(f'Converted {py_file} -> {notebook_file}')

if __name__ == '__main__':
    convert_file('01_Fire_Polygon_to_Fingerprint.py')
