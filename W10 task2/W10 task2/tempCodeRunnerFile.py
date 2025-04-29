import json
import sys

def extract_code_from_notebook(notebook_path, python_path):
    """
    Extract code cells from a Jupyter notebook (.ipynb) and write them to a Python file (.py).
    
    Args:
        notebook_path (str): Path to the input Jupyter notebook file
        python_path (str): Path to the output Python file
    """
    print(f"Converting {notebook_path} to {python_path}")
    
    try:
        # Load the notebook file
        with open(notebook_path, 'r', encoding='utf-8') as notebook_file:
            notebook = json.load(notebook_file)
        
        # Open the output Python file
        with open(python_path, 'w', encoding='utf-8') as python_file:
            # Add a header comment
            python_file.write(f"# Code extracted from {notebook_path}\n\n")
            
            # Extract code from each code cell
            cell_count = 1
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code':
                    # Write a cell separator comment
                    python_file.write(f"# Cell {cell_count}\n")
                    
                    # Write the code content
                    code = ''.join(cell['source'])
                    python_file.write(code)
                    
                    # Add a newline if the cell doesn't end with one
                    if code and not code.endswith('\n'):
                        python_file.write('\n')
                    
                    # Add an extra newline between cells
                    python_file.write('\n')
                    
                    # If the cell has outputs, add them as comments
                    if 'outputs' in cell and cell['outputs']:
                        python_file.write("# Output:\n")
                        for output in cell['outputs']:
                            if 'text' in output:
                                for line in output['text']:
                                    python_file.write(f"# {line}")
                            elif 'data' in output and 'text/plain' in output['data']:
                                for line in output['data']['text/plain']:
                                    python_file.write(f"# {line}")
                        python_file.write('\n')
                    
                    cell_count += 1
                    
        print(f"Successfully converted notebook to Python file: {python_path}")
        return True
        
    except Exception as e:
        print(f"Error converting notebook: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python notebook_to_python.py notebook.ipynb output.py")
        sys.exit(1)
    
    notebook_path = sys.argv[1]
    python_path = sys.argv[2]
    
    extract_code_from_notebook(notebook_path, python_path)