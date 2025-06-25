import nbformat

# Load both notebooks
with open("notebook_A.ipynb") as f:
    nb_a = nbformat.read(f, as_version=4)

with open("notebook_B.ipynb") as f:
    nb_b = nbformat.read(f, as_version=4)

# Append all cells from B into A
nb_a.cells.extend(nb_b.cells)

# Save the result
with open("merged_notebook.ipynb", "w") as f:
    nbformat.write(nb_a, f)
