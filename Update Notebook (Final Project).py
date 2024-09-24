import schedule
import time
from nbconvert import NotebookExporter
import nbformat

def run_notebook():
    with open('Final Project.ipynb') as f:
        nb = nbformat.read(f, as_version=4)
    exporter = NotebookExporter()
    body, resources = exporter.from_notebook_node(nb)
    exec(body, globals())

# Schedule notebook execution every 24 hours
schedule.every(24).hours.do(run_notebook)

while True:
    schedule.run_pending()
    time.sleep(1)
