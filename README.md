# DE_WeilInstitute
A Repository for the Data Engineer Technical Interview at the Weil Institute at the University of Michigan.

The Jupyter Notebook html (walkthrough.html) with have an overview of the technical challenge content.

# Project Setup

## Create a Virtual Environment

In a shell script, create a virtual environment:
```sh
python -m venv .venv
```

Then Activate the virtual Environment:
```sh
source .venv/Scripts/activate
```

Install the required dependencies:
```sh
pip install -r requirements.txt
```

Run the test script to check all imports processed properly:
```sh
python test_imports.py
```

You should see dependencies were successfully imported.
