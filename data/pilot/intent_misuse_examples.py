"""
intent_misuse_examples.py — 10 diverse hand-written Intent Misuse examples.

Each example uses a REAL, valid API — but the WRONG way for the task.
Static tools (pylint, mypy) should catch ≈0% of these, since nothing is
syntactically broken. This is the core thesis validation.

Each example is a dict with:
  - label: short description
  - code: the buggy generated code
  - correct_code: what the code *should* look like
  - explanation: why this is Intent Misuse
  - library: which library domain
"""

INTENT_MISUSE_EXAMPLES = [
    # 1. Dict subscript on Response object
    {
        "label": "response_subscript",
        "code": '''\
import requests
def get_data(url):
    resp = requests.get(url)
    data = resp['results']
    return data
''',
        "correct_code": '''\
import requests
def get_data(url):
    resp = requests.get(url)
    data = resp.json()['results']
    return data
''',
        "explanation": (
            "resp is a Response object, not a dict. Must call .json() first. "
            "This is the canonical Intent Misuse: real API, wrong usage pattern."
        ),
        "library": "requests",
    },

    # 2. Wrong sort method on DataFrame
    {
        "label": "wrong_sort_method",
        "code": '''\
import pandas as pd
def sort_by_column(df, col):
    result = df.sort(col)
    return result
''',
        "correct_code": '''\
import pandas as pd
def sort_by_column(df, col):
    result = df.sort_values(col)
    return result
''',
        "explanation": (
            "df.sort() was removed in pandas 1.0 (it was deprecated since 0.17). "
            "The correct method is df.sort_values(). LLMs trained on old code "
            "often generate the deprecated form."
        ),
        "library": "pandas",
    },

    # 3. Wrong parameter type assumption
    {
        "label": "wrong_param_type",
        "code": '''\
import pandas as pd
def load_csv(path):
    df = pd.read_csv(path, header=True)
    return df
''',
        "correct_code": '''\
import pandas as pd
def load_csv(path):
    df = pd.read_csv(path, header=0)
    return df
''',
        "explanation": (
            "pd.read_csv's header parameter takes an int (row number) or list "
            "of ints, not a boolean. header=True is interpreted as header=1, "
            "which skips the actual first data row. Syntactically valid, "
            "semantically wrong."
        ),
        "library": "pandas",
    },

    # 4. Treating os.listdir() result as dict
    {
        "label": "listdir_as_dict",
        "code": '''\
import os
def get_file_info(directory):
    contents = os.listdir(directory)
    sizes = contents['sizes']
    return sizes
''',
        "correct_code": '''\
import os
def get_file_info(directory):
    contents = os.listdir(directory)
    sizes = [os.path.getsize(os.path.join(directory, f)) for f in contents]
    return sizes
''',
        "explanation": (
            "os.listdir() returns a list of strings, not a dict with metadata. "
            "The LLM seems to confuse it with os.scandir() or assumes a richer "
            "return type."
        ),
        "library": "os",
    },

    # 5. String method on bytes (content vs text)
    {
        "label": "content_vs_text",
        "code": '''\
import requests
def get_lines(url):
    resp = requests.get(url)
    lines = resp.content.split(',')
    return lines
''',
        "correct_code": '''\
import requests
def get_lines(url):
    resp = requests.get(url)
    lines = resp.text.split(',')
    return lines
''',
        "explanation": (
            "resp.content returns bytes, resp.text returns str. Calling "
            ".split(',') on bytes requires b',' not ','. Using .text is "
            "the correct choice for string splitting."
        ),
        "library": "requests",
    },

    # 6. Wrong aggregation method
    {
        "label": "wrong_aggregation",
        "code": '''\
import pandas as pd
def total_sales(df):
    total = df['revenue'].count()
    return total
''',
        "correct_code": '''\
import pandas as pd
def total_sales(df):
    total = df['revenue'].sum()
    return total
''',
        "explanation": (
            "count() returns the number of non-null entries, not the sum. "
            "For total sales, you need sum(). Both are real pandas methods, "
            "both return a number — but the semantics are completely different."
        ),
        "library": "pandas",
    },

    # 7. Missing .json() before key access on POST response
    {
        "label": "post_no_json",
        "code": '''\
import requests
def create_resource(url, payload):
    resp = requests.post(url, json=payload)
    resource_id = resp['id']
    return resource_id
''',
        "correct_code": '''\
import requests
def create_resource(url, payload):
    resp = requests.post(url, json=payload)
    resource_id = resp.json()['id']
    return resource_id
''',
        "explanation": (
            "Same pattern as example 1 but with POST instead of GET. "
            "Response object is not subscriptable — need .json() first."
        ),
        "library": "requests",
    },

    # 8. Wrong numpy operation for vector magnitude
    {
        "label": "wrong_numpy_magnitude",
        "code": '''\
import numpy as np
def compute_magnitude(vector):
    mag = np.abs(vector)
    return mag
''',
        "correct_code": '''\
import numpy as np
def compute_magnitude(vector):
    mag = np.linalg.norm(vector)
    return mag
''',
        "explanation": (
            "np.abs() returns element-wise absolute values (an array), not "
            "the vector magnitude (a scalar). np.linalg.norm() is the correct "
            "API. Both are real numpy functions — classic Intent Misuse."
        ),
        "library": "numpy",
    },

    # 9. json.loads on a file path instead of file contents
    {
        "label": "json_loads_on_path",
        "code": '''\
import json
def load_config(filepath):
    config = json.loads(filepath)
    return config
''',
        "correct_code": '''\
import json
def load_config(filepath):
    with open(filepath, 'r') as f:
        config = json.load(f)
    return config
''',
        "explanation": (
            "json.loads() parses a JSON string, not a file path. Passing a "
            "file path string like '/etc/config.json' to json.loads() will "
            "raise a JSONDecodeError. Need json.load(open(filepath)) or "
            "json.loads(open(filepath).read())."
        ),
        "library": "json",
    },

    # 10. os.path.join result treated as file object
    {
        "label": "path_join_as_file",
        "code": '''\
import os
def read_file(directory, filename):
    filepath = os.path.join(directory, filename)
    content = filepath.read()
    return content
''',
        "correct_code": '''\
import os
def read_file(directory, filename):
    filepath = os.path.join(directory, filename)
    with open(filepath, 'r') as f:
        content = f.read()
    return content
''',
        "explanation": (
            "os.path.join() returns a string, not a file object. Calling "
            ".read() on a string doesn't do file I/O — str.read() doesn't "
            "even exist (AttributeError). Need open(filepath).read()."
        ),
        "library": "os",
    },
]

# Also include correct examples for baseline comparison
CORRECT_EXAMPLES = [
    {
        "label": "correct_requests",
        "code": '''\
import requests
def get_data(url):
    resp = requests.get(url)
    data = resp.json()['results']
    return data
''',
        "library": "requests",
    },
    {
        "label": "correct_pandas",
        "code": '''\
import pandas as pd
def process(path):
    df = pd.read_csv(path)
    result = df.sort_values('date').head(10)
    return result
''',
        "library": "pandas",
    },
    {
        "label": "correct_json",
        "code": '''\
import json
def load_config(filepath):
    with open(filepath, 'r') as f:
        config = json.load(f)
    return config
''',
        "library": "json",
    },
]


if __name__ == "__main__":
    print(f"Loaded {len(INTENT_MISUSE_EXAMPLES)} Intent Misuse examples")
    print(f"Loaded {len(CORRECT_EXAMPLES)} correct examples")
    print()
    for i, ex in enumerate(INTENT_MISUSE_EXAMPLES, 1):
        print(f"  {i:2d}. [{ex['library']:8s}] {ex['label']}")
        print(f"      {ex['explanation'][:80]}...")
        print()
