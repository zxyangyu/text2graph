## Install

**Install**

```shell
# clone this repo first
cd text2graph
pip install -e .
```

### Function: `insert`

**Description:**
This function takes a list of strings representing the documents and returns a json object representing the result of the extraction.

**Parameters:**
- `scope` (list[str]): A list of strings representing the documents to be inserted.

**Returns:**
- `dict`: A json object representing the result of the extraction.

**Example Usage:**
```python

with open("book.txt", encoding="utf-8-sig") as f:
    scope = f.read()
result = extraction([scope])
print(result)

```
**run demo**
```shell
cd demo
python main.py
```