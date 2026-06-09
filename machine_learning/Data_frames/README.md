# On-the-Job Training (OJT) Report

---

| Field               | Details                                        |
|---------------------|------------------------------------------------|
| **Trainee Name**    | Shubhada A. Palwe                              |
| **Training Domain** | Machine Learning — Data Handling               |
| **Topic**           | Pandas DataFrame                               |
| **Tools Used**      | Python, pandas                                 |
| **Total Versions**  | 5                                              |
|                             

---

## Objective

To learn how to use **Pandas DataFrame** — the most important 2D data structure in Python for Machine Learning. A DataFrame is like a full Excel table with rows and columns, and is the standard way to handle datasets before training a model.

---

## What is a DataFrame?

If a **Pandas Series** is a single column of data, a **Pandas DataFrame** is the whole table — multiple columns, each with its own name and data type.

```
Series  →  one column
DataFrame  →  many columns (full table)
```

A DataFrame is built from a **dictionary** where:
- Each **key** = a column name
- Each **value** = a list of values in that column

---

## Learning Progression — Version by Version

---

### Version 1 — `Pandas_Dataframe_1.py`
**Topic: Creating a DataFrame from a dictionary**

I created my first DataFrame using a Python dictionary and passed it to `pd.DataFrame()`.

```python
Data = {
    "Name": ["Sagar", "Amit", "Pooja"],
    "Age":  [23, 26, 25],
    "City": ["Pune", "Mumbai", "Satara"]
}

dobj = pd.DataFrame(Data)
print(dobj)
```

**What the output looks like:**

```
    Name  Age    City
0  Sagar   23    Pune
1   Amit   26  Mumbai
2  Pooja   25  Satara
```

**What I noticed:**
- Pandas automatically adds a row index (0, 1, 2) on the left side.
- Each dictionary key became a column header.
- Each list became a column of values.
- All columns are aligned neatly — it looks like a real table!

**Key learning:** A DataFrame = a dictionary converted into a table. The keys become column names, the lists become column data.

---

### Version 2 — `Pandas_Dataframe_2.py`
**Topic: Accessing a single column**

After printing the full table, I learned how to extract just one column using its name inside square brackets.

```python
print(dobj["Age"])
print("-------------")
print(dobj["City"])
```

**What the output looks like:**

```
0    23
1    26
2    25
Name: Age, dtype: int64
```

**What I noticed:**
- `dobj["Age"]` returns a **Pandas Series** — a single column with its index.
- This is the link between DataFrame and Series: a DataFrame is made up of multiple Series side by side.
- The column name and dtype are shown at the bottom of each output.

**Key learning:** `dobj["ColumnName"]` extracts one column as a Series. A DataFrame is essentially a collection of Series sharing the same index.

---

### Version 3 — `Pandas_Dataframe_3.py`
**Topic: Accessing multiple columns at once**

Instead of selecting one column, I selected two columns together by passing a **list of column names** inside the brackets.

```python
print(dobj[["Name", "City"]])
```

**What the output looks like:**

```
    Name    City
0  Sagar    Pune
1   Amit  Mumbai
2  Pooja  Satara
```

**What I noticed:**
- Double square brackets `[[ ]]` are used when selecting multiple columns.
- Single bracket `[ ]` → returns a Series (one column).
- Double bracket `[[ ]]` → returns a DataFrame (subset of columns).
- The Age column is completely hidden — we only see what we asked for.

**Key learning:** Use `dobj[["Col1", "Col2"]]` to get a sub-table with only the columns you need. This is called **column selection** or **column subsetting**.

---

### Version 4 — `Pandas_Dataframe_4.py`
**Topic: Accessing a specific row using `.loc[]`**

Now instead of selecting columns, I selected a specific **row** using `.loc[]` with the row index number.

```python
print(dobj.loc[0])  # fetch row at index 0
```

**What the output looks like:**

```
Name    Sagar
Age        23
City     Pune
Name: 0, dtype: object
```

**What I noticed:**
- `.loc[0]` returns the first row — but displayed vertically (each column name on the left, its value on the right).
- This is still a Series — now the index is the column names, not row numbers.
- `.loc` stands for **label-based location**. You access rows by their index label.

**Key learning:** `.loc[row_index]` fetches a full row. Rows and columns both have labels in a DataFrame — `.loc` lets you navigate by those labels.

---

### Version 5 — `Pandas_Dataframe_5.py`
**Topic: Using `.loc[]` to select all rows of a specific column**

This was the most powerful version. I used `.loc` with a **slice** to say: "give me all rows, but only the Name column."

```python
print(dobj.loc[:, "Name"])
```

**What the output looks like:**

```
0    Sagar
1     Amit
2    Pooja
Name: Name, dtype: object
```

**What I noticed:**
- `.loc[:, "Name"]` means: all rows (`:`) + only the "Name" column.
- The `:` means "select everything" — same as in Python list slicing.
- This gives the same result as `dobj["Name"]`, but using `.loc` gives more control (you can combine row and column selection in one step).
- `.loc[row_selector, column_selector]` is the full syntax.

**Key learning:** `.loc[:, "Column"]` is the precise way to select a column using row+column indexing together. The `:` means all rows. This prepares you for filtering — e.g., `.loc[0:1, "Name"]` to get only first 2 rows of the Name column.

---

## Summary Table

| Version | Code Used | What It Does |
|---------|-----------|--------------|
| 1 | `pd.DataFrame(dict)` | Create a full table from a dictionary |
| 2 | `dobj["Age"]` | Get one column as a Series |
| 3 | `dobj[["Name","City"]]` | Get multiple columns as a DataFrame |
| 4 | `dobj.loc[0]` | Get one full row |
| 5 | `dobj.loc[:, "Name"]` | Get all rows of one column using `.loc` |

---

## DataFrame vs Series — Quick Comparison

| Feature | Pandas Series | Pandas DataFrame |
|---------|--------------|-----------------|
| Dimensions | 1D (one column) | 2D (rows + columns) |
| Created from | List or array | Dictionary |
| Access by column | — | `dobj["col"]` |
| Access by row | — | `dobj.loc[0]` |
| Used for | Single feature / label | Full dataset |

---

## How DataFrame Fits in the ML Pipeline

| ML Step | How DataFrame Helps |
|---------|-------------------|
| Data Collection | Load CSV, Excel, or JSON into a DataFrame |
| Data Analysis | Use `.describe()`, `.info()`, `.head()` to explore |
| Data Cleaning | Remove nulls, fix data types, rename columns |
| Feature Engineering | Create new columns from existing ones |
| Model Input | Split into `X` (features) and `Y` (labels) from the DataFrame |

---

## Observations & Reflections

- The DataFrame is the **core data structure** of all real-world ML projects. Every dataset you load (CSV, database, API) becomes a DataFrame.
- I now understand the relationship: **Dictionary → DataFrame → Series (each column)**. They all connect.
- `.loc[]` is more powerful than simple bracket access — it lets you select by both row AND column in one step, which will become essential when filtering data.
- The double bracket `[[ ]]` trick for selecting multiple columns was a small thing that felt easy to miss but is used constantly.

---

## What's Next

- Load a real CSV file into a DataFrame using `pd.read_csv()`.
- Use `.head()`, `.tail()`, `.info()`, `.describe()` for quick data analysis.
- Filter rows using conditions: `dobj[dobj["Age"] > 24]`.
- Use `.iloc[]` (index-based location) vs `.loc[]` (label-based location).
- Apply DataFrame to the Iris dataset as X and Y for model training.

---

## Files

| File | Concept |
|------|---------|
| `Pandas_Dataframe_1.py` | Create DataFrame from dictionary |
| `Pandas_Dataframe_2.py` | Access single column with `[ ]` |
| `Pandas_Dataframe_3.py` | Access multiple columns with `[[ ]]` |
| `Pandas_Dataframe_4.py` | Access a row with `.loc[index]` |
| `Pandas_Dataframe_5.py` | Access all rows of a column with `.loc[:, col]` |

---


README.md: OJT report with version-by-version breakdown,
Series vs DataFrame comparison, ML pipeline mapping, and next steps.

Trainee: Shubhada A. Palwe | Tools: Python, pandas
```

---

*OJT Report — Machine Learning Track | Topic: Pandas DataFrame*
*Trainee: Shubhada A. Palwe*
