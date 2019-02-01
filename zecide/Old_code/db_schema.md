### Database schema

#### Structure of the tree

Each factor is considered as a node in the tree. Node has the following values
 - All nodes have an X (True) value
 - Root node has only X (True) value
 - Node can have multiple children (sub-factors)
 - Only root node has no parent, all other nodes have parents
 - Calculating X value, the Y value are the X values of the child nodes

#### Database layout

```py
database = {

  'F':
  {
    'X': 'X value of the factor',
    'Z': 'Z value of the factor (calculated from the other features)',
    'factors': ['list of sub-factors (factors name)'],
    'parent': 'parent node (factor name)
  }

}
```

Data example

Simple structure - Single level (1 parent with 6 children)

```py

db =
{
  'F': {'X': 0.54, 'factors': ['f1', 'f2', 'f3', 'f4', 'f5', 'f6'], 'parent': None},
  'f1': {'X': 0.78, 'Z': 0.24, 'factors': [], 'parent': 'F'},
  'f2': {'X': 0.4212, 'Z': 0.46, 'factors': [], 'parent': 'F'},
  'f3': {'X': 0.36, 'Z': 0.22, 'factors': [], 'parent': 'F'},
  'f4': {'X': 0.21, 'Z': 0.47, 'factors': [], 'parent': 'F'},
  'f5': {'X': 0.36, 'Z': 0.31, 'factors': [], 'parent': 'F'},
  'f6': {'X': 0.61, 'Z': 0.23, 'factors': [], 'parent': 'F'}
}

```

![Simple](./Images/simple.jpg)

Complex structure - Multiple level

```py

db =
{
    'F': {'X': 0.54, 'factors': ['f1', 'f2'], 'parent': None},
    'f1': {'X': 0.78, 'Z': 0.24, 'factors': ['f3', 'f4'], 'parent': 'F'},
    'f2': {'X': 0.4212, 'Z': 0.46, 'factors': ['f5', 'f6'], 'parent': 'F'},
    'f3': {'X': 0.36, 'Z': 0.22, 'factors': [], 'parent': 'f1'},
    'f4': {'X': 0.21, 'Z': 0.47, 'factors': [], 'parent': 'f1'},
    'f5': {'X': 0.36, 'Z': 0.31, 'factors': [], 'parent': 'f2'},
    'f6': {'X': 0.61, 'Z': 0.23, 'factors': [], 'parent': 'f2'}}

```

![Complex](./Images/complex.jpg)
