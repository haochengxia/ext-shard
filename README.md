shard
=====

An C++ extension for sharded structure


Installation
------------

**On Unix (Linux, OS X)**

 - clone this repository
 - `pip install ./shard`

**On Windows (Requires Visual Studio 2015)**

 - For Python 3.5:
     - clone this repository
     - `pip install ./shard`

Manual Installation

```bash
$ python setup.py install
```

Building the documentation
--------------------------

Documentation for the example project is generated using Sphinx. Sphinx has the
ability to automatically inspect the signatures and documentation strings in
the extension module to generate beautiful documentation in a variety formats.
The following command generates HTML-based reference documentation; for other
formats please refer to the Sphinx manual:

 - `shard/docs`
 - `make html`


Running the tests
-----------------

Running the tests requires `pytest`.

```bash
pytest .
```
