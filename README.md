# dlkit

Simple deep learning toolkit for stock forcasting.

## Manage dependencies

Poetry is used to manage the project. Dependencies are devide in 
groups: main (default, runtime dependencies), test, dev. 
Install all dependencies by:

```shell
poetry install 
```

or install groups optionally:

```shell
# the runtime dependencies is the implicit group main
poetry install --only main 
# poetry install --with test
```

## Test

All testings are done in nox, which are defined in noxfile.py (nox config script). See all the sessions defined:

```shell
nox --list
```

Run all tests:

```shell
nox
```

Run one specific session:

```shell
nox --sessions lint_and_format
```
