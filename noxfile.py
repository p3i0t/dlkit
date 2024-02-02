"""
Nox is configured via a file called noxfile.py in your project’s directory. 
This file is a Python file that defines a set of sessions. 
A session is an environment and a set of commands to run in that environment. 


If you want to disable virtualenv creation altogether, you can set python to False,
or set venv_backend to "none", both are equivalent.
"""
import nox


@nox.session(
    python=[
        "3.10",
        "3.11",
    ],
    venv_backend="conda",
)
# @nox.session()
def lint_and_format(session):
    """lint with ruff, configurations are in pyproject.toml."""
    session.install("ruff")
    session.run("ruff", "check", "dlkit", "tests", "noxfile.py")
    session.run("ruff", "format", "dlkit", "tests", "noxfile.py")


@nox.session(
    python=[
        "3.10",
        "3.11",
    ],
    venv_backend="conda",
)
# @nox.session()
def test(session):
    """Unit test with pytest."""
    session.run("poetry", "install", external=True)
    # session.install("pytest")
    session.run("pytest", "tests", "-p", "no:warnings")
