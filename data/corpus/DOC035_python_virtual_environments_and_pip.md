# Python Virtual Environments and pip

*Source: Open technical documentation and standard references (public knowledge, paraphrased)*

This document gives a short overview of Python Virtual Environments and pip.

## Key points

A virtual environment is an isolated folder that keeps the packages of one Python project separate from other projects. To create a virtual environment named venv, run the command python -m venv venv. On Windows the environment is activated by running venv\Scripts\activate.

On Linux and macOS the environment is activated with source venv/bin/activate. The command pip install -r requirements.txt installs every package listed in the requirements file. The command pip freeze > requirements.txt saves the currently installed packages and versions into a file.

The command deactivate leaves the virtual environment. The command pip list shows all packages installed in the active environment.

> Note: This is a simplified educational summary written in our own words from public information. It is not legal, financial or medical advice. Verify details with the official source before relying on them.
