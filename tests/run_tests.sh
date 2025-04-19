#!/usr/bin/env bash
# Setup a virtual environment for testing
rm -rf .venv/
uv venv
uv pip install -e ".[dev]"
uv pip install coverage  # Install coverage package
source .venv/bin/activate

# Run tests with coverage
echo "Running tests with coverage..."
coverage erase  # Clear any previous coverage data

# Run tests with coverage

coverage run --append -m unittest tests/test_hybrid_pipeline_config.py
coverage run --append -m unittest tests/test_hybrid_pipeline.py

# Generate coverage reports
echo "Generating coverage report..."
coverage report -m
coverage html

# Displaying the location of the HTML report
echo "HTML coverage report generated in htmlcov/ directory"
echo "Open htmlcov/index.html in your browser to view it"

# Cleanup
deactivate
rm -rf .venv/