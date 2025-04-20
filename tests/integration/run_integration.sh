rm -rf .venv
uv venv
uv pip install -e ".[dev]"
source .venv/bin/activate

docker-compose up -d
sleep 30

pytest tests/integration/test_hybrid_pipeline_integration.py -v
deactivate
rm -rf .venv
docker-compose down -v