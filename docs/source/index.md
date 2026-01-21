## Documentation

Documentation for project_name

## Training API

Run the API:

```bash
uv run uvicorn rice_cnn_classifier.api:app --host 0.0.0.0 --port 8000
```

Start training:

```bash
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d '{"epochs": 5}'
```

Check status:

```bash
curl http://localhost:8000/train/<job_id>
```
