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

### Vertex AI settings

The API submits Vertex AI custom jobs. Configure these environment variables for the API service:

- `VERTEX_PROJECT_ID`
- `VERTEX_REGION`
- `TRAIN_IMAGE_URI`

You can also pass `project_id`, `region`, and `image_uri` in the request body.
