# Start With Docker

Send the whole project folder to the user, not only the Dockerfile.

The important Docker files are:

- `Dockerfile`: builds the Python/FastAPI API image.
- `docker-compose.yml`: starts both the API and MongoDB.
- `.dockerignore`: keeps local cache/virtualenv files out of the Docker image.

## Requirements

The user must install Docker Desktop first:

https://www.docker.com/products/docker-desktop/

## Start The Project

Open PowerShell in this project folder and run:

```powershell
docker compose up --build
```

This starts:

- MongoDB on port `27017`
- API server on port `8000`

Open:

```text
http://localhost:8000/health
http://localhost:8000/docs
```

## Start In Background

```powershell
docker compose up -d --build
```

## Stop Everything

```powershell
docker compose down
```

## See Logs

```powershell
docker compose logs -f
```

## Optional Configuration

The project can start without a `.env` file.

If the user needs ERP, camera, or recognition settings, copy:

```powershell
copy .env.example .env
```

Then edit `.env` and restart:

```powershell
docker compose down
docker compose up --build
```
