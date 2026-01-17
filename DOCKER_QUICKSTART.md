# 🐳 Docker Quick Start

## One-Command Setup

```bash
docker-compose up --build
```

Then open: **http://localhost:8501**

## What Gets Installed Automatically

✅ All packages from `requirements.txt`:
- Streamlit
- PyTorch & torchvision  
- MediaPipe 0.10.13
- NumPy, Pandas, SciPy
- scikit-learn
- Pillow
- All dependencies

✅ System libraries:
- OpenGL (for MediaPipe)
- Graphics libraries
- Runtime dependencies

## Prerequisites

1. **Docker installed** - [Get Docker](https://docs.docker.com/get-docker/)
2. **Model file** - Place `hybrid_model_v2.pth` in `models/` directory

## Commands

```bash
# Start
docker-compose up --build

# Stop
docker-compose down

# View logs
docker-compose logs -f

# Rebuild
docker-compose up --build --force-recreate
```

## Troubleshooting

**Port in use?** Change port in `docker-compose.yml`:
```yaml
ports:
  - "8502:8501"  # Use 8502 instead
```

**Model not found?** Ensure `models/hybrid_model_v2.pth` exists

**MediaPipe errors?** All deps install automatically. Check logs:
```bash
docker-compose logs bmi-predictor
```

---

**That's it! All dependencies are handled automatically.** 🎉
