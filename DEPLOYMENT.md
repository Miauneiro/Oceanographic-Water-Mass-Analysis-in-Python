# Deployment Guide

## Local Development Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning repository)

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/Miauneiro/Oceanographic-Water-Mass-Analysis-in-Python.git
cd Oceanographic-Water-Mass-Analysis-in-Python
```

2. Create virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

Launch the Streamlit web interface:
```bash
streamlit run app.py
```

The application will open automatically in your default browser at `http://localhost:8501`.

## Cloud Deployment Options

### Streamlit Cloud (Recommended for Public Demos)

1. Push repository to GitHub
2. Visit https://streamlit.io/cloud
3. Sign in with GitHub
4. Click "New app"
5. Select your repository and `app.py`
6. Deploy

**Advantages:**
- Free tier available
- Automatic HTTPS
- Easy sharing with public URL
- Integrated with GitHub for automatic updates

### Heroku Deployment

Create additional files:

**Procfile:**
```
web: sh setup.sh && streamlit run app.py
```

**setup.sh:**
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

Deploy:
```bash
heroku login
heroku create your-app-name
git push heroku main
```

### Docker Containerization

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for Cartopy
RUN apt-get update && apt-get install -y \
    libgeos-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t oceanographic-analysis .
docker run -p 8501:8501 oceanographic-analysis
```

## Production Considerations

### Performance Optimization

1. **Data caching**: Streamlit automatically caches function results
2. **Memory management**: Application cleans up temporary files after processing
3. **Parallel processing**: For large datasets, consider adding multiprocessing for CTD file reading

### Security

1. **File upload limits**: Configure in Streamlit settings
```toml
# .streamlit/config.toml
[server]
maxUploadSize = 200  # MB
```

2. **Input validation**: Already implemented for NetCDF variable checking
3. **Sanitization**: File paths are controlled via tempfile module

### Monitoring

For production deployments, add:
- Error logging with Python's logging module
- Performance metrics tracking
- User analytics (if required)

### Scaling

**For institutional deployment:**
- Deploy behind nginx reverse proxy
- Use Kubernetes for auto-scaling
- Implement Redis caching for repeated analyses
- Database integration for storing analysis results

## Troubleshooting

### Cartopy Installation Issues

If Cartopy fails to install:

**On Ubuntu/Debian:**
```bash
sudo apt-get install libgeos-dev libproj-dev
pip install cartopy
```

**On macOS:**
```bash
brew install proj geos
pip install cartopy
```

**On Windows:**
Use conda instead of pip:
```bash
conda install -c conda-forge cartopy
```

### NetCDF Library Issues

**Linux:**
```bash
sudo apt-get install libnetcdf-dev libhdf5-dev
```

**macOS:**
```bash
brew install netcdf hdf5
```

### Memory Issues with Large Datasets

If processing many large files:
1. Reduce interpolation resolution (fewer Znew points)
2. Process files in batches
3. Increase Docker memory limits if using containers

## API Integration

For programmatic access, the core module can be imported:

```python
from oceanography import ler_perfis, calcular_densidade, plotar_TS

# Your analysis code here
```

This enables integration into:
- Automated data processing pipelines
- Research vessel real-time analysis systems
- Climate model validation workflows
- Operational oceanography systems

## Maintenance

### Updating Dependencies

Check for updates:
```bash
pip list --outdated
```

Update specific packages:
```bash
pip install --upgrade streamlit gsw
```

### Testing

Before deploying updates:
1. Test with sample NetCDF files
2. Verify all visualization options work
3. Check water mass mixing calculations
4. Test with edge cases (missing data, single profile, etc.)

## Support

For issues or questions:
- GitHub Issues: https://github.com/Miauneiro/Oceanographic-Water-Mass-Analysis-in-Python
- Email: joaofteixeiramanero@gmail.com
- Documentation: This README and inline code comments
