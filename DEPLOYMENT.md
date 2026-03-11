# Streamlit Cloud Deployment Guide

## Quick Start

### 1. Create `.streamlit/secrets.toml` (LOCAL ONLY - NOT IN GIT)
```toml
api_base = "http://localhost:8000"
newsapi_key = "3ebf5a2ef01b41e2aa4812ba8421f9b5"
```

### 2. Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io
2. Click "Create app"
3. Connect GitHub
   - **Repo:** llm-financial-news-analyzer
   - **Branch:** main
   - **Main file:** streamlit_app.py
4. After deploy, click Settings (gear ⚙️)
5. Click "Secrets" 
6. Paste:
```toml
api_base = "https://your-deployed-fastapi.herokuapp.com"
newsapi_key = "3ebf5a2ef01b41e2aa4812ba8421f9b5"
```

### 3. Deploy FastAPI Backend

Choose one:

**Option A: Heroku** (Free tier deprecated, but still works for existing apps)
```bash
cd c:\Users\dell\Documents\llm-financial-news-analyzer
heroku create your-app-name
heroku config:set NEWSAPI_KEY="3ebf5a2ef01b41e2aa4812ba8421f9b5"
git push heroku main
```

**Option B: Railway.app** (Recommended - free tier)
1. Connect GitHub repo
2. Select branch: `main`
3. Set root directory: (leave empty)
4. In Railway dashboard, add env vars:
   - `NEWSAPI_KEY=3ebf5a2ef01b41e2aa4812ba8421f9b5`
5. Railway auto-deploys on git push

**Option C: Render.com** 
1. Connect GitHub
2. Create "Web Service"
3. Build: `pip install -r requirements.txt`
4. Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### 4. Update Streamlit Secrets with Backend URL

After backend is deployed, get its URL (e.g., `https://my-api.railway.app`)

In Streamlit Cloud Settings → Secrets:
```toml
api_base = "https://my-api.railway.app"
newsapi_key = "3ebf5a2ef01b41e2aa4812ba8421f9b5"
```

---

## Troubleshooting

**If Streamlit deployment fails:**
- Check app logs in Streamlit Cloud dashboard
- Make sure `streamlit_requirements.txt` exists or update `requirements.txt` to be minimal
- Ensure `.streamlit/config.toml` and `streamlit_app.py` exist

**If API calls fail:**
- Verify `api_base` URL is correct in secrets
- Check backend is running
- Test API with: `curl https://your-api.com/docs`

**Port/Connection issues:**
- FastAPI must accept `0.0.0.0` (not localhost)
- Streamlit Cloud can't reach localhost - needs public URL
