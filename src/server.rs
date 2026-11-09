use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path as UrlPath, State},
    http::{StatusCode, header},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::BTreeMap,
    fs,
    net::Ipv4Addr,
    path::PathBuf,
    sync::{Arc, Mutex},
};
use tch::Device;

use crate::{
    config::Config,
    device::{Backend, Profile, backends, device},
    engine::Engine,
    session::{Session, SessionStore, Turn, dialogue_prompt},
};

#[derive(Clone, Serialize)]
struct ModelSpec {
    name: String,
    task: String,
    context_limit: usize,
    #[serde(skip)]
    directory: PathBuf,
}

type Loaded = (String, Device, Profile, Engine);

struct App {
    models: BTreeMap<String, ModelSpec>,
    vocabulary: PathBuf,
    sessions: SessionStore,
    active: Mutex<Option<Loaded>>,
}

struct ApiError(StatusCode, String);
type ApiResult<T> = std::result::Result<T, ApiError>;

impl ApiError {
    fn invalid(message: impl Into<String>) -> Self {
        Self(StatusCode::UNPROCESSABLE_ENTITY, message.into())
    }
    fn missing(message: impl Into<String>) -> Self {
        Self(StatusCode::NOT_FOUND, message.into())
    }
    fn conflict() -> Self {
        Self(
            StatusCode::CONFLICT,
            "Session changed. Reload it and retry.".into(),
        )
    }
    fn internal(error: impl std::fmt::Display) -> Self {
        eprintln!("NCM request failed: {error}");
        Self(
            StatusCode::SERVICE_UNAVAILABLE,
            "Request failed. Check the server log.".into(),
        )
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({"detail": self.1}))).into_response()
    }
}

#[derive(Deserialize)]
struct NewSession {
    model: String,
}

#[derive(Deserialize)]
struct Reset {
    revision: i64,
}

#[derive(Deserialize)]
struct Message {
    text: String,
    revision: i64,
    #[serde(default)]
    backend: Backend,
    #[serde(default)]
    profile: Profile,
    context_limit: Option<usize>,
    history_turns: Option<usize>,
    temperature: Option<f64>,
    top_k: Option<i64>,
    top_p: Option<f64>,
    cfg_scale: Option<f64>,
}

impl App {
    fn current(&self, id: String, revision: Option<i64>) -> ApiResult<Session> {
        let session = self
            .sessions
            .get(id)
            .map_err(ApiError::internal)?
            .ok_or_else(|| ApiError::missing("Session not found"))?;
        if revision.is_some_and(|r| r != session.revision) {
            return Err(ApiError::conflict());
        }
        Ok(session)
    }

    fn replace(&self, session: Session, turns: Vec<Turn>) -> ApiResult<Session> {
        self.sessions
            .replace(session, turns)
            .map_err(ApiError::internal)?
            .ok_or_else(ApiError::conflict)
    }

    fn message(&self, id: String, request: Message) -> ApiResult<Json<serde_json::Value>> {
        let session = self.current(id, Some(request.revision))?;
        let spec = self
            .models
            .get(&session.model)
            .ok_or_else(|| ApiError::missing("Model is unavailable"))?;
        let text = request.text.trim();
        if text.is_empty() || text.chars().count() > 30_000 {
            return Err(ApiError::invalid("Enter between 1 and 30000 characters"));
        }
        if session.turns.len() >= 250 {
            return Err(ApiError(
                StatusCode::CONFLICT,
                "Reset the session or start a new one".into(),
            ));
        }
        let turns = request.history_turns.unwrap_or(8);
        if turns > 24
            || request
                .context_limit
                .is_some_and(|n| !(1..=spec.context_limit).contains(&n))
        {
            return Err(ApiError::invalid("Invalid context or history limit"));
        }
        let prompt = if spec.task == "dialogue" {
            dialogue_prompt(
                &session.turns[session.turns.len().saturating_sub(turns)..],
                text,
            )
        } else {
            text.to_owned()
        };
        let wanted = device(request.backend, request.profile)
            .map_err(|e| ApiError::invalid(e.to_string()))?;
        let result = {
            let mut active = self.active.lock().map_err(ApiError::internal)?;
            if active.as_ref().is_none_or(|(name, device, profile, _)| {
                name != &spec.name || *device != wanted || *profile != request.profile
            }) {
                *active = None;
                let engine = Engine::load(
                    &spec.directory,
                    &self.vocabulary,
                    request.backend,
                    request.profile,
                )
                .map_err(ApiError::internal)?;
                *active = Some((spec.name.clone(), wanted, request.profile, engine));
            }
            let engine = &active.as_ref().unwrap().3;
            let mut sampling = engine.config.generation.sampling;
            sampling.temperature = request.temperature.unwrap_or(sampling.temperature);
            sampling.top_k = request.top_k.unwrap_or(sampling.top_k);
            sampling.top_p = request.top_p.unwrap_or(sampling.top_p);
            sampling.cfg_scale = request.cfg_scale.unwrap_or(sampling.cfg_scale);
            sampling
                .validate(&engine.config.model.vocab_size)
                .map_err(|e| ApiError::invalid(e.to_string()))?;
            engine
                .generate(&prompt, sampling, request.context_limit, true)
                .map_err(ApiError::internal)?
        };
        let mut turns = session.turns.clone();
        turns.push(Turn {
            user: text.to_owned(),
            assistant: result.text,
        });
        let session = self.replace(session, turns)?;
        Ok(Json(json!({"session": session, "stats": result.stats})))
    }
}

async fn blocking<T: Send + 'static>(
    run: impl FnOnce() -> ApiResult<T> + Send + 'static,
) -> ApiResult<T> {
    tokio::task::spawn_blocking(run)
        .await
        .map_err(ApiError::internal)?
}

pub async fn serve(
    models: PathBuf,
    vocabulary: PathBuf,
    sessions: PathBuf,
    port: u16,
) -> Result<()> {
    let mut specs = BTreeMap::new();
    for entry in
        fs::read_dir(&models).with_context(|| format!("reading models in {}", models.display()))?
    {
        let entry = entry?;
        let directory = entry.path();
        if let Ok(config) = Config::read(&directory)
            && directory.join(&config.checkpoints.generator).is_file()
            && directory.join(&config.checkpoints.tokenizer).is_file()
            && let Ok(name) = entry.file_name().into_string()
        {
            specs.insert(
                name.clone(),
                ModelSpec {
                    name,
                    task: config.task,
                    context_limit: config.model.prefix_len as usize,
                    directory,
                },
            );
        }
    }
    let state = Arc::new(App {
        models: specs,
        vocabulary,
        sessions: SessionStore::open(sessions)?,
        active: Mutex::new(None),
    });
    let router = Router::new()
        .route("/", get(|| async { Html(include_str!("../frontend/index.html")) }))
        .route("/static/app.js", get(|| async {
            ([(header::CONTENT_TYPE, "text/javascript; charset=utf-8")], include_str!("../frontend/app.js"))
        }))
        .route("/static/style.css", get(|| async {
            ([(header::CONTENT_TYPE, "text/css; charset=utf-8")], include_str!("../frontend/style.css"))
        }))
        .route("/api/config", get(|State(app): State<Arc<App>>| async move {
            Json(json!({"models": app.models.values().collect::<Vec<_>>(), "backends": backends()}))
        }))
        .route("/api/sessions", post(|State(app): State<Arc<App>>, Json(request): Json<NewSession>| async move {
            blocking(move || {
                if !app.models.contains_key(&request.model) { return Err(ApiError::missing("Model is unavailable")); }
                Ok((StatusCode::CREATED, Json(app.sessions.create(request.model).map_err(ApiError::internal)?)))
            }).await
        }))
        .route("/api/sessions/{id}", get(|State(app): State<Arc<App>>, UrlPath(id): UrlPath<String>| async move {
            blocking(move || app.current(id, None).map(Json)).await
        }))
        .route("/api/sessions/{id}/reset", post(|State(app): State<Arc<App>>, UrlPath(id): UrlPath<String>, Json(request): Json<Reset>| async move {
            blocking(move || app.replace(app.current(id, Some(request.revision))?, Vec::new()).map(Json)).await
        }))
        .route("/api/sessions/{id}/messages", post(|State(app): State<Arc<App>>, UrlPath(id): UrlPath<String>, Json(request): Json<Message>| async move {
            blocking(move || app.message(id, request)).await
        }))
        .layer(DefaultBodyLimit::max(256 * 1024))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind((Ipv4Addr::LOCALHOST, port)).await?;
    eprintln!("NCM Chat listening on http://{}", listener.local_addr()?);
    axum::serve(listener, router).await?;
    Ok(())
}
