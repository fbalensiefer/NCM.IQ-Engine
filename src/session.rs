use anyhow::Result;
use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
    time::Duration,
};
use uuid::Uuid;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Turn {
    pub user: String,
    pub assistant: String,
}

#[derive(Debug, Serialize)]
pub struct Session {
    pub id: String,
    pub model: String,
    pub revision: i64,
    pub turns: Vec<Turn>,
}

pub fn dialogue_prompt(turns: &[Turn], message: &str) -> String {
    use std::fmt::Write;
    let mut prompt = String::new();
    for turn in turns {
        let _ = writeln!(prompt, "User: {}\nAssistant: {}", turn.user, turn.assistant);
    }
    let _ = write!(prompt, "User: {message}\nAssistant:");
    prompt
}

pub struct SessionStore(PathBuf);

impl SessionStore {
    pub fn open(path: PathBuf) -> Result<Self> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        let store = Self(path);
        store.connection()?.execute_batch(
            "PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY, model TEXT NOT NULL, revision INTEGER NOT NULL, turns TEXT NOT NULL
            );"
        )?;
        Ok(store)
    }

    fn connection(&self) -> Result<Connection> {
        let conn = Connection::open(Path::new(&self.0))?;
        conn.busy_timeout(Duration::from_secs(10))?;
        Ok(conn)
    }

    pub fn create(&self, model: String) -> Result<Session> {
        let id = Uuid::new_v4().simple().to_string();
        self.connection()?.execute(
            "INSERT INTO sessions VALUES (?1, ?2, 0, '[]')",
            (&id, &model),
        )?;
        Ok(Session {
            id,
            model,
            revision: 0,
            turns: Vec::new(),
        })
    }

    pub fn get(&self, id: String) -> Result<Option<Session>> {
        let entry = self
            .connection()?
            .query_row(
                "SELECT model, revision, turns FROM sessions WHERE id = ?1",
                [&id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                },
            )
            .optional()?;
        if let Some((model, revision, turns)) = entry {
            Ok(Some(Session {
                id,
                model,
                revision,
                turns: serde_json::from_str(&turns)?,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn replace(&self, mut session: Session, turns: Vec<Turn>) -> Result<Option<Session>> {
        let updated = self.connection()?.execute(
            "UPDATE sessions SET turns = ?1, revision = revision + 1 WHERE id = ?2 AND revision = ?3",
            (serde_json::to_string(&turns)?, &session.id, session.revision),
        )?;
        if updated == 0 {
            return Ok(None);
        }
        session.revision += 1;
        session.turns = turns;
        Ok(Some(session))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_turns_and_rejects_stale_updates() -> Result<()> {
        let file = std::env::temp_dir().join(format!("ncm-test-{}.db", Uuid::new_v4()));
        let store = SessionStore::open(file.clone())?;
        let session = store.create("dialogue".into())?;
        let id = session.id.clone();
        let stale = store.get(id.clone())?.unwrap();
        let updated = store
            .replace(
                session,
                vec![Turn {
                    user: "hi".into(),
                    assistant: "hello".into(),
                }],
            )?
            .unwrap();
        assert!(store.replace(stale, Vec::new())?.is_none());
        assert_eq!(updated.revision, 1);
        let reloaded = SessionStore::open(file.clone())?.get(id)?.unwrap();
        assert_eq!(
            dialogue_prompt(&reloaded.turns, "again"),
            "User: hi\nAssistant: hello\nUser: again\nAssistant:"
        );
        fs::remove_file(file)?;
        Ok(())
    }
}
