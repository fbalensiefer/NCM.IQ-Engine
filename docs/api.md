# Local HTTP API

`ncm-iq serve` binds to `127.0.0.1:8000` by default. Sessions are stored in the configured SQLite database. A session is tied to its selected model.

| Method | Path | Request |
|---|---|---|
| GET | `/api/config` | Lists available model names, tasks, context limits, and backends. |
| POST | `/api/sessions` | `{"model":"xsum"}` creates a session at revision 0. |
| GET | `/api/sessions/{id}` | Returns the session, its revision, and all saved turns. |
| POST | `/api/sessions/{id}/messages` | Requires `text` and the current `revision`. Returns the updated session and generation stats. |
| POST | `/api/sessions/{id}/reset` | Requires the current `revision`. Clears the turns and returns the updated session. |

Message requests can override `backend`, `profile`, `context_limit`, `history_turns`, `temperature`, `top_k`, `top_p`, and `cfg_scale`. The defaults are automatic backend selection, Balanced inference, and eight previous turns for dialogue. XSum uses the current message as its source passage. Generation stats include timings, checkpoint bytes, and peak process RSS, with allocator measurements on MPS.

Each successful message or reset advances the revision. A stale revision returns HTTP 409. Retrieve the session before retrying.

By default, `stats.milliseconds` contains only `total`, measured through device completion. Set `"profile_timings": true` in a message request to collect the synchronized stage breakdown. The web UI exposes this as **Measure individual inference stages**. The additional synchronization can increase request latency.

```sh
curl -fsS http://127.0.0.1:8000/api/config
session=$(curl -fsS http://127.0.0.1:8000/api/sessions \
  -H 'Content-Type: application/json' -d '{"model":"xsum"}')
id=$(printf '%s' "$session" | jq -r .id)
curl -fsS "http://127.0.0.1:8000/api/sessions/$id/messages" \
  -H 'Content-Type: application/json' \
  -d '{"text":"The library opens its reading room on Monday.","revision":0}'
```
