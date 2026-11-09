#!/usr/bin/env bash
# Exercise the Rust server with the locally trained badge-dialogue checkpoint.
set -euo pipefail
binary=${1:-target/release/ncm-chat}
model=${2:-rust-badge-dialogue}
temp=$(mktemp -d)
server_pid=
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  rm -rf "$temp"
}
trap cleanup EXIT
"$binary" serve --port 0 --sessions "$temp/session.sqlite3" > "$temp/server.log" 2>&1 &
server_pid=$!
base=
for ((attempt=0; attempt<100; attempt++)); do
  base=$(sed -n 's/^NCM Chat listening on //p' "$temp/server.log" | head -1)
  if [[ -n "$base" ]]; then break; fi
  if ! kill -0 "$server_pid" 2>/dev/null; then cat "$temp/server.log" >&2; exit 1; fi
  sleep 0.1
done
[[ -n "$base" ]]
post() { curl -fsS -H 'Content-Type: application/json' -d "$2" "$base$1"; }
curl -fsS "$base/" | grep -q 'NCM Chat'
curl -fsS "$base/static/app.js" > "$temp/app.js"
curl -fsS "$base/api/config" | jq -e --arg model "$model" 'any(.models[]; .name == $model and .task == "dialogue")' >/dev/null

for color in teal purple orange; do
  session=$(post /api/sessions "$(jq -nc --arg model "$model" '{model:$model}')")
  api="/api/sessions/$(jq -r '.id' <<< "$session")"
  first=$(post "$api/messages" "$(jq -nc --arg color "$color" '{text:("Remember this badge color: " + $color + "."),revision:0,backend:"cpu"}')")
  jq -e '.session.revision == 1 and .stats.milliseconds.total > 0' <<< "$first" >/dev/null
  second=$(post "$api/messages" '{"text":"What badge color did I give you?","revision":1,"backend":"cpu"}')
  jq -e '.session.revision == 2 and (.session.turns | length) == 2' <<< "$second" >/dev/null
  expected=$(jq -c '.session' <<< "$second")
  actual=$(curl -fsS "$base$api" | jq -c .)
  jq -ne --argjson expected "$expected" --argjson actual "$actual" '$expected == $actual' >/dev/null
  conflict=$(curl -sS -o /dev/null -w '%{http_code}' -H 'Content-Type: application/json' -d '{"revision":1}' "$base$api/reset")
  [[ "$conflict" == 409 ]]
  post "$api/reset" '{"revision":2}' | jq -e '.revision == 3 and .turns == []' >/dev/null
  jq -c --arg color "$color" '{color:$color,turns:.session.turns}' <<< "$second"
done

session=$(post /api/sessions '{"model":"xsum"}')
api="/api/sessions/$(jq -r '.id' <<< "$session")"
post "$api/messages" '{"text":"The library is opening a new reading room for children on Monday after six months of renovation.","revision":0,"backend":"auto"}' |
  jq -e '.stats.milliseconds.neural_decoder > 0 and (.session.turns[0].assistant | length) > 0' >/dev/null
