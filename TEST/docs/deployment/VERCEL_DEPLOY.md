# Vercel 배포 — 가장 쉬운 경로

Supabase의 `process_events` 데이터 적재는 완료된 상태다. 이 프로젝트는 루트의 `app.py`를 FastAPI 진입점으로 사용하므로 Vercel에서 별도 빌드 명령이나 서버 설정이 필요 없다.

## 0. 배포 전 자동 점검

Finder에서 `scripts/배포_전_점검.command`를 더블클릭한다. `배포 전 점검 완료`가 나오면 다음 단계로 간다.

GitHub 없이도 이 폴더에서 Vercel로 바로 배포할 수 있다. GitHub는 친구와 협업할 때만 별도로 사용한다.

## 1. Vercel CLI 로그인

터미널에서 아래 명령을 실행한다. 브라우저가 열리면 Vercel 계정으로 로그인한다.

```bash
npx --yes vercel login
```

로그인이 끝나면 Codex에게 `Vercel 로그인 완료`라고 알린다. 프로젝트 생성과 배포는 Codex가 진행한다.

## 2. 환경변수 두 개 입력

Codex가 만든 Vercel 프로젝트의 **Settings → Environment Variables**에 아래 두 값을 각각 등록한다. Production과 Preview 모두 선택한다.

| 이름 | 값 |
|---|---|
| `SUPABASE_URL` | Supabase Project Settings → API에 있는 Project URL |
| `SUPABASE_SECRET_KEY` | Supabase Project Settings → API Keys의 Secret key |

`SUPABASE_PUBLISHABLE_KEY`는 현재 필요 없다. Secret key는 채팅에 보내지 않는다.

## 3. Codex가 배포·검증

환경변수 입력이 끝난 뒤 Codex에게 `환경변수 입력 완료`라고 알린다. Codex가 Production 배포와 `/api/health` 확인을 진행한다.

## 배포 후 확인

배포된 주소 뒤에 `/api/health`를 붙여 연다. 아래처럼 나오면 성공이다.

```json
{"ok": true, "model_ready": true, "storage": "supabase"}
```

`storage`가 `sqlite`이면 Vercel 환경변수 두 개 중 하나가 누락된 것이다. `model_ready`가 `false`이면 `artifacts/model/defect_model.pkl`이 GitHub 저장소에 포함되지 않은 것이다.
