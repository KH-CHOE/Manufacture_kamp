# Supabase 연결 안내

이 프로젝트는 현재 SQLite로도 실행된다. 아래 과정을 마치면 Vercel 배포 환경에서는 Supabase를 사용하고, 로컬에서는 기존 SQLite를 계속 사용할 수 있다.

## 1. 테이블 만들기

1. Supabase 프로젝트를 연다.
2. 왼쪽 메뉴에서 **SQL Editor**를 누른다.
3. **New query**를 누른다.
4. [`supabase/schema.sql`](../supabase/schema.sql) 전체를 붙여 넣는다.
5. **Run**을 누른다.
6. 왼쪽 **Table Editor**에서 `process_events` 테이블이 보이는지 확인한다.

이 테이블은 브라우저에서 직접 읽지 않는다. Row Level Security가 켜져 있으며, FastAPI 서버만 서버 전용 키로 접근한다.

## 2. 실제 이력 적재

이 단계는 로컬 SQLite의 73,611건 이력을 Supabase로 복사한다. 실제 제조 데이터가 Supabase 프로젝트에 저장되는 단계이므로, 공유 범위를 확인한 뒤 실행한다.

1. Supabase의 **Project Settings → API Keys**에서 **Secret key**를 확인한다.
2. 프로젝트 폴더에서 아래 명령을 실행한다. `SUPABASE_SECRET_KEY` 자리에는 키를 붙여 넣지만, 채팅·GitHub·소스 코드에는 저장하지 않는다.

```bash
export SUPABASE_URL='https://aftudpjdnjxexnkukeyb.supabase.co'
export SUPABASE_SECRET_KEY='여기에_Secret_key_입력'
python3 scripts/seed_supabase.py --replace
```

3. 완료 메시지 `Copied 73,611 events to Supabase.`를 확인한다.
4. **Table Editor → process_events**에서 행이 들어왔는지 확인한다.
5. 터미널을 닫거나 아래 명령으로 키를 지운다.

```bash
unset SUPABASE_SECRET_KEY
```

`--replace`는 Supabase의 기존 `process_events` 데이터를 지운 뒤 현재 SQLite 이력으로 다시 적재한다. 최초 1회에는 안전하며, 이후에는 의도적으로 재적재할 때만 사용한다.

## 3. Vercel 배포 때 등록할 환경 변수

Vercel 프로젝트의 **Settings → Environment Variables**에 아래 값을 Production과 Preview에 등록한다.

| 이름 | 값 | 공개 여부 |
|---|---|---|
| `SUPABASE_URL` | `https://aftudpjdnjxexnkukeyb.supabase.co` | 서버 설정 |
| `SUPABASE_SECRET_KEY` | Supabase Secret key | 비밀. 브라우저·GitHub·채팅 금지 |
| `SUPABASE_PUBLISHABLE_KEY` | `sb_publishable_...` | 브라우저 인증을 추가할 때 사용 |

현재 대시보드는 브라우저가 Supabase를 직접 호출하지 않는다. 따라서 Publishable key는 다음 단계의 로그인 기능 전까지는 사용하지 않는다.

## 4. 배포 전 확인할 선택

- 실제 제조 이력과 학습 모델을 GitHub/Vercel에 올려도 되는지
- URL을 아는 사람 모두가 보는 공개 대시보드인지, 친구 이메일 로그인 방식인지

실제 데이터 공유가 부담되면 Vercel에는 `share/foundry_guard_demo.html`만 배포하고, 실제 대시보드는 로그인 기능을 만든 뒤 배포한다.
