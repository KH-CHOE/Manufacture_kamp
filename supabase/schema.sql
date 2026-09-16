-- Run once in Supabase Dashboard → SQL Editor → New query.
create table if not exists public.process_events (
  event_id bigint generated always as identity primary key,
  event_at timestamp without time zone not null,
  pass_or_fail smallint check (pass_or_fail in (0, 1)),
  probability double precision not null check (probability >= 0 and probability <= 1),
  values_json jsonb not null,
  is_dashboard_sample boolean not null default false,
  created_at timestamptz not null default now()
);

create index if not exists process_events_dashboard_idx
  on public.process_events (is_dashboard_sample, event_at);

alter table public.process_events enable row level security;

-- The browser never calls this table directly. FastAPI uses the server-only
-- secret key; no public SELECT/INSERT policy is intentionally created.
revoke all on table public.process_events from anon, authenticated;
