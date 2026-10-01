#!/bin/zsh

cd "$(dirname "$0")/.." || exit 1
export SUPABASE_URL='https://aftudpjdnjxexnkukeyb.supabase.co'

echo ''
echo 'Supabase Secret key를 붙여넣고 Enter를 누르세요.'
echo '입력 중에는 글자가 보이지 않는 것이 정상입니다.'
read -s "SUPABASE_SECRET_KEY?Secret key: "
export SUPABASE_SECRET_KEY
echo ''
echo '제조 이력 73,611건을 Supabase로 업로드합니다. 잠시 기다려 주세요...'

python3 scripts/seed_supabase.py --replace
result=$?
unset SUPABASE_SECRET_KEY

if [ $result -eq 0 ]; then
  echo ''
  echo '완료되었습니다. Supabase Table Editor에서 process_events를 확인해 주세요.'
else
  echo ''
  echo '업로드에 실패했습니다. 이 창을 닫지 말고 오류 메시지를 Codex에 알려 주세요.'
fi

echo ''
echo '아무 키나 누르면 창이 닫힙니다.'
read -k 1
