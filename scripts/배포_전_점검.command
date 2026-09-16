#!/bin/zsh

cd "$(dirname "$0")/.." || exit 1
python3 -m py_compile app.py build_dashboard.py database.py storage.py supabase_store.py scripts/*.py || exit 1
python3 -c 'from app import health; assert health()["model_ready"]; print("배포 전 점검 완료:", health())' || exit 1
echo "이제 GitHub에 올린 뒤 Vercel에서 Import 하세요."
echo "아무 키나 누르면 창이 닫힙니다."
read -k 1
