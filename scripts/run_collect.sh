#!/usr/bin/env bash
# 매일 아침 전 거래일 원천 데이터 수집 + 위험도 점수 계산(cron에서 실행).
# 마지막 수집일 다음 날 ~ 어제의 거래일을 받으므로, 여러 번 실행하거나 며칠 빠져도 안전하다.
set -euo pipefail
cd /mnt/c/projects/Claude_eda_agent
mkdir -p logs
exec 9>/tmp/stock_risk_collect.lock
flock -n 9 || { echo "$(date '+%F %T') 이미 실행 중이라 건너뜀" >> "logs/collect_$(date +%Y%m).log"; exit 0; }
/home/sd2-01/miniforge3/envs/stock_risk/bin/python -W ignore -m src.collect --auto >> "logs/collect_$(date +%Y%m).log" 2>&1
