"""Export a single one-hour recommendation using local dashboard data.
Run: python3 -m optimization.export_plan --day 2021-07-12 --cursor 48
"""
import argparse
import asyncio
import json
from pathlib import Path
import pandas as pd
from backend.app import app, lifespan, optimize_staffing, StaffingSpec

async def export(day,cursor):
    async with lifespan(app):
        result=optimize_staffing(StaffingSpec(day=day,cursor=cursor))
        out=Path(__file__).parent/'output';out.mkdir(exist_ok=True)
        name=f'hourly-{day}-{result["forecastTime"][11:16].replace(":","")}'
        (out/f'{name}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        flat={k:v for k,v in result.items() if not isinstance(v,dict)}
        flat.update({f'{section}_{k}':v for section in ['optimized','fixed','economics'] for k,v in result[section].items()})
        pd.DataFrame([flat]).to_csv(out/f'{name}.csv',index=False,encoding='utf-8-sig')
        print(f'Exported {name} · {result["status"]}')

if __name__=='__main__':
    args=argparse.ArgumentParser();args.add_argument('--day',default='2021-07-12');args.add_argument('--cursor',default=48,type=int)
    spec=args.parse_args();asyncio.run(export(spec.day,spec.cursor))
