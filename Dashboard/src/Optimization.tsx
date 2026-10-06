import {useEffect,useState} from 'react';
import {Users,Wallet} from 'lucide-react';
import type {Snapshot} from './types';
import {TariffTable,defaultRates,rateDraft,validRates,seasons,bands,type Season,type Band,type Rates,type RateDraft} from './TariffTable';

type Plan={asOf:string;forecastTime:string;targetTime:string;intervalStart:string;intervalEnd:string;production:number;prediction:number;dayType:string;recommendedStaff:number|null;optimized:{labor:number|null;energy:number;baseAllocated:number;total:number|null};economics:{unitPrice:number;productionProfit:number;remainder:number|null};fixed:{baseRate:number;billingPeak:number;wage:number;dayWage:number;season:Season;bandKey:Band;rate:number}};
type Settings={unitPrice:number;dayWage:number;rates:Rates;baseRate:number;billingPeak:number};
type SaveFlags={unitPrice:boolean;dayWage:boolean;billingPeak:boolean};
const num=(v:number,d=0)=>v.toLocaleString('ko-KR',{maximumFractionDigits:d});
const won=(v:number|null)=>v===null?'산정 불가':`${num(v)}원`;
const clock=(v:string)=>v.slice(11,16);
// 생산 이익과 요금표의 기본 입력값.
const storageKey='kamp-optimization-defaults-v2';
const defaults:Settings={unitPrice:150,dayWage:8720,rates:defaultRates,baseRate:7220,billingPeak:200};
function stored(){try{return JSON.parse(localStorage.getItem(storageKey)||'{}');}catch{return {};}}
function loadSettings():Settings{
 const data=stored();
 const valid=(v:unknown,min:number,max:number)=>typeof v==='number'&&Number.isFinite(v)&&v>=min&&v<=max;
 return {unitPrice:valid(data?.unitPrice,0,1000000)?data.unitPrice:150,dayWage:valid(data?.dayWage,.01,1000000)?data.dayWage:8720,rates:validRates(data?.rates)?data.rates:defaultRates,baseRate:valid(data?.baseRate,0,1000000)?data.baseRate:7220,billingPeak:valid(data?.billingPeak,0,1000000)?data.billingPeak:200};
}
function loadFlags():SaveFlags{
 const flags=stored()?.saveFlags;
 return {unitPrice:typeof flags?.unitPrice==='boolean'?flags.unitPrice:true,dayWage:typeof flags?.dayWage==='boolean'?flags.dayWage:true,billingPeak:typeof flags?.billingPeak==='boolean'?flags.billingPeak:true};
}
export function Optimization({snapshot:s,pending}:{snapshot:Snapshot;pending:boolean}){
 const [initial]=useState(loadSettings);
 const [applied,setApplied]=useState<Settings>(initial);
 const [profit,setProfit]=useState(String(initial.unitPrice));
 const [wage,setWage]=useState(String(initial.dayWage));
 const [rates,setRates]=useState<RateDraft>(()=>rateDraft(initial.rates));
 const [baseRate,setBaseRate]=useState(String(initial.baseRate));
 const [peak,setPeak]=useState(String(initial.billingPeak));
 const [saved,setSaved]=useState(loadFlags);
 const [notice,setNotice]=useState('');
 const [plan,setPlan]=useState<Plan|null>(null),[busy,setBusy]=useState(false),[error,setError]=useState(''),[reload,setReload]=useState(0);
 useEffect(()=>{
  if(pending){setPlan(null);return;}
  const controller=new AbortController();setBusy(true);setError('');
  fetch('/api/optimization/replay',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({day:s.day,cursor:s.cursor,unit_price:applied.unitPrice,day_wage:applied.dayWage,rate_table:applied.rates,base_rate:applied.baseRate,billing_peak:applied.billingPeak}),signal:controller.signal})
   .then(async r=>{const data=await r.json();if(!r.ok)throw new Error(typeof data.detail==='string'?data.detail:'인력 추천을 불러오지 못했습니다.');return data as Plan;})
   .then(setPlan).catch(e=>{if(e.name!=='AbortError'){setError(e.message);setPlan(null);}})
   .finally(()=>{if(!controller.signal.aborted)setBusy(false);});
  return()=>controller.abort();
 },[s.day,s.cursor,pending,reload,applied]);
 const changeRate=(season:Season,band:Band,value:string)=>setRates(r=>({...r,[season]:{...r[season],[band]:value}}));
 const applyInputs=(event:React.FormEvent<HTMLFormElement>)=>{
  event.preventDefault();
  const unitPrice=Number(profit),dayWage=Number(wage),basic=Number(baseRate),billingPeak=Number(peak);
  const nextRates=Object.fromEntries(seasons.map(season=>[season,Object.fromEntries(bands.map(band=>[band,Number(rates[season][band])]))])) as Rates;
  if(!peak.trim()||!Number.isFinite(billingPeak)||billingPeak<0||billingPeak>1000000||!baseRate.trim()||!Number.isFinite(basic)||basic<0||basic>1000000||!profit.trim()||!wage.trim()||!Number.isFinite(unitPrice)||unitPrice<0||unitPrice>1000000||!Number.isFinite(dayWage)||dayWage<=0||dayWage>1000000||!validRates(nextRates)||seasons.some(season=>bands.some(band=>!rates[season][band].trim()))){setNotice('생산 이익·인건비·피크 전력·요금표를 유효한 숫자로 입력해주세요.');return;}
  try{localStorage.setItem(storageKey,JSON.stringify({unitPrice:saved.unitPrice?unitPrice:null,dayWage:saved.dayWage?dayWage:null,rates:nextRates,baseRate:basic,billingPeak:saved.billingPeak?billingPeak:null,saveFlags:saved}));setNotice('');}
  catch{setNotice('계산에 적용했지만 이 브라우저에 기본값을 저장하지 못했어요.');}
  setApplied({unitPrice,dayWage,rates:nextRates,baseRate:basic,billingPeak});
 };
 const resetInputs=()=>{
  setApplied(defaults);setProfit('150');setWage('8720');setRates(rateDraft(defaultRates));setBaseRate('7220');setPeak('200');setSaved({unitPrice:true,dayWage:true,billingPeak:true});
  try{localStorage.removeItem(storageKey);setNotice('');}catch{setNotice('초기값을 적용했지만 저장된 기본값을 삭제하지 못했어요.');}
 };
 const ready=plan&&plan.asOf===s.time&&!pending;
 return <div className="resource-page">
  {error&&<div className="error-banner" role="alert">{error}<button onClick={()=>setReload(r=>r+1)} disabled={busy||pending}>다시 불러오기</button></div>}
  {!ready?<div className="loading" aria-live="polite">{busy||pending?'실적 목표에 맞는 인력을 계산하고 있어요.':'관제 탭에서 다른 시점을 선택해주세요.'}</div>:<>
   <section className="resource-summary planning-summary" aria-label="인력 추천 요약">
    <div><span>실적 목표</span><strong>{num(plan.production,2)}<small>개</small></strong><p>{clock(plan.intervalStart)}~{clock(plan.intervalEnd)} · {plan.dayType}</p></div>
    <div className="recommended-metric"><span>현재 추천 인원</span><strong className="blue-text">{plan.recommendedStaff??'미산정'}<small>{plan.recommendedStaff===null?'':'명'}</small></strong><p>{clock(plan.intervalStart)}~{clock(plan.intervalEnd)} · {plan.dayType}</p></div>
   </section>
   <div className="planning-grid">
    <form className="panel planning-inputs" onSubmit={applyInputs}>
     <div className="panel-heading"><h2>계산 조건</h2><Users size={21}/></div>
     <dl className="planning-factors">
      <div><dt><label htmlFor="hour-power">정시 예측 전력</label></dt><dd className="factor-input"><input id="hour-power" type="text" readOnly value={num(plan.prediction,1)}/><small>kW</small></dd><p>{clock(plan.forecastTime)} → {clock(plan.targetTime)} 예측</p></div>
      <div><dt><label htmlFor="hour-duration">적용 시간</label></dt><dd className="factor-input"><input id="hour-duration" type="text" readOnly value="1"/><small>시간</small></dd><p>{clock(plan.intervalStart)}~{clock(plan.intervalEnd)}</p></div>
      <div><dt><label htmlFor="unit-price">생산 이익</label></dt><dd className="factor-input"><input id="unit-price" type="number" min="0" max="1000000" step="0.01" required value={profit} onChange={e=>setProfit(e.target.value)}/><small>원/개</small></dd><label className="save-default"><input type="checkbox" aria-label="생산 이익 기본값" checked={saved.unitPrice} onChange={e=>setSaved(v=>({...v,unitPrice:e.target.checked}))}/>기본값</label></div>
      <div><dt><label htmlFor="day-wage">인건비</label></dt><dd className="factor-input"><input id="day-wage" type="number" min="0.01" max="1000000" step="0.01" required value={wage} onChange={e=>setWage(e.target.value)}/><small>원/명·시간</small></dd><label className="save-default"><input type="checkbox" aria-label="인건비 기본값" checked={saved.dayWage} onChange={e=>setSaved(v=>({...v,dayWage:e.target.checked}))}/>기본값</label></div>
      <div><dt><label htmlFor="applied-base">기본 요금</label></dt><dd className="factor-input"><input id="applied-base" type="text" readOnly value={num(plan.fixed.baseRate,2)}/><small>원/kW</small></dd></div>
      <div><dt><label htmlFor="energy-rate">전력량 단가</label></dt><dd className="factor-input"><input id="energy-rate" type="text" readOnly value={num(plan.fixed.rate,2)}/><small>원/kWh</small></dd></div>
     </dl>
     <div className="planning-lower"><dl className="planning-factors peak-factor"><div><dt><label htmlFor="billing-peak">작년 피크 전력</label></dt><dd className="factor-input"><input id="billing-peak" type="number" min="0" max="1000000" step="0.01" required value={peak} onChange={e=>setPeak(e.target.value)}/><small>kW</small></dd><label className="save-default"><input type="checkbox" aria-label="작년 피크 전력 기본값" checked={saved.billingPeak} onChange={e=>setSaved(v=>({...v,billingPeak:e.target.checked}))}/>기본값</label></div></dl><div className="factor-actions"><button className="primary" type="submit" disabled={busy||pending}>{busy?'계산 중':'적용'}</button><button className="secondary" type="button" onClick={resetInputs} disabled={busy||pending}>초기화</button></div></div>
     {notice&&<p className="factor-notice" role="alert">{notice}</p>}
     <div className="planning-equation"><span>운영비 계산식</span><p><b>N x 인건비</b> + 전력량요금 + 기본요금 배분</p></div>
    </form>
    <section className={`panel planning-result ${busy?'calculating':''}`} aria-busy={busy}>
     <div className="panel-heading"><h2>계산 결과</h2><Wallet size={21}/></div>
     <dl className="planning-ledger">
      <div><dt>예상 생산 이익</dt><dd>{won(plan.economics.productionProfit)}</dd></div>
      <div className="ledger-income"><dt>예상 이익</dt><dd className="blue-text">{won(plan.economics.productionProfit)}</dd></div>
      <div className="ledger-expenses"><dt>인건비 {plan.recommendedStaff===null?'':`· ${plan.recommendedStaff}명`}</dt><dd>{won(plan.optimized.labor)}</dd></div>
      <div><dt>전력량요금</dt><dd>{won(plan.optimized.energy)}</dd></div>
      <div><dt>기본요금 1시간 배분</dt><dd>{won(plan.optimized.baseAllocated)}</dd></div>
      <div className="ledger-total"><dt>예상 운영비</dt><dd className="danger-text">{won(plan.optimized.total)}</dd></div>
     </dl>
     <div className="planning-remainder"><span>시간당 영업 이익</span><strong className={plan.economics.remainder!==null&&plan.economics.remainder<0?'danger-text':plan.economics.remainder!==null&&plan.economics.remainder>0?'blue-text':''}>{won(plan.economics.remainder)}</strong></div>
    </section>
   </div>
   <TariffTable draft={rates} baseRate={baseRate} onBaseChange={setBaseRate} onChange={changeRate} onSubmit={applyInputs} busy={busy||pending}/>
  </>}
 </div>;
}
