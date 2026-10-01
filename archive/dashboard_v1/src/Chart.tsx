import type {Point} from './types';
import {useEffect,useRef,useState} from 'react';
const clock=(s:string)=>s.slice(11,16);
export function PowerChart({points,threshold}:{points:Point[];threshold:number}) {
  const [hover,setHover]=useState<number|null>(null);
  const container=useRef<HTMLDivElement>(null);
  const [width,setWidth]=useState(800);
  useEffect(()=>{if(!container.current)return;const observer=new ResizeObserver(([entry])=>setWidth(entry.contentRect.width));observer.observe(container.current);return()=>observer.disconnect();},[]);
  const w=Math.max(280,width),h=260,left=40,right=18,top=28,bottom=36;
  const ticks=w<450?[0,.5,1]:[0,.25,.5,.75,1];
  const max=Math.max(threshold*1.2,...points.flatMap(p=>[p.actual||0,p.predicted||0]),100);
  const start=new Date(points[0].time).getTime();
  const end=Math.max(new Date(points[points.length-1].time).getTime(),start+3600000);
  const x=(t:string)=>left+(new Date(t).getTime()-start)/(end-start)*(w-left-right);
  const y=(v:number)=>h-bottom-v/max*(h-top-bottom);
  const path=(key:'actual'|'predicted')=>{let pen=false;let last=0;return points.map(p=>{if(p[key]===null){pen=false;return '';}const ms=new Date(p.time).getTime();const cmd=pen&&ms-last<=900000?'L':'M';pen=true;last=ms;return `${cmd}${x(p.time)},${y(p[key]!)}`;}).join(' ')};
  const last=points[points.length-1];
  const selected=hover===null?null:points[hover];
  return <div ref={container} className="chart-wrap"><svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label="현재 시점까지 실제 전력과 각 시점의 다음 15분 예측. 마지막 점은 아직 관측되지 않은 예측입니다.">
    {[0,.25,.5,.75,1].map(f=><g key={f}><line x1={left} y1={y(max*f)} x2={w-right} y2={y(max*f)} stroke="#edf0f3"/><text x={left-12} y={y(max*f)+4} textAnchor="end" className="axis">{Math.round(max*f)}</text></g>)}
    <line x1={left} y1={y(threshold)} x2={w-right} y2={y(threshold)} stroke="#d97706" strokeDasharray="5 5"/><text x={w-right} y={y(threshold)-8} textAnchor="end" fill="#b45f05" fontSize="12">알림 기준 {threshold}</text>
    <path d={path('predicted')} fill="none" stroke="#8ab8fa" strokeWidth="2.5" strokeDasharray="5 4"/>
    <path d={path('actual')} fill="none" stroke="#3182f6" strokeWidth="3" strokeLinejoin="round"/>
    <circle cx={x(last.time)} cy={y(last.predicted!)} r="6" fill="#3182f6" stroke="white" strokeWidth="3"/>
    {ticks.map(f=><text key={f} x={left+f*(w-left-right)} y={h-8} textAnchor="middle" className="axis">{clock(new Date(start+f*(end-start)).toLocaleString('sv-SE').replace(' ','T'))}</text>)}
    {selected&&<><line x1={x(selected.time)} x2={x(selected.time)} y1={top} y2={h-bottom} stroke="#aeb8c4" strokeDasharray="3 4"/><circle cx={x(selected.time)} cy={y(selected.actual??selected.predicted??0)} r="4" fill="#191f28"/></>}
    {points.map((p,i)=><rect key={p.time} x={x(p.time)-8} y={top} width="16" height={h-top-bottom} fill="transparent" onMouseEnter={()=>setHover(i)} onMouseLeave={()=>setHover(null)}/>)}
  </svg>{selected&&<div className="chart-tooltip">{clock(selected.time)}　실제 {selected.actual??'—'} / 예측 {selected.predicted??'—'}</div>}</div>
}
