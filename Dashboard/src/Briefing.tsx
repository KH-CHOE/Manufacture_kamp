import {useEffect,useRef,useState} from 'react';
import {Bot,Eye,EyeOff,KeyRound,RefreshCw,Send,Sparkles} from 'lucide-react';
import type {Briefing as BriefingData,ChatReply} from './types';

const KEY_STORE='kamp-openai-key';
const AUTO_STORE='kamp-briefing-auto';
const readKey=()=>{try{return localStorage.getItem(KEY_STORE)||'';}catch{return '';}};

// 서버가 쓰는 표기 그대로 — **굵게** 와 줄바꿈만 처리한다
function Rich({text}:{text:string}){
 return <>{text.split('\n').filter(l=>l.trim()).map((line,i)=><p key={i}>{line.split(/(\*\*[^*]+\*\*)/g).map((part,j)=>part.startsWith('**')&&part.endsWith('**')?<strong key={j}>{part.slice(2,-2)}</strong>:<span key={j}>{part}</span>)}</p>)}</>;
}

async function post<T>(path:string,body:unknown,key:string):Promise<T>{
 const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json',...(key?{'X-OpenAI-Key':key}:{})},body:JSON.stringify(body)});
 if(!r.ok){const d=await r.json().catch(()=>null);throw new Error(typeof d?.detail==='string'?d.detail:'브리핑을 불러오지 못했습니다.');}
 return r.json();
}

/** 재생 시점의 전력 브리핑과 질의. 키는 이 브라우저에만 저장하고 요청 헤더로만 보낸다. */
export function BriefingPanel({day,cursor,threshold,ready}:{day:string;cursor:number;threshold:number|null;ready:boolean}){
 const [key,setKey]=useState(readKey),[draft,setDraft]=useState(''),[show,setShow]=useState(false);
 const [auto,setAuto]=useState(()=>{try{return localStorage.getItem(AUTO_STORE)!=='off';}catch{return true;}});
 const [brief,setBrief]=useState<BriefingData|null>(null),[loading,setLoading]=useState(false),[error,setError]=useState('');
 const [chat,setChat]=useState<{role:'user'|'assistant';content:string}[]>([]),[question,setQuestion]=useState(''),[asking,setAsking]=useState(false);
 const inflight=useRef(false),wanted=useRef<string>(''),done=useRef<string>('');
 const target=`${day}|${cursor}|${threshold??''}|${key?'k':''}`;

 const load=async(force=false)=>{
  if(!day||(!force&&done.current===target))return;
  if(inflight.current){wanted.current=target;return;}       // 겹쳐 보내지 않는다 — 끝나면 최신 칸으로 한 번 더
  inflight.current=true;setLoading(true);setError('');
  const ask=target;
  try{const b=await post<BriefingData>('/api/briefing',{day,cursor,threshold},key);setBrief(b);done.current=ask;}
  catch(e){setError((e as Error).message);}
  finally{inflight.current=false;setLoading(false);if(wanted.current&&wanted.current!==ask){wanted.current='';setTimeout(()=>latest.current(),0);}}
 };
 const latest=useRef(load);latest.current=load;   // 다시 부를 때는 최신 칸의 load 를 쓴다
 useEffect(()=>{if(auto&&ready)load();},[target,auto,ready]);
 useEffect(()=>{setChat([]);},[day]);

 const saveKey=()=>{const k=draft.trim();if(!k)return;try{localStorage.setItem(KEY_STORE,k);}catch{}setKey(k);setDraft('');done.current='';};
 const clearKey=()=>{try{localStorage.removeItem(KEY_STORE);}catch{}setKey('');done.current='';};
 const toggleAuto=()=>{const n=!auto;setAuto(n);try{localStorage.setItem(AUTO_STORE,n?'on':'off');}catch{}};
 const send=async()=>{
  const q=question.trim();if(!q||asking)return;
  const next=[...chat,{role:'user' as const,content:q}];setChat(next);setQuestion('');setAsking(true);
  try{const r=await post<ChatReply>('/api/chat',{day,cursor,threshold,messages:next},key);setChat([...next,{role:'assistant',content:r.text}]);}
  catch(e){setChat([...next,{role:'assistant',content:`⚠ ${(e as Error).message}`}]);}
  finally{setAsking(false);}
 };

 return <section className="panel briefing-panel" aria-label="전력 브리핑">
  <div className="panel-heading"><div><h2><Sparkles size={18}/> {brief?`${brief.label} 브리핑`:'전력 브리핑'}</h2>
   <p>{brief?`재생 자료 기준 ${brief.confirmedAt}까지 확정값 · ${brief.source==='llm'?`AI 브리핑 (${brief.model})`:'숫자 요약 (API 키 없음)'}`:'재생 시점의 예측과 관리 권고를 15분마다 정리해요.'}</p></div>
   <div className="briefing-actions"><label className="auto-toggle"><input type="checkbox" checked={auto} onChange={toggleAuto}/>자동 갱신</label>
   <button className="secondary" onClick={()=>load(true)} disabled={loading||!day}><RefreshCw size={16} className={loading?'spin':''}/>브리핑 갱신</button></div></div>
  {error&&<div className="briefing-error" role="alert">{error}</div>}
  <div className="briefing-body" aria-live="polite">{brief?<Rich text={brief.text}/>:<p className="muted">{loading?'브리핑을 준비하고 있어요…':'브리핑 갱신을 눌러 시작하세요.'}</p>}</div>
  <div className="key-row">{key?<><KeyRound size={15}/><span>API 키 저장됨 · {key.slice(0,3)}…{key.slice(-4)}</span><button className="text-button" onClick={clearKey}>지우기</button></>:
   <form onSubmit={e=>{e.preventDefault();saveKey();}}><KeyRound size={15}/><input type={show?'text':'password'} placeholder="OpenAI API 키 (sk-…)" value={draft} onChange={e=>setDraft(e.target.value)} autoComplete="off" aria-label="OpenAI API 키"/><button type="button" className="icon-button" onClick={()=>setShow(!show)} aria-label={show?'키 가리기':'키 보기'}>{show?<EyeOff size={16}/>:<Eye size={16}/>}</button><button className="secondary">저장</button></form>}
   <small>키는 이 브라우저에만 저장되고 서버는 요청마다 받아 쓰고 보관하지 않아요.</small></div>
  <div className="chat-box">
   {chat.length>0&&<div className="chat-log">{chat.map((m,i)=><div key={i} className={`chat-msg ${m.role}`}>{m.role==='assistant'&&<Bot size={16}/>}<div><Rich text={m.content}/></div></div>)}{asking&&<div className="chat-msg assistant"><Bot size={16}/><div><p className="muted">답을 찾고 있어요…</p></div></div>}</div>}
   <form className="chat-input" onSubmit={e=>{e.preventDefault();send();}}><input placeholder={key?'예: 지금 압축기 시동을 미루면 요금이 얼마나 달라져?':'질문하려면 API 키를 먼저 저장하세요'} value={question} onChange={e=>setQuestion(e.target.value)} disabled={!key||asking} aria-label="전력 관리 질문"/><button className="primary" disabled={!key||asking||!question.trim()} aria-label="질문 보내기"><Send size={16}/></button></form>
  </div>
 </section>;
}
