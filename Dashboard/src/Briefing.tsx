import {useEffect,useRef,useState} from 'react';
import {Bot,Eye,EyeOff,KeyRound,RefreshCw,Send,Volume2,VolumeX} from 'lucide-react';
import type {Briefing as BriefingData,ChatReply} from './types';

const KEY_STORE='kamp-openai-key';
const AUTO_STORE='kamp-briefing-auto';
const readKey=()=>{try{return localStorage.getItem(KEY_STORE)||'';}catch{return '';}};
const VOICE_STORE='kamp-briefing-voice';
const canSpeak=typeof window!=='undefined'&&'speechSynthesis' in window;

// 브라우저 음성용 텍스트로 변환한다.
const hm=(h:string,m:string)=>`${Number(h)}시${m==='00'?'':` ${Number(m)}분`}`;
export function toSpeech(text:string){
 return text.replace(/\*\*/g,'')
  .replace(/(\d{1,2}):(\d{2})\s*~\s*(\d{1,2}):(\d{2})/g,(_,a,b,c,d)=>`${hm(a,b)}부터 ${hm(c,d)}까지`)
  .replace(/(\d{1,2}):(\d{2})/g,(_,a,b)=>hm(a,b))
  .replace(/\bkW\b/g,'킬로와트').replace(/\+(\d)/g,'플러스 $1').replace(/(^|[^\d])-(\d)/g,'$1마이너스 $2')
  .replace(/\s*[—·]\s*/g,', ').replace(/\s+/g,' ').trim();
}
function speak(text:string,onEnd:()=>void){
 const synth=window.speechSynthesis;synth.cancel();
 const u=new SpeechSynthesisUtterance(toSpeech(text));
 u.lang='ko-KR';u.rate=1.05;
 const voice=synth.getVoices().find(v=>v.lang.toLowerCase().startsWith('ko'));if(voice)u.voice=voice;
 u.onend=onEnd;u.onerror=onEnd;synth.speak(u);
}

// 문단과 굵은 글씨를 렌더링한다.
function Rich({text}:{text:string}){
 return <>{text.split('\n').filter(l=>l.trim()).map((line,i)=><p key={i}>{line.split(/(\*\*[^*]+\*\*)/g).map((part,j)=>part.startsWith('**')&&part.endsWith('**')?<strong key={j}>{part.slice(2,-2)}</strong>:<span key={j}>{part}</span>)}</p>)}</>;
}

async function post<T>(path:string,body:unknown,key:string):Promise<T>{
 const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json',...(key?{'X-OpenAI-Key':key}:{})},body:JSON.stringify(body)});
 if(!r.ok){const d=await r.json().catch(()=>null);throw new Error(typeof d?.detail==='string'?d.detail:'브리핑을 불러오지 못했습니다.');}
 return r.json();
}

/** 재생 시점의 AI 요약과 질의응답. 키는 요청 헤더로 전달한다. */
export function BriefingPanel({day,cursor,threshold,ready}:{day:string;cursor:number;threshold:number|null;ready:boolean}){
 const [key,setKey]=useState(readKey),[draft,setDraft]=useState(''),[show,setShow]=useState(false);
 const [auto,setAuto]=useState(()=>{try{return localStorage.getItem(AUTO_STORE)!=='off';}catch{return true;}});
 const [brief,setBrief]=useState<BriefingData|null>(null),[loading,setLoading]=useState(false),[error,setError]=useState('');
 const [chat,setChat]=useState<{role:'user'|'assistant';content:string}[]>([]),[question,setQuestion]=useState(''),[asking,setAsking]=useState(false);
 const inflight=useRef(false),wanted=useRef<string>(''),done=useRef<string>('');
 const [speaking,setSpeaking]=useState(false),[voiceNote,setVoiceNote]=useState('');
 const audioRef=useRef<HTMLAudioElement|null>(null),audioCache=useRef(new Map<string,string>());
 const [voiceAuto,setVoiceAuto]=useState(()=>{try{return localStorage.getItem(VOICE_STORE)==='on';}catch{return false;}});
 const spoken=useRef('');
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
 const stopVoice=()=>{audioRef.current?.pause();audioRef.current=null;if(canSpeak)window.speechSynthesis.cancel();setSpeaking(false);};
 // 키가 있으면 OpenAI 음성(서버 /api/speech), 없거나 실패하면 브라우저 음성. 같은 문장은 만든 음성을 다시 쓴다
 const play=async(text:string)=>{
  stopVoice();setSpeaking(true);setVoiceNote('');
  const said=toSpeech(text);
  if(key){try{
   let url=audioCache.current.get(said);
   if(!url){const r=await fetch('/api/speech',{method:'POST',headers:{'Content-Type':'application/json','X-OpenAI-Key':key},body:JSON.stringify({text:said})});
    if(!r.ok){const d=await r.json().catch(()=>null);throw new Error(typeof d?.detail==='string'?d.detail:'AI 음성을 만들지 못했어요.');}
    url=URL.createObjectURL(await r.blob());audioCache.current.set(said,url);}
   const a=new Audio(url);audioRef.current=a;a.onended=()=>{if(audioRef.current===a)setSpeaking(false);};a.onerror=()=>setSpeaking(false);
   await a.play();return;
  }catch(e){setVoiceNote(`${(e as Error).message} 브라우저 음성으로 읽어요.`);}}
  if(canSpeak)speak(text,()=>setSpeaking(false));else setSpeaking(false);
 };
 const playRef=useRef(play);playRef.current=play;
 // 피크 위험 브리핑만 자동으로 읽는다(켜 둔 경우). 같은 브리핑은 한 번만
 useEffect(()=>{if(!voiceAuto||!brief)return;const id=`${brief.label}|${brief.text}`;
  if(brief.facts?.['피크위험']===true&&spoken.current!==id){spoken.current=id;playRef.current(brief.text);}},[brief,voiceAuto]);
 useEffect(()=>()=>{audioRef.current?.pause();if(canSpeak)window.speechSynthesis.cancel();audioCache.current.forEach(u=>URL.revokeObjectURL(u));},[]);
 const readNow=()=>{if(!brief)return;if(speaking){stopVoice();return;}play(brief.text);};
 const toggleVoice=()=>{const n=!voiceAuto;setVoiceAuto(n);try{localStorage.setItem(VOICE_STORE,n?'on':'off');}catch{}if(!n)stopVoice();};
 const voiceReady=canSpeak||!!key;

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

 return <section className="panel briefing-panel" aria-label="AI 챗">
  <div className="panel-heading"><div><h2>AI 챗</h2></div>
   <div className="briefing-actions"><label className="auto-toggle"><input type="checkbox" checked={auto} onChange={toggleAuto}/>자동 갱신</label>
   {voiceReady&&<label className="auto-toggle" title="새 브리핑이 피크 위험일 때만 자동으로 읽어요"><input type="checkbox" checked={voiceAuto} onChange={toggleVoice}/>위험 시 음성</label>}
   {voiceReady&&<button className="secondary" onClick={readNow} disabled={!brief} title={key?'OpenAI 음성으로 읽어요':'브라우저 음성으로 읽어요 (API 키를 넣으면 OpenAI 음성)'} aria-label={speaking?'읽기 멈추기':'브리핑 읽어 주기'}>{speaking?<VolumeX size={16}/>:<Volume2 size={16}/>}{speaking?'멈춤':'읽어 주기'}</button>}
   <button className="secondary" onClick={()=>load(true)} disabled={loading||!day}><RefreshCw size={16} className={loading?'spin':''}/>브리핑 갱신</button></div></div>
  {error&&<div className="briefing-error" role="alert">{error}</div>}
  {voiceNote&&<div className="briefing-error" role="status">{voiceNote}</div>}
  <div className="briefing-body" aria-live="polite">{brief?<Rich text={brief.text}/>:<p className="muted">{loading?'브리핑을 준비하고 있어요…':'브리핑 갱신을 눌러 시작하세요.'}</p>}</div>
  <div className="key-row">{key?<><KeyRound size={15}/><span>API 키 저장됨 · {key.slice(0,3)}…{key.slice(-4)}</span><button className="text-button" onClick={clearKey}>지우기</button></>:
   <form onSubmit={e=>{e.preventDefault();saveKey();}}><KeyRound size={15}/><input type={show?'text':'password'} placeholder="OpenAI API 키 (sk-…)" value={draft} onChange={e=>setDraft(e.target.value)} autoComplete="off" aria-label="OpenAI API 키"/><button type="button" className="icon-button" onClick={()=>setShow(!show)} aria-label={show?'키 가리기':'키 보기'}>{show?<EyeOff size={16}/>:<Eye size={16}/>}</button><button className="secondary">저장</button></form>}
  </div>
  <div className="chat-box">
   {chat.length>0&&<div className="chat-log">{chat.map((m,i)=><div key={i} className={`chat-msg ${m.role}`}>{m.role==='assistant'&&<Bot size={16}/>}<div><Rich text={m.content}/></div></div>)}{asking&&<div className="chat-msg assistant"><Bot size={16}/><div><p className="muted">답을 찾고 있어요…</p></div></div>}</div>}
   <form className="chat-input" onSubmit={e=>{e.preventDefault();send();}}><input placeholder={key?'예: 다음 15분 최대치를 낮추려면 지금 무엇을 미루면 좋을까?':'질문하려면 API 키를 먼저 저장하세요'} value={question} onChange={e=>setQuestion(e.target.value)} disabled={!key||asking} aria-label="전력 관리 질문"/><button className="primary" disabled={!key||asking||!question.trim()} aria-label="질문 보내기"><Send size={16}/></button></form>
  </div>
 </section>;
}
