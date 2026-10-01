export const seasons=['summer','spring_autumn','winter'] as const;
export const bands=['off','mid','peak'] as const;
export type Season=typeof seasons[number];
export type Band=typeof bands[number];
export type Rates=Record<Season,Record<Band,number>>;
export type RateDraft=Record<Season,Record<Band,string>>;
export const seasonNames:Record<Season,string>={summer:'여름철',spring_autumn:'봄·가을철',winter:'겨울철'};
const bandNames:Record<Band,string>={off:'경부하',mid:'중간부하',peak:'최대부하'};
export const defaultRates:Rates={summer:{off:56.6,mid:109.5,peak:191.6},spring_autumn:{off:56.6,mid:79.1,peak:109.8},winter:{off:63.6,mid:109.7,peak:167.2}};
export function rateDraft(rates:Rates):RateDraft{
 return Object.fromEntries(seasons.map(season=>[season,Object.fromEntries(bands.map(band=>[band,String(rates[season][band])]))])) as RateDraft;
}
export function validRates(value:unknown):value is Rates{
 if(typeof value!=='object'||value===null)return false;
 const rows=value as Record<string,unknown>;
 return seasons.every(season=>typeof rows[season]==='object'&&rows[season]!==null&&bands.every(band=>{const v=(rows[season] as Record<string,unknown>)[band];return typeof v==='number'&&Number.isFinite(v)&&v>=0&&v<=100000;}));
}
export function TariffTable({draft,baseRate,onChange,onBaseChange,onSubmit,busy}:{draft:RateDraft;baseRate:string;onChange:(season:Season,band:Band,value:string)=>void;onBaseChange:(value:string)=>void;onSubmit:(event:React.FormEvent<HTMLFormElement>)=>void;busy:boolean}){
 return <form className="panel tariff-editor" onSubmit={onSubmit}>
  <div className="panel-heading"><h2>전력 요금표</h2><span className="tariff-meta">2021년 산업용(을) · 고압A 선택Ⅰ<small>원/kW</small></span></div>
  <div className="tariff-table-scroll"><table className="tariff-table"><caption>기본 요금과 계절·부하별 전력량 단가 입력</caption><thead><tr><th scope="col">구분</th><th scope="col">여름철<small>6~8월</small></th><th scope="col">봄·가을철<small>3~5월, 9~10월</small></th><th scope="col">겨울철<small>11~2월</small></th></tr></thead><tbody>
   <tr><th scope="row"><label htmlFor="tariff-base">기본 요금</label></th><td colSpan={3}><div className="tariff-base"><input id="tariff-base" type="number" min="0" max="1000000" step="0.01" required value={baseRate} onChange={e=>onBaseChange(e.target.value)}/></div></td></tr>
   {bands.map(band=><tr key={band}><th scope="row">{bandNames[band]}</th>{seasons.map(season=><td key={season}><input type="number" min="0" max="100000" step="0.01" required aria-label={`${seasonNames[season]} ${bandNames[band]} 요금`} value={draft[season][band]} onChange={e=>onChange(season,band,e.target.value)}/></td>)}</tr>)}
  </tbody></table></div>
  <div className="tariff-footer"><button className="primary" type="submit" disabled={busy}>{busy?'계산 중':'적용'}</button></div>
 </form>;
}
