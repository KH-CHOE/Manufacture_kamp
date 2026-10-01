export interface Meta {days:string[];defaultDay:string;model:string;algorithm:string;modelType:string;ensemble?:boolean;blendWeight?:number;seeds?:number[]|null;version:string;features:string[];horizon:number;rows:number;mse:number;mae:number;unit:string;source:string}
export interface Point {time:string;actual:number|null;predicted:number|null}
export interface Alert {time:string;targetTime:string;predicted:number;key:string}
export interface Recommendation {value:number;annualPeak:number;annualPeakTime:string;year:number;coverageStart:string;asOf:string;errorMargin:number;errorSamples:number;basis:string}
export interface Snapshot {day:string;cursor:number;count:number;time:string;targetTime:string;current:number;prediction:number;threshold:number;margin:number;atRisk:boolean;dailyPeak:number;recommendation:Recommendation;points:Point[];alerts:Alert[];timeline:string[];context:Record<string,number>}
export interface Cost {energy:number;labor:number;total:number}
export interface Result {prediction:number;baselinePrediction:number;baseline:Cost;scenario:Cost;productionRange:{min:number;max:number};assumption:string}
