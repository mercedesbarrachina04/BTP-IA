'use strict';
const http = require('http');
const fs = require('fs');
const path = require('path');
const PORT = Number(process.env.PORT || 3000);
const MAX_BODY = 1024 * 1024;
const categories = ['Software','Services','Materials','Logistics'];
let model = null;
let trainCount = 0;
let evaluation = null;
const seed = [
 ['Microsoft 365 license renewal','Software'],['Adobe Creative Cloud licenses','Software'],
 ['ERP software annual subscription','Software'],['Cloud database platform licenses','Software'],
 ['Security antivirus software license','Software'],['Software maintenance renewal','Software'],
 ['SAP finance consulting project','Services'],['External audit professional services','Services'],
 ['Technical helpdesk support services','Services'],['Legal consulting fees','Services'],
 ['Training workshops for employees','Services'],['Business process advisory services','Services'],
 ['Printer paper and office stationery','Materials'],['Computer keyboard and mouse','Materials'],
 ['Packing boxes and packaging materials','Materials'],['Industrial spare parts purchase','Materials'],
 ['Office desks and chairs','Materials'],['Cleaning supplies purchase','Materials'],
 ['Freight transport of goods','Logistics'],['Express parcel delivery','Logistics'],
 ['Warehouse distribution shipment','Logistics'],['International cargo freight','Logistics'],
 ['Last mile transport delivery','Logistics'],['Shipping and customs handling','Logistics']
];
function tokenize(s) {return (String(s||'').toLowerCase().normalize('NFKD').replace(/[\u0300-\u036f]/g,'').match(/[a-z0-9]+/g)||[]).filter(x=>x.length>1);}
function fit(rows) {
  if (!Array.isArray(rows) || rows.length<8) throw Error('Supply at least 8 labeled invoices.');
  const byClass = Object.fromEntries(categories.map(c=>[c,{n:0,total:0,counts:{}}]));
  const vocabulary = new Set();
  for (const r of rows) {
    if (!r.description || !categories.includes(r.category)) throw Error('Each invoice needs a description and a valid category.');
    const c=byClass[r.category];c.n++;
    for(const t of tokenize(r.description)){c.counts[t]=(c.counts[t]||0)+1;c.total++;vocabulary.add(t);}
  }
  for(const c of categories) if(byClass[c].n<2) throw Error('At least 2 training invoices required for each category.');
  return {byClass,vocabSize:vocabulary.size,n:rows.length};
}
function predict(m,description) {
  const words=tokenize(description),scores={};
  for(const label of categories){const c=m.byClass[label];let score=Math.log(c.n/m.n);
    for(const w of words) score+=Math.log(((c.counts[w]||0)+1)/(c.total+m.vocabSize));
    scores[label]=score;
  }
  const peak=Math.max(...Object.values(scores));const exp=Object.fromEntries(categories.map(c=>[c,Math.exp(scores[c]-peak)]));
  const sum=Object.values(exp).reduce((a,b)=>a+b,0);const probabilities=Object.fromEntries(categories.map(c=>[c,exp[c]/sum]));
  const category=categories.reduce((a,b)=>probabilities[a]>=probabilities[b]?a:b);
  return {category,score:Number(probabilities[category].toFixed(3)),review:probabilities[category]<0.65,probabilities};
}
function csvRows(input) {
  const result=[];let row=[],cell='',quoted=false;
  const s=String(input).replace(/^\uFEFF/,'');
  for(let i=0;i<s.length;i++) {const ch=s[i];
    if(quoted){if(ch==='"'&&s[i+1]==='"'){cell+='"';i++;}else if(ch==='"')quoted=false;else cell+=ch;}
    else if(ch==='"')quoted=true;
    else if(ch===','){row.push(cell);cell='';}
    else if(ch==='\n'){row.push(cell.replace(/\r$/,''));result.push(row);row=[];cell='';}
    else cell+=ch;
  }
  if(quoted)throw Error('Malformed CSV: missing closing quote.');
  if(cell.length||row.length){row.push(cell.replace(/\r$/,''));result.push(row);}
  const head=result.shift()?.map(s=>s.trim().toLowerCase())||[];
  const di=head.indexOf('description'),ci=head.indexOf('category');
  if(di<0||ci<0)throw Error('CSV headers must be description,category');
  return result.filter(r=>r.some(c=>c.trim())).map(r=>({description:(r[di]||'').trim(),category:(r[ci]||'').trim()}));
}
function json(res,status,data){res.writeHead(status,{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-store','X-Content-Type-Options':'nosniff'});res.end(JSON.stringify(data));}
function readJson(req){return new Promise((resolve,reject)=>{let s='',n=0;req.on('data',d=>{n+=d.length;if(n>MAX_BODY){reject(Error('Maximum request size is 1 MB'));req.destroy();return;}s+=d;});req.on('end',()=>{try{resolve(JSON.parse(s||'{}'));}catch(e){reject(Error('Invalid JSON'));}});req.on('error',reject);});}
const app=fs.readFileSync(path.join(__dirname,'public','index.html'));
http.createServer(async(req,res)=>{
  const url=new URL(req.url,'http://localhost');
  if(req.method==='GET'&&url.pathname==='/'){res.writeHead(200,{'Content-Type':'text/html; charset=utf-8','X-Content-Type-Options':'nosniff','Cache-Control':'no-store'});return res.end(app);}
  if(req.method==='GET'&&url.pathname==='/health'){return json(res,200,{status:'ok',trained:!!model});}
  if(req.method!=='POST'||!['/api/train','/api/classify','/api/evaluate'].includes(url.pathname))return json(res,404,{error:'Not found'});
  try {const data=await readJson(req);
    if(url.pathname==='/api/train'){
      let rows;if(data.useDemo)rows=seed.map(([description,category])=>({description,category}));else rows=csvRows(data.csv||'');
      if(rows.length>1000)throw Error('Maximum 1,000 records per dataset.');
      // Deterministic holdout for demonstration. Never train on evaluation rows.
      const holdout=[],training=[];
      if(data.useDemo){for(let i=0;i<rows.length;i++)(i%6===0?holdout:training).push(rows[i]);}
      else training.push(...rows);
      const trained=fit(training);model=trained;trainCount=training.length;
      evaluation=holdout.length?{count:holdout.length,correct:holdout.filter(r=>predict(trained,r.description).category===r.category).length,rows:holdout.map(r=>({description:r.description,expected:r.category,predicted:predict(trained,r.description).category}))}:null;
      return json(res,200,{trained:trainCount,categories,evaluation,mode:data.useDemo?'demo-holdout':'user-training'});
    }
    if(!model)return json(res,409,{error:'Train a model first.'});
    if(url.pathname==='/api/classify'){
      const description=String(data.description||'').trim();if(!description||description.length>500)return json(res,400,{error:'Enter an invoice description (up to 500 characters).'});
      return json(res,200,{description,...predict(model,description)});
    }
    if(url.pathname==='/api/evaluate'){
      const rows=csvRows(data.csv||'');if(rows.length>1000)throw Error('Maximum 1,000 test records.');
      if(!rows.length)throw Error('No test rows found.');
      const results=rows.map(r=>({description:r.description,expected:r.category,...predict(model,r.description)}));
      return json(res,200,{total:rows.length,correct:results.filter(r=>r.expected===r.category).length,results});
    }
  }catch(err){return json(res,400,{error:err.message});}
}).listen(PORT,'0.0.0.0',()=>console.log('Invoice ML demo listening on '+PORT));
