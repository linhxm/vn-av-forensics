"""Small built-in demo so a Node build is optional."""

PAGE = """<!doctype html><html lang="vi"><meta charset="utf-8"><title>Phân tích môi–tiếng</title>
<style>body{max-width:1000px;margin:32px auto;padding:0 20px;font:17px system-ui;background:#101b28;color:#edf4ff}
button,input{padding:12px;margin:8px 0}video{width:100%;max-height:55vh}a{color:#77d8ff}
pre{white-space:pre-wrap}li{margin:10px 0}canvas{width:100%;background:#fff;border-radius:8px}
.hint{color:#b5c6dc;line-height:1.6}</style><h1>Phân tích bất nhất môi–tiếng</h1>
<p class="hint">Chọn clip một người, có audio và thấy rõ miệng, tối đa 60 giây.
Điểm thể hiện bất nhất, chưa phải xác suất giả mạo. Nhánh chưa được huấn luyện sẽ không có kết luận.</p>
<p id="health">Đang kiểm tra model…</p><input id="file" type="file" accept="video/*">
<button id="submit" onclick="analyze()">Phân tích</button><p id="status"></p>
<video id="video" controls></video><h2 id="score"></h2><canvas id="chart" width="1000" height="180"></canvas>
<p id="coverage"></p><ul id="intervals"></ul><p id="downloads"></p><pre id="heads"></pre>
<script>const $=id=>document.getElementById(id);let report=null;
fetch('/api/health').then(r=>r.json()).then(h=>{$('health').textContent=h.ready?'Model sẵn sàng':'Thiếu: '+h.missing.join(', ')});
async function analyze(){let f=$('file').files[0];if(!f)return;$('submit').disabled=true;
try{let data=new FormData();data.append('video',f);let r=await fetch('/api/jobs',{method:'POST',body:data});
if(!r.ok)throw Error(await r.text());let job=await r.json();$('video').src='/api/jobs/'+job.id+'/media';
while(true){r=await fetch('/api/jobs/'+job.id);if(!r.ok)throw Error(await r.text());job=await r.json();
$('status').textContent=job.message;if(job.status==='failed')throw Error(job.error);
if(job.status==='complete'){show(job);break}await new Promise(resolve=>setTimeout(resolve,1000))}
}catch(e){$('status').textContent=e.message}finally{$('submit').disabled=false}}
function show(job){report=job.result;$('score').textContent='Điểm clip: '+(report.clip_inconsistency_score?.toFixed(3)??'Không đủ bằng chứng');
$('coverage').textContent='Độ phủ: '+JSON.stringify(report.coverage);
$('heads').textContent='Trạng thái các nhánh: '+JSON.stringify(report.head_availability,null,2)+'\\nDanh tính: '+report.identity_status;
$('intervals').replaceChildren();for(let span of report.suspicious_intervals){let li=document.createElement('li'),b=document.createElement('button');
b.textContent=`${span.start.toFixed(2)}–${span.end.toFixed(2)} s · ${span.relation}`;
b.onclick=()=>{$('video').currentTime=span.start;$('video').play()};li.append(b);$('intervals').append(li)}
if(!report.suspicious_intervals.length)$('intervals').textContent='Không có khoảng vượt ngưỡng trong vùng đã đánh giá.';
$('downloads').replaceChildren();for(let [name,path] of [['Tải JSON','report'],['Tải CSV','timeline.csv']]){
let a=document.createElement('a');a.href='/api/jobs/'+job.id+'/'+path;a.textContent=name+' ';$('downloads').append(a)}
let c=$('chart').getContext('2d');c.clearRect(0,0,1000,180);c.strokeStyle='#3156d3';c.lineWidth=2;c.beginPath();let active=false;
for(let w of report.window_scores){if(w.inconsistency_score===null){active=false;continue}
let x=w.start/report.duration_s*1000,y=165-w.inconsistency_score*150;
if(active)c.lineTo(x,y);else c.moveTo(x,y);active=true}c.stroke()}
$('chart').onclick=e=>{if(report){let b=$('chart').getBoundingClientRect();$('video').currentTime=(e.clientX-b.left)/b.width*report.duration_s}};
</script></html>"""
