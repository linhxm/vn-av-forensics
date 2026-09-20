import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { createJob, getHealth, getJob, type Job } from "./api";
import { RelationResultView } from "./RelationResultView";
import "./pipeline.css";

function App() {
  const [job,setJob]=useState<Job|null>(null);
  const [file,setFile]=useState<File|null>(null);
  const [error,setError]=useState("");
  const [uploading,setUploading]=useState(false);
  const [health,setHealth]=useState<Awaited<ReturnType<typeof getHealth>>|null>(null);
  const busy=uploading||job?.status==="queued"||job?.status==="processing";
  useEffect(()=>{getHealth().then(setHealth).catch(e=>setError(String(e)));},[]);
  useEffect(()=>{
    if(!job||!["queued","processing"].includes(job.status))return;
    let alive=true;
    const timer=setTimeout(()=>getJob(job.id).then(next=>{if(alive)setJob(next);}).catch(e=>{if(alive){setError(String(e));setJob({...job,status:"interrupted"});}}),1200);
    return ()=>{alive=false;clearTimeout(timer);};
  },[job]);
  async function submit(){if(!file)return;setUploading(true);setError("");setJob(null);try{setJob(await createJob(file));}catch(e){setError(String(e));}finally{setUploading(false);}}
  const result=job?.result??null;
  return <main>
    <h1>Phân tích quan hệ âm thanh–hình ảnh</h1>
    <p>Video một người, thấy rõ miệng, tối đa 60 giây. Kết quả từng nhánh và chất lượng checkpoint được hiển thị riêng.</p>
    <section><h2>1. Đầu vào</h2>
      {health&&!health.ready&&<p role="alert">Backend chưa sẵn sàng: {health.missing.join(", ")}</p>}
      <input aria-label="Video cần phân tích" type="file" accept=".mp4,.webm,.mov,.mkv,.avi,.mpg" disabled={busy} onChange={e=>setFile(e.target.files?.[0]??null)}/>
      <button disabled={!file||busy||health?.ready===false} onClick={submit}>{uploading?"Đang tải…":busy?"Đang xử lý…":"Phân tích"}</button>
      {error&&<p role="alert" className="warning">{error}</p>}
    </section>
    {job&&<section aria-live="polite"><h2>2. Tiến trình thực tế</h2><p>{job.message}</p>
      <ol>{(job.steps??[]).map((step,i)=><li key={i}>{step.message} <small>{new Date(step.at).toLocaleTimeString()}</small></li>)}</ol>
      {job.error&&<p className="warning">{job.error}</p>}
    </section>}
    {job&&result&&<RelationResultView key={job.id} result={result} jobId={job.id} filename={job.filename}/>}
  </main>;
}
createRoot(document.getElementById("root")!).render(<App/>);
