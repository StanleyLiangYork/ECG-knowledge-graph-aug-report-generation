"""Content-addressed checkpoints around the unchanged report evaluator."""
import contextlib
import fcntl
import math
import signal
import threading
import time
from pathlib import Path
from importlib.metadata import version
from unittest.mock import patch
import report_evaluation as evaluation
import report_study
import candidate_preparation
import core
from report_study import atomic_json, read, signature, sha
from report_generation_resume import _StopRequested


def evaluate_resumable(root,config,study,extra_texts=None,cxr_report_root=None,
                       allow_partial=False,max_run_hours=30.0,stop_file=None):
    """Reuse completed metric calls; always repeat provenance/coverage checks.

    A stopped evaluation returns complete=False, not publishable final metrics.
    Input/reference loading and an in-flight metric call restart after a hard kill.
    """
    if not isinstance(max_run_hours,(int,float)) or not math.isfinite(max_run_hours) or max_run_hours<=0:
        raise ValueError('max_run_hours must be positive and finite')
    if threading.current_thread() is not threading.main_thread():raise RuntimeError('Use the notebook main thread')
    study=Path(study);stop_file=Path(stop_file) if stop_file else study/'STOP_evaluation'
    start=time.monotonic();deadline=start+max_run_hours*3600;handlers={};requested={'reason':None}
    from rouge_score import rouge_scorer, scoring
    implementation=dict(sources={Path(m.__file__).name:sha(m.__file__) for m in
                        (evaluation,report_study,candidate_preparation,core)},
                        wrapper=sha(__file__),libraries={n:version(n) for n in ('numpy','rouge-score','nltk')})
    cache=study/'evaluation_checkpoints'/signature(implementation);cache.mkdir(parents=True,exist_ok=True)
    stats={'reused':0,'computed':0};active={'operation':'startup'}
    def receive(sig,frame):requested['reason']=signal.Signals(sig).name
    def guard():
        if requested['reason']:raise _StopRequested(requested['reason'])
        if stop_file.exists():raise _StopRequested('stop_file')
        if time.monotonic()>=deadline:raise _StopRequested('time_budget')
    def memo(name,inputs,compute,decode=lambda v:v):
        active['operation']=name;guard();key=signature(dict(operation=name,inputs=inputs));path=cache/(key+'.json')
        if path.exists():
            saved=read(path)
            if saved['key']!=key or saved['sha256']!=signature(saved['value']):raise ValueError('Corrupt evaluation checkpoint: '+str(path))
            stats['reused']+=1;return decode(saved['value'])
        value=compute();atomic_json(path,dict(key=key,value=value,sha256=signature(value)))
        stats['computed']+=1;return value
    finding=evaluation.score_findings;bootstrap=evaluation.bootstrap_difference
    retrieval=evaluation.retrieval_audit;rouge=rouge_scorer.RougeScorer.score
    def finding_cached(*args,**kwargs):return memo('finding_score',[args,kwargs],lambda:finding(*args,**kwargs))
    def bootstrap_cached(*args,**kwargs):return memo('patient_bootstrap',[args,kwargs],lambda:bootstrap(*args,**kwargs))
    def retrieval_cached(out,threshold=.5):
        guard();out=Path(out)
        return memo('retrieval_audit',dict(targets=sha(out/'targets.json'),rankings=sha(out/'rankings.json'),threshold=threshold),lambda:retrieval(out,threshold))
    def rouge_cached(self,target,prediction):
        # The existing evaluator constructs rougeL without stemming/custom tokenizers.
        return memo('rougeL',dict(target=target,prediction=prediction,use_stemmer=False),
                    lambda:rouge(self,target,prediction),
                    lambda v:{k:scoring.Score(*s) for k,s in v.items()})
    with contextlib.ExitStack() as stack:
        # Shared generation locks prevent evaluating files while a generator appends.
        for name,kind in [('evaluation_resume.lock',fcntl.LOCK_EX),('ecg_reports.lock',fcntl.LOCK_SH),('cxr_reports.lock',fcntl.LOCK_SH)]:
            lock=stack.enter_context((study/name).open('a'));fcntl.flock(lock,kind | fcntl.LOCK_NB)
        status_path=study/'evaluation_resume_status.json'
        atomic_json(status_path,dict(complete=False,state='running',checkpoint_dir=str(cache)))
        result=None;reason='completed'
        try:
            for name in ('SIGTERM','SIGUSR1'):
                sig=getattr(signal,name,None)
                if sig is not None:handlers[sig]=signal.getsignal(sig);signal.signal(sig,receive)
            with patch.object(evaluation,'score_findings',side_effect=finding_cached), \
                 patch.object(evaluation,'bootstrap_difference',side_effect=bootstrap_cached), \
                 patch.object(evaluation,'retrieval_audit',side_effect=retrieval_cached), \
                 patch.object(rouge_scorer.RougeScorer,'score',rouge_cached):
                guard()
                result=evaluation.evaluate(root,config,study,extra_texts=extra_texts,
                                            cxr_report_root=cxr_report_root,allow_partial=allow_partial)
        except _StopRequested as exc:reason=str(exc)
        except KeyboardInterrupt:reason='keyboard_interrupt'
        except Exception as exc:
            atomic_json(status_path,dict(complete=False,state='error',error=str(exc),checkpoint_dir=str(cache),**stats))
            raise
        finally:
            for sig,handler in handlers.items():signal.signal(sig,handler)
        status=dict(complete=result is not None,state='complete' if result is not None else 'stopped',
                    stop_reason=reason,checkpoint_dir=str(cache),stop_file=str(stop_file),
                    elapsed_hours=(time.monotonic()-start)/3600,max_run_hours=max_run_hours,
                    last_operation=active['operation'],**stats)
        atomic_json(status_path,status)
        print('Evaluation resume status:',status,flush=True)
        return dict(result or {},complete=result is not None,resume_status=status)
