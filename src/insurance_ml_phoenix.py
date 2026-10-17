"""Traditional ML cost evaluation with Phoenix tracing (no LLM judge).

Real OpenML data; a deliberately synthetic 25% outcome-cost inflation scenario.
This is a held-out simulation, not evidence of observed temporal/concept drift.
"""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import TweedieRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_tweedie_deviance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['Area','VehPower','VehAge','DrivAge','BonusMalus','VehBrand','VehGas','Density','Region']
CATEGORICAL = ['Area','VehBrand','VehGas','Region']
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]
PROJECT = 'insurance-traditional-ml'


def read_arff(path):
    columns=[]
    with path.open() as f:
        for skip,line in enumerate(f,1):
            if line.lower().startswith('@attribute'):
                columns.append(line.split()[1])
            if line.strip().lower()=='@data':
                break
        else:
            raise ValueError(f'No ARFF data section: {path}')
    return pd.read_csv(path, skiprows=skip, names=columns, quotechar="'")


def load_policies(folder):
    freq=read_arff(folder/'freMTPL2freq.arff')
    sev=read_arff(folder/'freMTPL2sev.arff')
    if freq.IDpol.duplicated().any():
        raise ValueError('Duplicate policy IDs in frequency data')
    if freq.IDpol.isna().any() or sev.IDpol.isna().any():
        raise ValueError('Missing policy ID')
    if sev.ClaimAmount.isna().any() or (sev.ClaimAmount<0).any():
        raise ValueError('Invalid severity amount')
    severity=sev.groupby('IDpol',as_index=False).ClaimAmount.sum()
    # One-to-one join after aggregating multiple claims: no claim-level leakage.
    policies=freq.merge(severity,on='IDpol',how='left',validate='one_to_one')
    policies['ClaimAmount']=policies.ClaimAmount.fillna(0)
    valid=policies.Exposure.notna() & policies.Exposure.gt(0)
    audit=dict(frequency_rows=len(freq),severity_rows=len(sev),
        severity_unique_policies=len(severity),severity_rows_beyond_first_claim=int(sev.IDpol.duplicated().sum()),
        unmatched_severity_policy_ids=int((~severity.IDpol.isin(freq.IDpol)).sum()),
        invalid_exposure_rows_removed=int((~valid).sum()),
        policies_with_positive_claim_count_but_no_positive_amount=int(((policies.ClaimNb>0)&(policies.ClaimAmount==0)).sum()),
        target_definition='Sum of recorded severity amounts for each policy, zero if no matched severity record; not guaranteed complete ultimate loss.')
    policies=policies.loc[valid].copy()
    if policies[FEATURES].isna().any().any():
        raise ValueError('Unexpected missing feature; inspect dataset before training')
    return policies,audit


def metrics(actual,predicted,exposure):
    a=np.asarray(actual,dtype=float);p=np.asarray(predicted,dtype=float);e=np.asarray(exposure,dtype=float)
    if not (np.isfinite(a).all() and np.isfinite(p).all() and np.isfinite(e).all()):
        raise ValueError('Non-finite metric inputs')
    return dict(policies=len(a),mae_eur=float(mean_absolute_error(a,p)),
        rmse_eur=float(np.sqrt(np.mean((a-p)**2))),
        mean_actual_eur=float(a.mean()),mean_predicted_eur=float(p.mean()),
        mean_underprediction_eur=float(np.mean(a-p)),
        actual_total_eur=float(a.sum()),predicted_total_eur=float(p.sum()),
        predicted_to_actual_ratio=float(p.sum()/a.sum()) if a.sum()>0 else None,
        exposure_weighted_tweedie_deviance=float(mean_tweedie_deviance(a/e,np.maximum(p/e,1e-9),power=1.5,sample_weight=e)))


def annotations(base, span_id, run_id, values):
    payload={'data':[dict(span_id=span_id,name=name,annotator_kind='CODE',identifier=run_id,
        result=dict(score=float(score),explanation=explanation),metadata={'run_id':run_id,'lower_is_better':True})
        for name,score,explanation in values]}
    for attempt in range(30):
        r=requests.post(base+'/v1/span_annotations?sync=true',json=payload,timeout=20)
        if r.status_code!=404:break
        time.sleep(.3)
    r.raise_for_status()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-size',type=int,default=150000)
    parser.add_argument('--inflation',type=float,default=1.25)
    parser.add_argument('--phoenix',default='http://localhost:6006')
    parser.add_argument('--offline',action='store_true',help='Compute reports without Phoenix')
    args=parser.parse_args()
    if args.sample_size<1000 or args.inflation<=1:
        parser.error('Use sample-size >=1000 and inflation >1')
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=ROOT/'outputs'/f'insurance-ml-{run_id}';out.mkdir(parents=True)
    if not args.offline:
        requests.get(args.phoenix+'/healthz',timeout=5).raise_for_status()
    provider=TracerProvider(resource=Resource.create({'service.name':'insurance-cost-model','openinference.project.name':PROJECT}))
    if not args.offline:
        provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=args.phoenix+'/v1/traces')))
    tracer=provider.get_tracer('insurance-ml-evaluation')
    try:
        with tracer.start_as_current_span('ml.evaluation_run') as run_span:
            run_span.set_attribute('run.id',run_id)
            run_span.set_attribute('model.type','traditional_ml_regularized_tweedie_regression')
            run_span.set_attribute('scenario.type','held_out_outcome_inflation_simulation')
            with tracer.start_as_current_span('ml.load_and_join'):
                policies,audit=load_policies(ROOT/'data/insurance')
                sampled=policies.sample(n=min(args.sample_size,len(policies)),random_state=42)
                train,test=train_test_split(sampled,test_size=.2,random_state=42)
                if set(train.IDpol)&set(test.IDpol):
                    raise AssertionError('Train/test policy leakage')
            print('Data audit:',json.dumps(audit),flush=True)
            print('Training policies:',len(train),'held-out policies:',len(test),flush=True)
            transform=ColumnTransformer([
                ('numeric',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),NUMERIC),
                ('categorical',OneHotEncoder(handle_unknown='ignore'),CATEGORICAL)],
                verbose_feature_names_out=False)
            model=make_pipeline(transform,TweedieRegressor(power=1.5,alpha=1.0,link='log',max_iter=300,tol=1e-6))
            with tracer.start_as_current_span('ml.fit'):
                # Rate target + exposure weights; ClaimNb/ClaimAmount/IDpol excluded from features.
                model.fit(train[FEATURES],train.ClaimAmount/train.Exposure,
                    tweedieregressor__sample_weight=train.Exposure)
            with tracer.start_as_current_span('ml.predict'):
                predicted_rate=np.maximum(model.predict(test[FEATURES]),1e-9)
                predicted=predicted_rate*test.Exposure.to_numpy()
            model_version='tweedie-p15-alpha1-v1-'+hashlib.sha256(','.join(map(str,sorted(train.IDpol))).encode()).hexdigest()[:12]
            mean_training_rate=float(train.ClaimAmount.sum()/train.Exposure.sum())
            dummy=mean_training_rate*test.Exposure.to_numpy()
            actual=test.ClaimAmount.to_numpy();exposure=test.Exposure.to_numpy()
            # Identical inputs and predictions in both scenarios; only outcomes change.
            batches={'baseline':actual,'simulated_inflation':actual*args.inflation}
            rows=[];prediction_rows=[];segment_rows=[];tracked_spans=[]
            rng=np.random.default_rng(42)
            selected=np.unique(np.r_[rng.choice(len(test),size=min(15,len(test)),replace=False),np.argsort(actual)[-5:]])
            drift={c:{'ks_statistic':0.0,'inputs_identical_by_construction':True} for c in FEATURES}
            drift['claim_amount']={'ks_statistic':float(ks_2samp(actual,batches['simulated_inflation']).statistic),
                'simulation_multiplier':args.inflation,'description':'Outcome distribution comparison, not feature drift; not a formal significance test.'}
            for batch_name,truth in batches.items():
                m=metrics(truth,predicted,exposure);dummy_metrics=metrics(truth,dummy,exposure)
                rows.append({'batch':batch_name,**m,'dummy_mae_eur':dummy_metrics['mae_eur']})
                with tracer.start_as_current_span('ml.batch.'+batch_name) as span:
                    span.set_attribute('run.id',run_id);span.set_attribute('batch.name',batch_name)
                    span.set_attribute('model.version',model_version)
                    span.set_attribute('simulation',batch_name!='baseline')
                    span.set_attribute('scenario.inflation_multiplier',args.inflation if batch_name!='baseline' else 1.0)
                    span.set_attribute('metrics.json',json.dumps(m))
                    span.set_attribute('input_drift','none: identical features and exposure')
                    for key,value in m.items():
                        if value is not None:span.set_attribute('metrics.'+key,value)
                    for i in selected:
                        with tracer.start_as_current_span('ml.predict_policy') as child:
                            child.set_attribute('run.id',run_id);child.set_attribute('batch.name',batch_name)
                            child.set_attribute('input.value',json.dumps({k:test.iloc[i][k] for k in FEATURES},default=str))
                            child.set_attribute('output.value',str(float(predicted[i])))
                            child.set_attribute('policy.id',str(int(test.iloc[i].IDpol)))
                            child.set_attribute('actual.cost_eur',float(truth[i]))
                            child.set_attribute('absolute_error_eur',float(abs(truth[i]-predicted[i])))
                            child.set_attribute('sample.selection','15 random policies plus 5 highest actual baseline losses; not representative')
                            tracked_spans.append((f'{child.get_span_context().span_id:016x}',float(abs(truth[i]-predicted[i]))))
                    sid=f'{span.get_span_context().span_id:016x}'
                if not args.offline:
                    annotations(args.phoenix,sid,run_id,[('mae_eur',m['mae_eur'],'Mean absolute error in EUR; lower is better, not a binary accuracy score.'),
                        ('rmse_eur',m['rmse_eur'],'Root mean squared error in EUR; lower is better.'),
                        ('tweedie_deviance',m['exposure_weighted_tweedie_deviance'],'Exposure-weighted Tweedie deviance, power 1.5; lower is better.')])
                pframe=pd.DataFrame({'batch':batch_name,'policy_id':test.IDpol.to_numpy(),
                    'exposure':exposure,'actual_cost_eur':truth,'predicted_cost_eur':predicted,
                    'absolute_error_eur':abs(truth-predicted),'dummy_predicted_cost_eur':dummy})
                prediction_rows.append(pframe)
                for column in ['VehAge','Region']:
                    grouping=pd.cut(test.VehAge,[-1,2,5,10,20,np.inf]).astype(str).to_numpy() if column=='VehAge' else test.Region.to_numpy()
                    for group in sorted(set(grouping)):
                        mask=grouping==group
                        segment_rows.append({'batch':batch_name,'feature':column,'segment':str(group),**metrics(truth[mask],predicted[mask],exposure[mask])})
                print(batch_name,json.dumps(m),flush=True)
            if not args.offline:
                for sid,error in tracked_spans:
                    annotations(args.phoenix,sid,run_id,[('absolute_error_eur',error,'Absolute prediction error in EUR from observed or simulated outcome; lower is better.')])
            pd.DataFrame(rows).to_csv(out/'batch_metrics.csv',index=False)
            pd.concat(prediction_rows).to_csv(out/'held_out_predictions.csv',index=False)
            pd.DataFrame(segment_rows).to_csv(out/'segment_metrics.csv',index=False)
            summary=dict(run_id=run_id,project=PROJECT,model_version=model_version,
                training_policies=len(train),test_policies=len(test),features=FEATURES,
                inflation=args.inflation,baseline_constant_annual_rate=mean_training_rate,
                data_audit=audit,batches=rows,drift=drift,
                limitations=['Historical French motor third-party liability data, not USAA data or modern repair estimates.',
                    'Random policy holdout, not a temporal backtest; train and test IDs are disjoint.',
                    'Severity records summed per policy; unmatched severity is audited; absent severity becomes zero.',
                    'Outcome inflation is an intentional simulation, not an observed drift finding.',
                    'Model trained on a reproducible sample to fit laptop resources; no hyperparameter search.',
                    'Trace samples intentionally include large losses and are not representative.',
                    'Metrics computed in Python and displayed as Phoenix CODE annotations; no native Phoenix drift engine or LLM judge.'])
            (out/'summary.json').write_text(json.dumps(summary,indent=2))
            base,infl=rows
            report=f'''# Traditional ML insurance evaluation\n\nRun: {run_id}. Phoenix project: `{PROJECT}`.\n\nModel: regularized Tweedie regression (power 1.5, log link, alpha 1.0). It predicts annual cost, converted to policy-period cost using exposure. Train: {len(train):,} policies; held-out test: {len(test):,} policies. Claim counts and amounts are not input features.\n\n| Metric | Baseline | Simulated {args.inflation:.2f}x costs |\n|---|---:|---:|\n| MAE (EUR) | {base['mae_eur']:.2f} | {infl['mae_eur']:.2f} |\n| RMSE (EUR) | {base['rmse_eur']:.2f} | {infl['rmse_eur']:.2f} |\n| Mean actual cost (EUR) | {base['mean_actual_eur']:.2f} | {infl['mean_actual_eur']:.2f} |\n| Mean predicted cost (EUR) | {base['mean_predicted_eur']:.2f} | {infl['mean_predicted_eur']:.2f} |\n| Mean actual minus prediction (EUR) | {base['mean_underprediction_eur']:.2f} | {infl['mean_underprediction_eur']:.2f} |\n| Predicted / actual total | {base['predicted_to_actual_ratio']:.3f} | {infl['predicted_to_actual_ratio']:.3f} |\n| Tweedie deviance | {base['exposure_weighted_tweedie_deviance']:.2f} | {infl['exposure_weighted_tweedie_deviance']:.2f} |\n| Constant-rate baseline MAE (EUR) | {base['dummy_mae_eur']:.2f} | {infl['dummy_mae_eur']:.2f} |\n\n## Interpretation\n\nInputs, exposure and predictions are identical across the two batches. Only actual claim amounts are multiplied by {args.inflation}. Therefore input drift is zero by construction, while outcome costs and prediction error can change. This illustrates why input drift alone cannot establish model quality. Metrics are calculated from outcomes, without an LLM judge. It does not show a real-world change over time.\n\nA constant-rate comparator is included; this run does not establish that Tweedie regression is the best model. MAE alone favors low predictions in sparse claims, so inspect aggregate calibration and deviance too. Segment metrics are descriptive, not causal or fairness conclusions.\n\n## Data and limitations\n\n'''+ '\n'.join('- '+s for s in summary['limitations'])+'\n\nSource: https://www.openml.org/d/41214 and https://www.openml.org/d/41215; CC0 according to the downloaded metadata. See data/insurance/manifest.json.\n'
            (out/'report.md').write_text(report)
            print('RESULT_DIR',out,flush=True)
    finally:
        provider.shutdown()

if __name__=='__main__':
    main()
